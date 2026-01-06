use candle_core::{Device, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, Linear, Module, VarBuilder};
use realfft::RealFftPlanner;
use rustfft::num_complex::Complex;
use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct VocosConfig {
    pub num_input_channels: usize,
    pub decoder_num_layers: usize,
    pub decoder_dim: usize,
    pub decoder_intermediate_dim: usize,
    pub hop_length: usize,
    pub n_fft: usize,
    pub upscale: usize,
    pub dw_kernel: usize,
}

pub struct ConvNeXtBlock {
    dwconv: Conv1d,
    norm: LayerNorm,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Option<Tensor>,
}

impl ConvNeXtBlock {
    pub fn new(
        dim: usize,
        intermediate_dim: usize,
        layer_scale_init_value: Option<f64>,
        dw_kernel_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let dwconv_cfg = Conv1dConfig {
            padding: dw_kernel_size / 2,
            groups: dim,
            ..Default::default()
        };
        let dwconv = candle_nn::conv1d(dim, dim, dw_kernel_size, dwconv_cfg, vb.pp("dwconv"))?;
        let norm = candle_nn::layer_norm(dim, 1e-6, vb.pp("norm"))?;
        let pwconv1 = candle_nn::linear(dim, intermediate_dim, vb.pp("pwconv1"))?;
        let pwconv2 = candle_nn::linear(intermediate_dim, dim, vb.pp("pwconv2"))?;

        let gamma = if let Some(init_value) = layer_scale_init_value {
            if init_value > 0.0 {
                if vb.contains_tensor("gamma") {
                    Some(vb.get(dim, "gamma")?)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            if vb.contains_tensor("gamma") {
                Some(vb.get(dim, "gamma")?)
            } else {
                None
            }
        };

        Ok(Self {
            dwconv,
            norm,
            pwconv1,
            pwconv2,
            gamma,
        })
    }
}

impl Module for ConvNeXtBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.dwconv.forward(x)?;
        let x = x.transpose(1, 2)?;
        let x = self.norm.forward(&x)?;
        let x = self.pwconv1.forward(&x)?;
        let x = x.gelu()?;
        let x = self.pwconv2.forward(&x)?;

        let x = if let Some(gamma) = &self.gamma {
            x.broadcast_mul(gamma)?
        } else {
            x
        };

        let x = x.transpose(1, 2)?;
        residual + x
    }
}

pub struct VocosBackbone {
    pub embed: Conv1d,
    pub norm: LayerNorm,
    pub convnext: Vec<ConvNeXtBlock>,
    pub final_layer_norm: LayerNorm,
}

impl VocosBackbone {
    pub fn new(cfg: &VocosConfig, vb: VarBuilder) -> Result<Self> {
        let input_channels = cfg.num_input_channels;
        let dim = cfg.decoder_dim;
        let intermediate_dim = cfg.decoder_intermediate_dim;
        let num_layers = cfg.decoder_num_layers;
        let dw_kernel_size = cfg.dw_kernel;
        let input_kernel_size = dw_kernel_size;

        let layer_scale_init_value = 1.0 / (num_layers as f64).sqrt();

        let embed_cfg = Conv1dConfig {
            padding: input_kernel_size / 2,
            ..Default::default()
        };
        let embed = candle_nn::conv1d(
            input_channels,
            dim,
            input_kernel_size,
            embed_cfg,
            vb.pp("embed"),
        )?;
        let norm = candle_nn::layer_norm(dim, 1e-6, vb.pp("norm"))?;

        let mut convnext = Vec::new();
        let vb_conv = vb.pp("convnext");
        for i in 0..num_layers {
            let block = ConvNeXtBlock::new(
                dim,
                intermediate_dim,
                Some(layer_scale_init_value),
                dw_kernel_size,
                vb_conv.pp(i),
            )?;
            convnext.push(block);
        }

        let final_layer_norm = candle_nn::layer_norm(dim, 1e-6, vb.pp("final_layer_norm"))?;

        Ok(Self {
            embed,
            norm,
            convnext,
            final_layer_norm,
        })
    }
}

impl Module for VocosBackbone {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.embed.forward(x)?;
        x = x.transpose(1, 2)?;
        x = self.norm.forward(&x)?;
        x = x.transpose(1, 2)?;

        for block in &self.convnext {
            x = block.forward(&x)?;
        }

        x = x.transpose(1, 2)?;
        x = self.final_layer_norm.forward(&x)?;
        x = x.transpose(1, 2)?;
        Ok(x)
    }
}

pub struct ISTFTHead {
    pub out: Linear,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub padding: String,
    pub window: Tensor,
}

impl ISTFTHead {
    pub fn new(cfg: &VocosConfig, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.decoder_dim;
        let n_fft = cfg.n_fft;
        let hop_length = cfg.hop_length;
        let win_length = n_fft;
        let padding = "center".to_string();

        let out_dim = n_fft + 2;
        let out = candle_nn::linear(dim, out_dim, vb.pp("out"))?;
        let window = hann_window(win_length, vb.device())?;

        Ok(Self {
            out,
            n_fft,
            hop_length,
            win_length,
            padding,
            window,
        })
    }
}

pub fn hann_window(size: usize, device: &Device) -> Result<Tensor> {
    let mut data = Vec::with_capacity(size);
    for i in 0..size {
        let n = i as f64;
        let m = size as f64;
        let val = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * n / m).cos());
        data.push(val as f32);
    }
    Tensor::from_vec(data, (size,), device)
}

pub struct SopranoDecoder {
    pub decoder: VocosBackbone,
    pub head: ISTFTHead,
    pub upscale: usize,
}

impl SopranoDecoder {
    pub fn new(cfg: &VocosConfig, vb: VarBuilder) -> Result<Self> {
        let decoder = VocosBackbone::new(cfg, vb.pp("decoder"))?;
        let head = ISTFTHead::new(cfg, vb.pp("head"))?;
        let upscale = cfg.upscale;

        Ok(Self {
            decoder,
            head,
            upscale,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Vec<f32>> {
        let x_interp = interpolate_linear(x, self.upscale)?;
        let x_decoded = self.decoder.forward(&x_interp)?;

        // Head
        let x_head = self.head.out.forward(&x_decoded.transpose(1, 2)?)?;
        let x_head = x_head.transpose(1, 2)?;

        let (mag, p) = {
            let chunks = x_head.chunk(2, 1)?;
            (chunks[0].clone(), chunks[1].clone())
        };
        let mag = mag.exp()?;
        let mag = mag.clamp(f32::NEG_INFINITY, 100.0)?;

        let cos_p = p.cos()?;
        let sin_p = p.sin()?;

        let re = (mag.clone() * cos_p)?;
        let im = (mag * sin_p)?;

        // ISTFT using realfft
        let (_, num_freqs, t_out) = re.dims3()?;
        let re_vec = re.flatten_all()?.to_vec1::<f32>()?;
        let im_vec = im.flatten_all()?.to_vec1::<f32>()?;

        let n_fft = self.head.n_fft;
        let mut planner: RealFftPlanner<f32> = RealFftPlanner::new();
        let irfft = planner.plan_fft_inverse(n_fft);

        let mut all_frames: Vec<f32> = Vec::with_capacity(t_out * n_fft);
        let window_vec = self.head.window.to_vec1::<f32>()?;

        for t in 0..t_out {
            let mut spectrum: Vec<Complex<f32>> = Vec::with_capacity(num_freqs);
            for f in 0..num_freqs {
                let idx = f * t_out + t;
                spectrum.push(Complex::new(re_vec[idx], im_vec[idx]));
            }

            // Zero out DC and Nyquist bins (matching Python's spectral_ops)
            spectrum[0] = Complex::new(0.0, 0.0);
            if num_freqs > 0 {
                spectrum[num_freqs - 1] = Complex::new(0.0, 0.0);
            }

            let mut output = irfft.make_output_vec();
            irfft
                .process(&mut spectrum, &mut output)
                .expect("IRFFT failed");

            let scale = 1.0 / n_fft as f32;

            for (i, &sample) in output.iter().enumerate() {
                all_frames.push(sample * scale * window_vec[i]);
            }
        }

        let ifft_vec = all_frames;
        let hop_length = self.head.hop_length;
        let win_length = self.head.win_length;
        let n_frames = t_out;

        let output_size = (n_frames - 1) * hop_length + win_length;
        let pad = if self.head.padding == "center" {
            self.head.n_fft / 2
        } else {
            (win_length - hop_length) / 2
        };

        let mut output_audio = vec![0.0f32; output_size];
        let mut envelope = vec![0.0f32; output_size];
        let window_sq_vec: Vec<f32> = window_vec.iter().map(|x| x * x).collect();

        for i in 0..n_frames {
            let start = i * hop_length;
            let frame_start = i * win_length;

            for j in 0..win_length {
                if start + j < output_size {
                    output_audio[start + j] += ifft_vec[frame_start + j];
                    envelope[start + j] += window_sq_vec[j];
                }
            }
        }

        let end_idx = if output_size > pad {
            output_size - pad
        } else {
            output_size
        };
        let start_idx = pad.min(end_idx);
        let mut result = Vec::with_capacity(end_idx - start_idx);

        for i in start_idx..end_idx {
            if envelope[i] > 1e-11 {
                result.push(output_audio[i] / envelope[i]);
            } else {
                result.push(output_audio[i]);
            }
        }

        Ok(result)
    }
}

pub fn interpolate_linear(x: &Tensor, scale_factor: usize) -> Result<Tensor> {
    let (_b, _c, t) = x.dims3()?;
    let new_t = scale_factor * (t - 1) + 1;
    let device = x.device();

    let mut indices_floor = Vec::with_capacity(new_t);
    let mut indices_ceil = Vec::with_capacity(new_t);
    let mut alphas = Vec::with_capacity(new_t);

    for i in 0..new_t {
        let p = i as f32 / scale_factor as f32;
        let p_floor = p.floor() as i64;
        let p_ceil = p.ceil() as i64;

        let p_floor = p_floor.clamp(0, (t - 1) as i64);
        let p_ceil = p_ceil.clamp(0, (t - 1) as i64);

        let alpha = p - p.floor();

        indices_floor.push(p_floor);
        indices_ceil.push(p_ceil);
        alphas.push(alpha);
    }

    let idx_floor = Tensor::from_vec(indices_floor, (new_t,), device)?;
    let idx_ceil = Tensor::from_vec(indices_ceil, (new_t,), device)?;
    let alpha = Tensor::from_vec(alphas, (1, 1, new_t), device)?;

    let x_floor = x.index_select(&idx_floor, 2)?;
    let x_ceil = x.index_select(&idx_ceil, 2)?;

    let one_minus_alpha = (1.0 - alpha.clone())?;
    let res = (x_floor.broadcast_mul(&one_minus_alpha)? + x_ceil.broadcast_mul(&alpha)?)?;

    Ok(res)
}
