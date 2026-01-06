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

struct ConvNeXtBlock {
    dwconv: Conv1d,
    norm: LayerNorm,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Option<Tensor>,
}

impl ConvNeXtBlock {
    fn new(
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
                // In PyTorch: nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
                // In safetensors, this parameter is stored as "gamma".
                // If it exists in the weights, we load it.
                // However, the logic in python checks init_value > 0 to decide whether to create it.
                // We should check if "gamma" exists in vb.
                if vb.contains_tensor("gamma") {
                    Some(vb.get(dim, "gamma")?)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            // If layer_scale_init_value is None, check if gamma exists anyway (maybe passed differently or default behaviour)
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
        // x: (B, C, T) -> (B, T, C)
        let x = x.transpose(1, 2)?;
        let x = self.norm.forward(&x)?;
        let x = self.pwconv1.forward(&x)?;
        let x = x.gelu()?;
        let x = self.pwconv2.forward(&x)?;

        let x = if let Some(gamma) = &self.gamma {
            // gamma is (C), x is (B, T, C)
            // broadcasting happens on last dim automatically?
            // Candle broadcasting: if shapes match from right.
            // gamma: (C), x: (B, T, C). Yes.
            x.broadcast_mul(gamma)?
        } else {
            x
        };

        let x = x.transpose(1, 2)?; // (B, C, T)

        // Residual connection
        residual + x
    }
}

struct VocosBackbone {
    embed: Conv1d,
    norm: LayerNorm,
    convnext: Vec<ConvNeXtBlock>,
    final_layer_norm: LayerNorm,
}

impl VocosBackbone {
    fn new(cfg: &VocosConfig, vb: VarBuilder) -> Result<Self> {
        let input_channels = cfg.num_input_channels;
        let dim = cfg.decoder_dim;
        let intermediate_dim = cfg.decoder_intermediate_dim;
        let num_layers = cfg.decoder_num_layers;
        let dw_kernel_size = cfg.dw_kernel;
        let input_kernel_size = dw_kernel_size;

        // Python:
        // layer_scale_init_value or 1 / num_layers**0.5
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

struct ISTFTHead {
    out: Linear,
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    padding: String,
    window: Tensor,
}

impl ISTFTHead {
    fn new(cfg: &VocosConfig, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.decoder_dim;
        let n_fft = cfg.n_fft;
        let hop_length = cfg.hop_length;
        let win_length = n_fft; // In Python code: win_length=n_fft
        let padding = "center".to_string(); // Python default is "center"

        let out_dim = n_fft + 2;
        let out = candle_nn::linear(dim, out_dim, vb.pp("out"))?;

        // Window: torch.hann_window(win_length)
        // We need to generate hann window.
        // Formula: 0.5 * (1 - cos(2 * pi * n / (M - 1)))
        // 0 <= n <= M-1
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

fn hann_window(size: usize, device: &Device) -> Result<Tensor> {
    let mut data = Vec::with_capacity(size);
    for i in 0..size {
        let n = i as f64;
        let m = size as f64;
        let val = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * n / m).cos()); // Standard Hann window
                                                                            // Note: PyTorch hann_window might be periodic=True by default which uses M instead of M-1?
                                                                            // torch.hann_window(window_length, periodic=True, *)
                                                                            // If periodic=True (default), formula uses N. If periodic=False, uses N-1.
                                                                            // STFT usually uses periodic window.
        data.push(val as f32);
    }
    Tensor::from_vec(data, (size,), device)
}

pub struct SopranoDecoder {
    decoder: VocosBackbone,
    head: ISTFTHead,
    upscale: usize,
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
        // x: (B, C, L)
        let (_b, _c, _t) = x.dims3()?;
        println!("DEBUG_RUST_INPUT shape: {:?}", x.dims());
        let x_flat = x.flatten_all()?.to_vec1::<f32>()?;
        println!(
            "DEBUG_RUST_INPUT tail: {:?}",
            &x_flat[x_flat.len().saturating_sub(10)..]
        );

        // Interpolate
        let x_interp = interpolate_linear(x, self.upscale)?;
        println!("DEBUG_RUST_INTERP shape: {:?}", x_interp.dims());
        let x_interp_flat = x_interp.flatten_all()?.to_vec1::<f32>()?;
        println!(
            "DEBUG_RUST_INTERP tail: {:?}",
            &x_interp_flat[x_interp_flat.len().saturating_sub(10)..]
        );

        let x_decoded = self.decoder.forward(&x_interp)?;
        println!("DEBUG_RUST_BACKBONE shape: {:?}", x_decoded.dims());
        let x_decoded_flat = x_decoded.flatten_all()?.to_vec1::<f32>()?;
        println!(
            "DEBUG_RUST_BACKBONE tail: {:?}",
            &x_decoded_flat[x_decoded_flat.len().saturating_sub(10)..]
        );

        // Head
        let x_head = self.head.out.forward(&x_decoded.transpose(1, 2)?)?; // (B, T, out_dim)
        let x_head = x_head.transpose(1, 2)?; // (B, out_dim, T)

        let chunks = x_head.chunk(2, 1)?;
        let mag = &chunks[0];
        let p = &chunks[1];
        let mag = mag.exp()?;
        let mag = mag.clamp(f32::NEG_INFINITY, 100.0)?;

        println!("DEBUG_RUST_MAG shape: {:?}", mag.dims());
        let mag_flat = mag.flatten_all()?.to_vec1::<f32>()?;
        println!(
            "DEBUG_RUST_MAG tail: {:?}",
            &mag_flat[mag_flat.len().saturating_sub(10)..]
        );

        println!("DEBUG_RUST_PHASE shape: {:?}", p.dims());
        let p_flat = p.flatten_all()?.to_vec1::<f32>()?;
        println!(
            "DEBUG_RUST_PHASE tail: {:?}",
            &p_flat[p_flat.len().saturating_sub(10)..]
        );

        let cos_p = p.cos()?;
        let sin_p = p.sin()?;

        let re = (mag.clone() * cos_p)?;
        let im = (mag * sin_p)?;

        // ISTFT using realfft
        // re, im: (B, F, T) where F = n_fft/2 + 1
        let (_, num_freqs, t_out) = re.dims3()?;

        // Convert tensors to Vec for realfft processing
        let re_vec = re.flatten_all()?.to_vec1::<f32>()?;
        let im_vec = im.flatten_all()?.to_vec1::<f32>()?;

        // Set up realfft planner
        let n_fft = self.head.n_fft;
        let mut planner: RealFftPlanner<f32> = RealFftPlanner::new();
        let irfft = planner.plan_fft_inverse(n_fft);

        // Process each frame
        let mut all_frames: Vec<f32> = Vec::with_capacity(t_out * n_fft);
        let window_vec = self.head.window.to_vec1::<f32>()?;

        for t in 0..t_out {
            // Build complex spectrum for this frame
            let mut spectrum: Vec<Complex<f32>> = Vec::with_capacity(num_freqs);
            for f in 0..num_freqs {
                let idx = f * t_out + t; // (F, T) layout after flattening
                spectrum.push(Complex::new(re_vec[idx], im_vec[idx]));
            }

            // Zero out imaginary parts of DC and Nyquist bins to satisfy realfft
            spectrum[0].im = 0.0;
            if num_freqs > 0 {
                spectrum[num_freqs - 1].im = 0.0;
            }

            // Perform IRFFT - output is real-valued time signal
            let mut output = irfft.make_output_vec(); // n_fft length
            irfft
                .process(&mut spectrum, &mut output)
                .expect("IRFFT failed");

            let scale = 1.0 / n_fft as f32;

            // Apply window and collect
            for (i, &sample) in output.iter().enumerate() {
                all_frames.push(sample * scale * window_vec[i]);
            }

            if t == 0 {
                println!(
                    "DEBUG_RUST_IRFFT_FRAME0_SCALED tail: {:?}",
                    &output.iter().map(|&s| s * scale).collect::<Vec<f32>>()[..10]
                );
            }
        }

        // Now all_frames contains all windowed frames as [frame0[0..n_fft], frame1[0..n_fft], ...]
        let ifft_vec = all_frames;
        let hop_length = self.head.hop_length;
        let win_length = self.head.win_length;
        let n_frames = t_out;
        let window = &self.head.window;

        // Overlap Add

        // Calculate output size
        let output_size = (n_frames - 1) * hop_length + win_length;

        // For "center" padding (Python default), pad = n_fft // 2
        // For "same" padding, pad = (win_length - hop_length) // 2
        let pad = if self.head.padding == "center" {
            self.head.n_fft / 2
        } else {
            (win_length - hop_length) / 2
        };

        let mut output_audio = vec![0.0f32; output_size];
        let mut envelope = vec![0.0f32; output_size];
        let window_vec = window.to_vec1::<f32>()?;
        let window_sq_vec: Vec<f32> = window_vec.iter().map(|x| x * x).collect();

        for i in 0..n_frames {
            let start = i * hop_length;
            let frame_start = i * win_length; // in ifft_vec

            for j in 0..win_length {
                if start + j < output_size {
                    output_audio[start + j] += ifft_vec[frame_start + j];
                    envelope[start + j] += window_sq_vec[j];
                }
            }
        }

        // Normalize and trim padding
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

        let audio_len = result.len();
        println!("DEBUG_RUST_FINAL_AUDIO shape: [{}]", audio_len);
        println!(
            "DEBUG_RUST_FINAL_AUDIO tail: {:?}",
            &result[audio_len.saturating_sub(10)..]
        );

        Ok(result)
    }
}

fn interpolate_linear(x: &Tensor, scale_factor: usize) -> Result<Tensor> {
    // x: (B, C, T)
    let (b, c, t) = x.dims3()?;
    let new_t = scale_factor * (t - 1) + 1;

    // We can use conv1d_transpose
    // Weight should be fixed to perform linear interpolation.
    // Kernel size = 2 * scale_factor - 1 ?
    // Stride = scale_factor.
    // Padding = scale_factor - 1.
    // Groups = C.

    // Actually, simpler approach given we don't need gradients:
    // Create new tensor of size (B, C, new_t)
    // Fill it.
    // Since we are in Candle, we want to avoid loops over tensor elements.

    // Let's try to use `upsample_nearest` followed by smoothing? No.

    // Use `gather`.
    // Generate indices.
    // For each output index i:
    // input position p = i / scale_factor
    // alpha = i % scale_factor / scale_factor
    // val = x[floor(p)] * (1-alpha) + x[ceil(p)] * alpha

    // Construct index tensors.
    let device = x.device();
    let mut indices_floor = Vec::with_capacity(new_t);
    let mut indices_ceil = Vec::with_capacity(new_t);
    let mut alphas = Vec::with_capacity(new_t);

    for i in 0..new_t {
        let p = i as f32 / scale_factor as f32;
        let p_floor = p.floor() as i64;
        let p_ceil = p.ceil() as i64;

        // Clamp indices to [0, t-1]
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

    // Gather on dim 2
    let x_floor = x.index_select(&idx_floor, 2)?;
    let x_ceil = x.index_select(&idx_ceil, 2)?;

    let one_minus_alpha = (1.0 - alpha.clone())?;

    let res = (x_floor.broadcast_mul(&one_minus_alpha)? + x_ceil.broadcast_mul(&alpha)?)?;

    Ok(res)
}
