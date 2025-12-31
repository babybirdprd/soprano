use candle_core::{Device, Result, Tensor, DType};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, Linear, Module, VarBuilder};
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
        let embed = candle_nn::conv1d(input_channels, dim, input_kernel_size, embed_cfg, vb.pp("embed"))?;
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
        let padding = "same".to_string();

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

    fn istft(&self, real: &Tensor, imag: &Tensor) -> Result<Tensor> {
        // real, imag: (B, N, T) where N = n_fft/2 + 1 usually for rfft, but here input is full spectrum?
        // Wait, Python code says:
        // ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        // spec is complex.

        // In python:
        // x = self.out(x.transpose(1,2)).transpose(1, 2)
        // mag, p = x.chunk(2, dim=1)
        // ...
        // S = mag * (x + 1j * y) -> S shape (B, N_FFT/2 + 1, T) usually?

        // In Python ISTFTHead:
        // out_dim = n_fft + 2.
        // mag, p = x.chunk(2, dim=1) -> each has (n_fft/2 + 1) channels?
        // No, n_fft + 2 split by 2 is n_fft/2 + 1.
        // So S has n_fft/2 + 1 frequency bins.
        // irfft expects this for real output.

        // Candle doesn't have irfft easily accessible or "fold".
        // We need to implement irfft using inverse dft if possible, or simple IDFT if N is small (2048 is not small).
        // BUT candle-core might not have FFT operations exposed.
        // I need to check if I can use a crate or if I have to implement it.
        // Since I cannot easily add new crates that are not in the environment (unless I can compile them),
        // I should check what's available. `candle-core` doesn't seem to have FFT.
        // Actually, maybe I can use `realfft` crate? Or is it pure rust?
        // Wait, I am allowed to add dependencies in Cargo.toml.
        // But `candle` is GPU accelerated. If I use CPU FFT, I need to move data back and forth.
        // If I run on CPU, it's fine.

        // Let's assume we can use a rust FFT library. `rustfft`.
        // But we need to handle batching.

        // Alternative: Implementing naive IDFT as matrix multiplication?
        // O(N^2). 2048^2 = 4M ops per column. Might be slow but works on GPU if implemented as matmul.
        // We need IDFT matrix.
        // IDFT(k, n) = 1/N * exp(i * 2*pi * k * n / N)
        // Since we are doing irfft (inverse real fft), the input is half spectrum.
        // Reconstructing full spectrum and then IDFT?

        // Constructing IDFT matrix:
        // W[n, k] = exp(i * 2*pi * n * k / N)
        // We want x[n] = 1/N * sum_k X[k] * W[n, k]
        // Since output is real, we can optimize.

        // Let's implement it using matrix multiplication for now.
        // It might be memory intensive if batch size or sequence length is large.
        // B, F, T.
        // We operate on F dim.
        // We need (N, F) matrix.
        // Output (B, N, T).

        // Steps:
        // 1. Reconstruct full complex spectrum from half spectrum (S).
        //    S: (B, N_freq, T). N_freq = n_fft/2 + 1.
        //    Full spectrum S_full: (B, n_fft, T).
        //    S_full[0] = S[0]
        //    S_full[k] = S[k] for 1 <= k < N/2
        //    S_full[N/2] = S[N/2] (Nyquist)
        //    S_full[N-k] = conj(S[k])

        // 2. IDFT via matrix mult.
        //    x = (IDFT_matrix @ S_full).real?
        //    Actually, we can do it with real arithmetic to avoid complex numbers if Candle doesn't support them well.
        //    x[n] = 1/N * ( X[0] + X[N/2]*cos(pi*n) + 2*sum_{k=1}^{N/2-1} (Re[X[k]]*cos(2pi*k*n/N) - Im[X[k]]*sin(2pi*k*n/N)) )

        //    So we can create a large weight matrix for this transformation.
        //    Input: Real and Imag parts of S (freqs 0 to N/2).
        //    Input dim: 2 * (N/2 + 1) = N + 2.
        //    Output dim: N.
        //    We can precompute a linear layer that maps (Re, Im) -> Time domain signal window.

        //    Wait, this is exactly what `torch.fft.irfft` does efficiently.
        //    As a linear operation, it is just a matrix multiplication.
        //    Weight matrix W of shape (N, N+2).
        //    We can precompute this W.

        //    This avoids using external FFT libraries and works on GPU with Candle!

        //    Let's precompute the IDFT matrix in `new`.

        let device = real.device();
        let (b_size, _, t_size) = real.dims3()?;

        // Combine real and imag into one tensor for matmul
        // We expect real: (B, F, T), imag: (B, F, T) where F = n_fft/2 + 1
        // We want input (B, T, 2*F) -> Output (B, T, n_fft)

        let real_t = real.transpose(1, 2)?; // (B, T, F)
        let imag_t = imag.transpose(1, 2)?; // (B, T, F)

        let input = Tensor::cat(&[&real_t, &imag_t], 2)?; // (B, T, 2F)

        // We need a stored IDFT matrix.
        // It should be registered as a buffer or just a tensor in the struct.
        // Let's assume we have `self.idft_matrix` of shape (2F, n_fft).
        // Then input @ idft_matrix -> (B, T, n_fft).

        // Wait, typical Linear is x @ w.t() + b.
        // So if idft_matrix is (n_fft, 2F), we can use linear or matmul.

        // Need to define idft_matrix in struct and `new`.

        // 3. Overlap-Add (Fold).
        //    Candle doesn't have `fold`.
        //    We have frames (B, T, n_fft).
        //    We need to overlap-add them.
        //    Output length L approx T * hop_length.
        //    We can manually implement this by creating a large zero tensor and adding slices.
        //    But modifying tensors in place is tricky in Candle (immutable mostly?).
        //    Actually, we can use `scatter_add` if available, or just a loop if T is not too huge.
        //    Or we can reshape and sum if we can construct the structure correctly.

        //    Let's look at `col2im` or similar? No.

        //    Looping over T frames might be slow in Rust/Candle graph, but maybe okay for inference.
        //    However, T can be ~100-1000.
        //    Can we vectorize?

        //    We can use the "reshape and sum" trick for constant overlap-add.
        //    If `n_fft` is multiple of `hop_length` (e.g. 2048 and 512 -> 4x overlap).
        //    We can arrange the data into 4 streams and sum them.
        //
        //    Frames: 0, 1, 2, 3, 4, 5...
        //    Stream 0: Frame 0, 4, 8... (padded with zeros in between?)
        //    This seems complicated to implement generally.

        //    Let's stick to a loop for now, or see if we can optimize.
        //    Since we are generating audio, T is length of tokens.
        //    For 10 seconds of audio, T ~ 10s * 24000sr / 512hop ~ 468 frames.
        //    Looping 500 times is fine in Rust.

        Err(candle_core::Error::Msg("Not implemented".into()))
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

fn make_idft_matrix(n_fft: usize, device: &Device) -> Result<Tensor> {
    // Construct matrix M of shape (2*(n_fft/2+1), n_fft)
    // such that [Re(S); Im(S)] @ M = x (time domain)
    // x[n] = 1/N * sum_{k=0}^{N-1} X[k] exp(i 2pi k n / N)
    // X[k] comes from Re, Im.
    // X[k] = Re[k] + i Im[k]
    // X[N-k] = Re[k] - i Im[k]

    // x[n] = 1/N * [ X[0] + X[N/2]cos(pi n) + sum_{k=1}^{N/2-1} (Re[k] + i Im[k])(cos(...) + i sin(...)) + (Re[k] - i Im[k])(cos(...) - i sin(...)) ]
    // ... expansion ...
    // x[n] = 1/N * [ Re[0] + Re[N/2]cos(pi n) + 2 sum_{k=1}^{N/2-1} (Re[k]cos(2pi k n / N) - Im[k]sin(2pi k n / N)) ]

    // So coeffs for Re[0]: 1/N
    // Coeffs for Re[N/2]: 1/N * cos(pi n)
    // Coeffs for Re[k]: 2/N * cos(2pi k n / N)
    // Coeffs for Im[k]: -2/N * sin(2pi k n / N)
    // Coeffs for Im[0]: 0
    // Coeffs for Im[N/2]: 0

    let num_freqs = n_fft / 2 + 1;
    let input_dim = 2 * num_freqs;
    let mut matrix = vec![0.0f32; input_dim * n_fft]; // Row-major: rows are input channels, cols are output time

    let scale = 1.0 / n_fft as f32;

    for n in 0..n_fft {
        for k in 0..num_freqs {
            // Index in flat vector: row * n_fft + col
            // Row mapping: 0..num_freqs are Real, num_freqs..2*num_freqs are Imag

            let arg = 2.0 * std::f64::consts::PI * (k as f64) * (n as f64) / (n_fft as f64);
            let cos_v = arg.cos() as f32;
            let sin_v = arg.sin() as f32;

            let re_idx = k;
            let im_idx = k + num_freqs;

            let factor = if k == 0 || k == num_freqs - 1 { 1.0 } else { 2.0 };
            // Wait, nyquist index is n_fft/2. num_freqs = n_fft/2 + 1. So last index is n_fft/2.

            // For Re[k]
            matrix[re_idx * n_fft + n] = scale * factor * cos_v;

            // For Im[k]
            if k > 0 && k < num_freqs - 1 {
                 matrix[im_idx * n_fft + n] = -scale * factor * sin_v;
            } else {
                 matrix[im_idx * n_fft + n] = 0.0;
            }
        }
    }

    // Transpose to (n_fft, input_dim) for linear layer?
    // Linear: x @ w.t(). If we want x @ M, then w = M.t().
    // Let's keep it as (input_dim, n_fft) and use matmul explicitly or transpose it.
    // Candle Linear uses weight shape (out_dim, in_dim).
    // Here out_dim = n_fft, in_dim = 2*num_freqs.
    // So we need (n_fft, 2*num_freqs).
    // Our `matrix` vector is currently (in_dim, out_dim) if we view it as rows=in.
    // Let's create tensor (in_dim, out_dim) and transpose.

    let tensor = Tensor::from_vec(matrix, (input_dim, n_fft), device)?;
    Ok(tensor.t()?) // (n_fft, input_dim)
}

pub struct SopranoDecoder {
    decoder: VocosBackbone,
    head: ISTFTHead,
    upscale: usize,
    idft_matrix: Tensor,
}

impl SopranoDecoder {
    pub fn new(cfg: &VocosConfig, vb: VarBuilder) -> Result<Self> {
        let decoder = VocosBackbone::new(cfg, vb.pp("decoder"))?;
        let head = ISTFTHead::new(cfg, vb.pp("head"))?;
        let upscale = cfg.upscale;
        let idft_matrix = make_idft_matrix(cfg.n_fft, vb.device())?;

        Ok(Self {
            decoder,
            head,
            upscale,
            idft_matrix,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Vec<f32>> {
        // x: (B, C, L)
        let (b, c, t) = x.dims3()?;

        // Interpolate
        // x = torch.nn.functional.interpolate(x, size=self.upscale*(T-1)+1, mode='linear', align_corners=True)
        // Candle doesn't have interpolate.
        // We can do nearest neighbor or implement linear interpolation.
        // Linear interpolation for 1D:
        // Output size: new_t = upscale * (t - 1) + 1
        // We can use conv1d transpose maybe? Or just write a kernel.
        // Or simple loop/gather since we need it.
        // Since it's 1D linear interpolation, it's efficient to do with index manipulation.
        // But for "candle" we want to use tensor ops.
        // We can use `upsample_nearest1d` if available, but we need linear.
        // We can use Conv1dTranspose to do linear interpolation?
        // A Conv1dTranspose with stride=upscale and appropriate weights is equivalent to linear interpolation?
        // Linear interpolation corresponds to a triangular kernel.

        // Kernel for linear interp (upscale=4):
        // [0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25] ? No.
        // It's [1-3/4, 1-2/4, 1-1/4, 1, ...]

        // Let's implement it manually using index_select or gather?
        // Or since `upscale` is small (4), we can interleave zeros and do average pooling? No, conv.
        // "Align corners = True" is tricky with conv.

        // Let's leave interpolation for a helper function.

        let x_interp = interpolate_linear(x, self.upscale)?;

        let x_decoded = self.decoder.forward(&x_interp)?;

        // Head
        let x_head = self.head.out.forward(&x_decoded.transpose(1, 2)?)?; // (B, T, out_dim)
        let x_head = x_head.transpose(1, 2)?; // (B, out_dim, T)

        let chunks = x_head.chunk(2, 1)?;
        let mag = &chunks[0];
        let p = &chunks[1];
        let mag = mag.exp()?;
        let mag = mag.clamp(f32::NEG_INFINITY, 100.0)?;

        let cos_p = p.cos()?;
        let sin_p = p.sin()?;

        let re = (mag.clone() * cos_p)?;
        let im = (mag * sin_p)?;

        // ISTFT
        // re, im: (B, F, T)
        let (b, f, t_out) = re.dims3()?;
        let re_t = re.transpose(1, 2)?; // (B, T, F)
        let im_t = im.transpose(1, 2)?;

        let spectral = Tensor::cat(&[&re_t, &im_t], 2)?; // (B, T, 2F)

        // Apply IDFT matrix
        // spectral: (B, T, 2F)
        // idft_matrix: (n_fft, 2F)
        // output: spectral @ idft_matrix.t() -> (B, T, n_fft)

        let ifft = spectral.matmul(&self.idft_matrix.t()?)?;

        // Apply window
        // window: (n_fft)
        let window = &self.head.window;
        let ifft = ifft.broadcast_mul(window)?; // (B, T, n_fft)

        // Overlap Add
        // We need to fold.
        // Flatten ifft to (B, T*n_fft).
        // Accumulate into output buffer.

        // Since B=1 usually for TTS, let's optimize for that or just loop.
        // We return Vec<f32> eventually.
        // Let's move to CPU and do overlap-add in Rust Vec?
        // It might be faster than trying to do it in Candle if ops are missing.

        let ifft_vec = ifft.flatten_all()?.to_vec1::<f32>()?;
        let hop_length = self.head.hop_length;
        let win_length = self.head.win_length;
        let n_frames = t_out;

        // Calculate output size
        let output_size = (n_frames - 1) * hop_length + win_length;
        // Padding "same": pad = (win_length - hop_length) // 2
        let pad = (win_length - hop_length) / 2;

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
        let valid_len = output_size - 2 * pad;
        let mut result = Vec::with_capacity(valid_len);

        for i in pad..(output_size - pad) {
             if envelope[i] > 1e-11 {
                 result.push(output_audio[i] / envelope[i]);
             } else {
                 result.push(output_audio[i]);
             }
        }

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
        let alpha = p - p_floor as f32;

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
