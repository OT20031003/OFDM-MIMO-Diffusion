import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision import transforms
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from torchvision import utils as vutil
import os
import glob
import lpips
from common_utils import SionnaChannelGeneratorGPU, ConditionalUNet, sample_ddpm, device
import torch.nn.functional as F
import tensorflow as tf
import matplotlib.pyplot as plt

# Sionna imports
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, OFDMModulator, OFDMDemodulator, ResourceGridDemapper
from sionna.phy.channel import ApplyOFDMChannel, AWGN
from sionna.phy.mimo import StreamManagement

def load_images_as_tensors(dir_path, image_size=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    image_paths = []
    supported_formats = ["*.jpg", "*.jpeg", "*.png"]
    for fmt in supported_formats:
        image_paths.extend(glob.glob(os.path.join(dir_path, fmt)))

    if not image_paths:
        print(f"警告: ディレクトリ '{dir_path}' にサポートされている画像ファイルが見つかりません。")
        return []

    tensors_list = []
    for t in trange(len(image_paths), desc="Loading Image"):
        path = image_paths[t]
        try:
            img = Image.open(path).convert("RGB")
            tensor_img = transform(img)
            tensors_list.append(tensor_img)
        except Exception as e:
            print(f"エラー: ファイル '{path}' の読み込みに失敗しました。スキップします。エラー内容: {e}")
    if not tensors_list:
        return torch.empty(0)

    return torch.stack(tensors_list, dim=0)

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def save_img_individually(img, path):
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    dirname = os.path.dirname(path)
    basename = os.path.splitext(os.path.basename(path))[0]
    ext = os.path.splitext(path)[1]
    os.makedirs(dirname, exist_ok=True)
    batch_size = img.shape[0]
    for i in range(batch_size):
        individual_path = os.path.join(dirname, f"{basename}_{i}{ext}")
        vutil.save_image(img[i], individual_path)
    print(f"{batch_size} images are saved in {dirname}/")

def remove_png(path):
    png_files = glob.glob(f'{path}/*.png')
    for file in png_files:
        try:
            os.remove(f"{file}")
        except OSError as e:
            print(f"Error: {e.strerror} - {e.filename}")
    print(f"remove_png complete")

# 環境設定
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def manual_zf_equalization_with_error(y_received_grid, h_est_tf, h_true_tf, rg, no):
    """
    ZF Equalization and calculation of Effective Noise Variance (SINR)
    considering estimation error.
    Returns effective_noise_var per batch.
    """
    # 1. Arrange axes
    # H_est -> [Batch, T, F, Rx, Stream]
    h_est_sq = tf.squeeze(h_est_tf, axis=[1, 3]) 
    h_est_perm = tf.transpose(h_est_sq, perm=[0, 3, 4, 1, 2]) 
    
    # H_true -> [Batch, T, F, Rx, Stream]
    h_true_sq = tf.squeeze(h_true_tf, axis=[1, 3])
    h_true_perm = tf.transpose(h_true_sq, perm=[0, 3, 4, 1, 2])

    # Y -> [Batch, T, F, Rx, 1]
    y_sq = tf.squeeze(y_received_grid, axis=1) 
    y_perm = tf.transpose(y_sq, perm=[0, 2, 3, 1]) 
    y_perm = tf.expand_dims(y_perm, axis=-1)

    # 2. Compute ZF Combiner W using Estimated Channel
    h_est_herm = tf.linalg.adjoint(h_est_perm)
    gram = tf.matmul(h_est_herm, h_est_perm)
    gram_inv = tf.linalg.inv(gram) 
    W = tf.matmul(gram_inv, h_est_herm) # [Batch, T, F, Stream, Rx]

    # 3. Compute Effective Noise Variance
    # H_err = H_true - H_est
    h_err = h_true_perm - h_est_perm # [Batch, T, F, Rx, Stream]
    
    # Inner Covariance: H_err * H_err^H + N0 * I
    h_err_herm = tf.linalg.adjoint(h_err)
    cov_h_err = tf.matmul(h_err, h_err_herm) # [Batch, T, F, Rx, Rx]
    
    rx_dim = tf.shape(h_est_perm)[-2]
    eye_rx = tf.eye(rx_dim, batch_shape=tf.shape(h_est_perm)[:-2], dtype=cov_h_err.dtype)
    cov_noise = tf.cast(no, dtype=cov_h_err.dtype) * eye_rx
    
    cov_inner = cov_h_err + cov_noise
    
    # Output Noise Covariance: W * Cov_inner * W^H
    w_herm = tf.linalg.adjoint(W)
    cov_eff = tf.matmul(tf.matmul(W, cov_inner), w_herm) # [Batch, T, F, Stream, Stream]
    
    # Extract diagonal (Variance per stream)
    diag_eff = tf.linalg.diag_part(cov_eff) # [Batch, T, F, Stream]
    
    # Average over Time(1), Freq(2), Stream(3) to get a value per Batch(0)
    # Result shape: [Batch]
    effective_noise_var = tf.reduce_mean(tf.math.real(diag_eff), axis=[1, 2, 3])
    
    # 4. Equalization
    x_hat_perm = tf.matmul(W, y_perm)
    
    # Restore shape [B, 1, Stream, T, F]
    x_hat_perm = tf.squeeze(x_hat_perm, axis=-1)
    x_hat_reshaped = tf.transpose(x_hat_perm, perm=[0, 3, 1, 2])
    x_hat = tf.expand_dims(x_hat_reshaped, axis=1)
    
    return x_hat, effective_noise_var

def torch_to_tf_channel(h_torch_complex, batch_size, rg):
    """PyTorchチャネル -> TensorFlowチャネル変換"""
    h_np = h_torch_complex.detach().cpu().numpy()
    h_reshaped = h_np.reshape(batch_size, 1, 4, 1, 4, rg.num_ofdm_symbols, rg.fft_size)
    return tf.constant(h_reshaped, dtype=tf.complex64)

def reconstruct_z(q_hat_tf, z_variance, transmit_data_size, batch_size, channel_size, h_size, w_size):
    # TF -> Torch
    q_hat_torch = torch.from_numpy(q_hat_tf.numpy()).to(device)
    q_hat_flat = q_hat_torch.view(batch_size, -1)
    q_hat_valid = q_hat_flat[:, :transmit_data_size]

    real_part_hat = q_hat_valid.real
    imag_part_hat = q_hat_valid.imag
    q_view_hat = torch.cat((real_part_hat, imag_part_hat), dim=1)
    
    eps = 0.000001
    scale_factor = torch.sqrt(2 * (z_variance + eps)).view(-1, 1)

    z_hat_flat = q_view_hat * scale_factor
    z_hat = z_hat_flat.view(batch_size, channel_size, h_size, w_size)
    
    return z_hat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    T = None
    mode = "Linear_Interp"
    #mode = "Diffusion"
    parser.add_argument("--prompt", type=str, nargs="?", default="a painting of a virus monster playing guitar")
    parser.add_argument("--outdir", type=str, nargs="?", default=f"outputs/OFDM/{mode}_T={T}")
    parser.add_argument("--nosample_outdir", type=str, nargs="?", default=f"outputs/OFDM/{mode}_nodiffusion")
    parser.add_argument("--sentimgdir", type=str, nargs='?', default="./sentimg")
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--plms", action='store_true')
    parser.add_argument("--input_path", type=str, default="input_img")
    parser.add_argument("--intermediate_path", type=str, default=None)
    
    opt = parser.parse_args()
    if opt.intermediate_path != None:
        os.makedirs(opt.intermediate_path, exist_ok=True)
    
    if T == None or T <0:
        opt.outdir = f"outputs/OFDM/{mode}_dynamic"
        opt.nosample_outdir = f"outputs/OFDM/{mode}_dynamic/nodiffusion"
        
    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml") 
    model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt") 

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    os.makedirs(opt.sentimgdir, exist_ok=True)
    os.makedirs(opt.nosample_outdir, exist_ok=True)

    img = load_images_as_tensors(opt.input_path)
    BATCH_SIZE = img.shape[0]
    save_img_individually(img, opt.sentimgdir + "/sentimg.png")
    img = img.to(device=device)
    z = model.encode_first_stage(img)
    z = model.get_first_stage_encoding(z).detach()
    z_variance = torch.var(z, dim=(1, 2, 3))
    
    CHANNEL_SIZE = z.shape[1]
    W_SIZE = z.shape[3]
    H_SIZE = z.shape[2]
    
    eps = 0.0000001
    q_real_data = z/ torch.sqrt(2*(z_variance + eps)).view(-1, 1, 1, 1)
    q_view = q_real_data.view(BATCH_SIZE, -1)
    real_part, imag_part = torch.chunk(q_view, 2, dim=1)
    q = torch.complex(real_part, imag_part)
    
    generator = SionnaChannelGeneratorGPU(batch_size=BATCH_SIZE)
    model_channel = ConditionalUNet(in_channels=32, cond_channels=32).to(device)
    
    CHECKPOINT_PATH = "ckpt_step_135000.pth"
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        model_channel.load_state_dict(ckpt['model_state_dict'])
    else:
        print("Checkpoint not found! Please run train.py first.")
        raise ValueError
    
    for snr in range(-5, 10, 1):
        print(f"--------SNR = {snr}-----------")
        x_gt, x_cond = generator.get_batch(snr_db=snr)
        x_est_diff = sample_ddpm(model_channel, x_cond, snr, x_gt.shape, device)
        
        def to_complex(x):
            c = x.shape[1] // 2
            return torch.complex(x[:, :c, ...], x[:, c:, ...])
        h_perfect_torch = to_complex(x_gt)
        h_linear_torch = to_complex(x_cond)
        h_diff_torch = to_complex(x_est_diff)
        
        mse_linear = torch.mean(torch.abs(h_perfect_torch - h_linear_torch)**2).item()
        mse_diff = torch.mean(torch.abs(h_perfect_torch - h_diff_torch)**2).item()
        print(f"\nChannel MSE (Linear): {mse_linear:.5f}")
        print(f"\nChannel MSE (Diffusion): {mse_diff:.5f}")
        
        rg = generator.rg
        n_data_per_stream = rg.num_data_symbols
        num_streams = rg.num_streams_per_tx
        total_capacity = n_data_per_stream * num_streams
        transmit_data_size = q.shape[1]
        
        if total_capacity >= transmit_data_size:
            padding_size = total_capacity - transmit_data_size
            q_padded = F.pad(q, (0, padding_size), mode='constant', value=0)
        else:
            raise ValueError("Data size exceeds grid capacity.")
        
        q_reshaped = q_padded.view(BATCH_SIZE, 1, num_streams, n_data_per_stream)
        q_np = q_reshaped.detach().cpu().numpy()

        mapper = ResourceGridMapper(rg)
        x_rg_tf = mapper(q_np)

        h_perfect_tf = torch_to_tf_channel(h_perfect_torch, BATCH_SIZE, rg)
        channel_applier = ApplyOFDMChannel(add_awgn=True)
        no = 10**(-snr / 10)
        y_rg_tf = channel_applier(x_rg_tf, h_perfect_tf, no)

        h_linear_tf = torch_to_tf_channel(h_linear_torch, BATCH_SIZE, rg)
        h_diff_tf = torch_to_tf_channel(h_diff_torch, BATCH_SIZE, rg)
        rx_tx_association = np.array([[1]])
        sm = StreamManagement(rx_tx_association, num_streams_per_tx=4)
        demapper = ResourceGridDemapper(rg, stream_management=sm)

        results = {}
        
        def evaluate_method(name, h_est_tf, h_true_tf):
            # 1. ZF Equalization (Returns batch-wise effective noise variance)
            x_hat, eff_noise_var_batch = manual_zf_equalization_with_error(y_rg_tf, h_est_tf, h_true_tf, rg, no)
            
            # 2. Demapping
            q_hat_tf = demapper(x_hat)
            
            # 3. Reconstruction z
            z_hat = reconstruct_z(q_hat_tf, z_variance, transmit_data_size, 
                                BATCH_SIZE, CHANNEL_SIZE, H_SIZE, W_SIZE)
            
            # 4. MSE Calculation
            mse_val = F.mse_loss(z, z_hat).item()
            
            # SINR Calculation (Batch-wise)
            eff_noise_var_np = eff_noise_var_batch.numpy()
            sinr_batch = 1.0 / (eff_noise_var_np + 1e-12)
            
            # Convert to Z-domain noise variance (Batch-wise)
            # Use per-image scale factor from z_variance
            scale_sq_batch = 2 * z_variance.detach().cpu().numpy() # [Batch]
            z_noise_var_batch = eff_noise_var_np * scale_sq_batch
            
            return mse_val, z_hat, sinr_batch, z_noise_var_batch

        # Case 1: Perfect CSI
        mse_perf, z_hat_perf, sinr_perf_batch, var_perf_batch = evaluate_method("Perfect CSI", h_perfect_tf, h_perfect_tf)
        results['Perfect CSI'] = mse_perf

        # Case 2: Linear Interp
        mse_lin, z_hat_lin, sinr_lin_batch, var_lin_batch = evaluate_method("Linear Interp", h_linear_tf, h_perfect_tf)
        results['Linear Interp'] = mse_lin

        # Case 3: Diffusion
        mse_diff, z_hat_diff, sinr_diff_batch, var_diff_batch = evaluate_method("Diffusion", h_diff_tf, h_perfect_tf)
        results['Diffusion'] = mse_diff
        
        # Calculate Average dB for display
        def to_db(sinr_batch):
            # Calculate mean of SINR in dB (or dB of mean, here using dB of mean for stability)
            mean_sinr = np.mean(sinr_batch)
            return 10 * np.log10(mean_sinr + 1e-12)
        
        sinr_db_perf = to_db(sinr_perf_batch)
        sinr_db_lin = to_db(sinr_lin_batch)
        sinr_db_diff = to_db(sinr_diff_batch)
        
        # Calculate mean Noise Var for display
        mean_var_perf = np.mean(var_perf_batch)
        mean_var_lin = np.mean(var_lin_batch)
        mean_var_diff = np.mean(var_diff_batch)

        print("-" * 70)
        print(f"Reconstructed z MSE & SINR (SNR={snr}dB) [Including Estimation Error]")
        print("-" * 70)
        print(f"{'Method':<20} {'MSE':<15} {'SINR(dB)':<15} {'Avg Noise Var':<15}")
        print(f"{'Perfect CSI':<20} {mse_perf:.5f}       {sinr_db_perf:.5f}       {mean_var_perf:.5f}")
        print(f"{'Linear Interp':<20} {mse_lin:.5f}       {sinr_db_lin:.5f}       {mean_var_lin:.5f}")
        print(f"{'Diffusion':<20} {mse_diff:.5f}       {sinr_db_diff:.5f}       {mean_var_diff:.5f}")
        print("-" * 70)
        
        # Select method based on 'mode'
        if mode == "Perfect_CSI":
            target_z_hat = z_hat_perf
            target_noise_var = var_perf_batch
        elif mode == "Linear_Interp":
            target_z_hat = z_hat_lin
            target_noise_var = var_lin_batch
        elif mode == "Diffusion":
            target_z_hat = z_hat_diff
            target_noise_var = var_diff_batch
        else:
            target_z_hat = z_hat_diff
            target_noise_var = var_diff_batch

        recoverd_img_no_samp = model.decode_first_stage(target_z_hat)
        cond = model.get_learned_conditioning(z.shape[0] * [""])
        print(f"####cond finisihed #####")
        target_noise_var = torch.from_numpy(target_noise_var).float().to(device)
        # Pass the batch-wise noise variance to the sampler
        samples = sampler.MIMO_decide_starttimestep_ddim_sampling(S=opt.ddim_steps, batch_size=BATCH_SIZE,
                        shape= z.shape[1:4], x_T=target_z_hat,
                        conditioning=cond, starttimestep=T, noise_variance=target_noise_var)

        print(f"d = {samples.shape}")
        recoverd_img = model.decode_first_stage(samples)
        print(f"recoverd_img = {recoverd_img.shape}")
        save_img_individually(recoverd_img, f"{opt.outdir}/output_{snr}.png")
        save_img_individually(recoverd_img_no_samp, f"{opt.nosample_outdir}/output_{snr}.png")