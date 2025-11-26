import argparse
import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from torchvision.utils import make_grid
from torchvision import transforms

# --- LDM Imports (ALLSU-MIMO.pyより) ---
# ユーザーの環境にこれらのライブラリがある前提です
try:
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.ddim import DDIMSampler
    from ldm.models.diffusion.plms import PLMSSampler
except ImportError:
    print("Warning: ldm modules not found. Ensure CompVis/latent-diffusion is installed.")

# --- Sionna & Physics Imports (common_utils.pyより) ---
import tensorflow as tf
from torch.utils.dlpack import from_dlpack
from sionna.phy.ofdm import ResourceGrid
from sionna.phy.channel.tr38901 import AntennaArray, CDL
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel

# ==========================================
# 1. 共通設定・モデル定義 (common_utils.py ベース)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNKNOWN_SNR_VALUE = -100.0

# TensorFlow GPU Memory Check
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

class ConditionalUNet(nn.Module):
    """
    拡散モデルベースのチャネル推定器
    """
    def __init__(self, in_channels=64, cond_channels=64, time_emb_dim=32):
        super().__init__()
        total_in_channels = in_channels + cond_channels
        
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.time_proj = nn.Linear(time_emb_dim, 512)

        self.snr_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.snr_proj = nn.Linear(time_emb_dim, 512)
        
        self.down1 = nn.Conv2d(total_in_channels, 128, 3, padding=1)
        self.down2 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        
        self.bot1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bot2 = nn.Conv2d(512, 512, 3, padding=1)
        
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv1 = nn.Conv2d(256 + 128, 256, 3, padding=1) 
        self.conv2 = nn.Conv2d(256, in_channels, 3, padding=1)
        
        self.act = nn.GELU()

    def forward(self, x, cond, t, snr): 
        x_in = torch.cat([x, cond], dim=1) 
        
        t = t.float().view(-1, 1) / 1000.0
        t_emb = self.time_mlp(t)
        t_emb = self.time_proj(t_emb).view(-1, 512, 1, 1)

        s = snr.float().view(-1, 1) / 30.0 
        s_emb = self.snr_mlp(s)
        s_emb = self.snr_proj(s_emb).view(-1, 512, 1, 1)
        
        x1 = self.act(self.down1(x_in))
        x2 = self.act(self.down2(self.pool(x1)))
        
        x_bot = self.act(self.bot1(x2) + t_emb + s_emb)
        x_bot = self.act(self.bot2(x_bot))
        
        x_up = self.up1(x_bot)
        if x_up.shape != x1.shape:
             x_up = torch.nn.functional.interpolate(x_up, size=x1.shape[2:])
        
        x_dec = torch.cat([x_up, x1], dim=1)
        x_dec = self.act(self.conv1(x_dec))
        out = self.conv2(x_dec)
        
        return out

@torch.no_grad()
def sample_channel_ddpm(model, x_cond, shape, device, snr_tensor):
    """
    チャネル推定用の拡散モデルサンプリング
    """
    T = 1000
    beta = torch.linspace(1e-4, 0.02, T).to(device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    
    model.eval()
    x = torch.randn(shape, device=device)
    
    # 簡易化のためtqdmなし、または外部ループで表示
    for t in reversed(range(T)):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        predicted_noise = model(x, x_cond, t_batch, snr_tensor)
        
        curr_alpha = alpha[t]
        curr_alpha_bar = alpha_bar[t]
        curr_beta = beta[t]
        
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = 0
            
        coeff1 = 1 / torch.sqrt(curr_alpha)
        coeff2 = (1 - curr_alpha) / torch.sqrt(1 - curr_alpha_bar)
        sigma = torch.sqrt(curr_beta)
        
        x = coeff1 * (x - coeff2 * predicted_noise) + sigma * noise
        x = torch.clamp(x, min=-5.0, max=5.0)
    return x

# ==========================================
# 2. OFDM System & Transmission Logic
# ==========================================
class OFDMMIMOSystem:
    def __init__(self):
        self.pilot_indices = [2, 11]
        self.num_symbols = 14
        self.fft_size = 76
        
        # 4 Streams (Tx) -> 8 Antennas (BS), 4 Antennas (UT) for Uplink
        # common_utilsの設定に合わせる: UT(4ant) -> BS(8ant) Uplink
        # SionnaのResourceGridは num_tx=1, num_streams_per_tx=4 となっているが
        # 物理的なアンテナ構成はCDLで定義される。
        # ここでは「4ストリーム」のデータを送ることを想定。
        self.num_streams = 4 
        
        self.rg = ResourceGrid(
            num_ofdm_symbols=self.num_symbols,
            fft_size=self.fft_size, subcarrier_spacing=15e3,
            num_tx=1, num_streams_per_tx=self.num_streams,
            cyclic_prefix_length=6, pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=self.pilot_indices
        )
        
        carrier_freq = 2.6e9
        # Uplink: UT(Tx) -> BS(Rx)
        self.ut_array = AntennaArray(num_rows=1, num_cols=2, polarization="dual", polarization_type="cross", antenna_pattern="38.901", carrier_frequency=carrier_freq)
        self.bs_array = AntennaArray(num_rows=1, num_cols=4, polarization="dual", polarization_type="cross", antenna_pattern="38.901", carrier_frequency=carrier_freq)
        
        self.cdl = CDL("B", 300e-9, carrier_freq, self.ut_array, self.bs_array, "uplink", min_speed=10)
        self.frequencies = subcarrier_frequencies(self.rg.fft_size, self.rg.subcarrier_spacing)
        
        # Linear Interpolation Grid
        self.t_grid = torch.arange(self.num_symbols, device=device).view(1, 1, -1, 1).float()

        # データ用マスクの作成 (パイロット以外)
        self.data_mask = np.ones((self.num_symbols, self.fft_size), dtype=bool)
        self.data_mask[self.pilot_indices, :] = False
        self.num_data_re = np.sum(self.data_mask) # 1スロットあたりのデータRE数

    def generate_channel(self, batch_size):
        """Sionnaでチャネル生成 (Frequency Domain)"""
        a, tau = self.cdl(batch_size=batch_size, 
                          num_time_steps=self.rg.num_ofdm_symbols, 
                          sampling_frequency=1/self.rg.ofdm_symbol_duration)
        h_freq = cir_to_ofdm_channel(self.frequencies, a, tau, normalize=True)
        
        # DLPack conversion
        try:
            h_torch = from_dlpack(tf.experimental.dlpack.to_dlpack(h_freq)).cfloat().to(device)
        except Exception:
            h_torch = torch.from_numpy(h_freq.numpy()).cfloat().to(device)
            
        # Shape: [Batch, Rx(1), RxAnt(8), Tx(1), TxAnt(4), T(14), F(76)]
        # Flatten spatial dims for processing: [Batch, RxAnt, TxAnt, T, F]
        b, _, rx_ant, _, tx_ant, t, f = h_torch.shape
        h_torch = h_torch.view(b, rx_ant, tx_ant, t, f)
        
        # モデル入力用にShape変換: [B, Channels(Rx*Tx*2), T, F]
        # Rx(8) * Tx(4) = 32 paths. Real+Imag = 64 channels.
        h_flat = h_torch.view(b, rx_ant*tx_ant, t, f)
        x_gt_img = torch.cat([h_flat.real, h_flat.imag], dim=1).float()
        
        return h_torch, x_gt_img

    def estimate_channel_linear(self, h_gt_complex, snr_db):
        """パイロット位置のチャネルから線形補間"""
        b, rx, tx, t, f = h_gt_complex.shape
        
        # 1. パイロット抽出 (Noisy)
        t1, t2 = self.pilot_indices
        val_t1 = h_gt_complex[:, :, :, t1:t1+1, :]
        val_t2 = h_gt_complex[:, :, :, t2:t2+1, :]
        
        # Noise Addition
        sig_pwr = torch.mean(torch.abs(h_gt_complex)**2)
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_std = torch.sqrt(sig_pwr / snr_linear / 2.0)
        
        n1 = (torch.randn_like(val_t1) + 1j * torch.randn_like(val_t1)) * noise_std
        n2 = (torch.randn_like(val_t2) + 1j * torch.randn_like(val_t2)) * noise_std
        
        val_t1_noisy = val_t1 + n1
        val_t2_noisy = val_t2 + n2
        
        # 2. 線形補間
        slope = (val_t2_noisy - val_t1_noisy) / (t2 - t1)
        h_cond_complex = val_t1_noisy + slope * (self.t_grid.view(1,1,1,-1,1) - t1)
        
        # 3. 画像形式に変換
        h_cond_flat = h_cond_complex.view(b, rx*tx, t, f)
        x_cond_img = torch.cat([h_cond_flat.real, h_cond_flat.imag], dim=1).float()
        
        return h_cond_complex, x_cond_img


    def zf_equalize(self, y, h_est, noise_var=0.0):
            """
            Zero-Forcing Equalization per RE
            y: [B, Rx, T, F]
            h_est: [B, Rx, Tx, T, F]
            Returns: x_hat [B, Tx, T, F]
            """
            # Permute to [B, T, F, Rx, 1] and [B, T, F, Rx, Tx] for matmul
            y_perm = y.permute(0, 2, 3, 1).unsqueeze(-1) # [B, T, F, Rx, 1]
            
            # 修正箇所: T(dim3)とF(dim4)を前に持ってきて、最後にRx(dim1), Tx(dim2)を配置する
            h_perm = h_est.permute(0, 3, 4, 1, 2)         # [B, T, F, Rx, Tx]
            
            # ZF: (H^H H)^-1 H^H y
            # Pseudo-inverse is safer
            # h_perm が [B, T, F, Rx, Tx] なので、pinvの結果は [B, T, F, Tx, Rx] になります
            h_pinv = torch.linalg.pinv(h_perm) # [B, T, F, Tx, Rx]
            
            x_hat = torch.matmul(h_pinv, y_perm) # [B, T, F, Tx, 1]
            
            return x_hat.squeeze(-1).permute(0, 3, 1, 2) # [B, Tx, T, F]
# ==========================================
# 3. Main Helper Functions
# ==========================================
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "state_dict" in pl_sd:
        sd = pl_sd["state_dict"]
    else:
        sd = pl_sd
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if verbose:
        print("missing keys:", m)
        print("unexpected keys:", u)
    model.cuda()
    model.eval()
    return model

def load_images(dir_path, size=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    files = glob.glob(os.path.join(dir_path, "*.png")) + glob.glob(os.path.join(dir_path, "*.jpg"))
    imgs = []
    for f in files:
        img = Image.open(f).convert("RGB")
        imgs.append(transform(img))
    if not imgs:
        return torch.empty(0)
    return torch.stack(imgs)

# ==========================================
# 4. Main Script
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr", type=float, default=0.0, help="Test SNR (dB)")
    parser.add_argument("--img_dir", type=str, default="input_img", help="Input images")
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )
    parser.add_argument("--out_dir", type=str, default="outputs/compare_zf", help="Output directory")
    parser.add_argument("--ckpt_diff", type=str, default="ckpt_step_135000.pth", help="Channel Estimator Checkpoint")
    parser.add_argument("--ckpt_ldm", type=str, default="models/ldm/text2img-large/model.ckpt", help="LDM Checkpoint")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. Load Models
    print("Initializing Systems...")
    ofdm_sys = OFDMMIMOSystem()
    
    # Channel Estimator
    chan_est_model = ConditionalUNet(in_channels=64, cond_channels=64).to(device)
    try:
        ckpt = torch.load(args.ckpt_diff, map_location=device)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        chan_est_model.load_state_dict(state_dict)
        print("Channel Estimation Model Loaded.")
    except Exception as e:
        print(f"Failed to load channel estimator: {e}")
        return

    # LDM (Source Coder)
    try:
        config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")
        ldm_model = load_model_from_config(config, args.ckpt_ldm)
        # if hasattr(ldm_model, 'cond_stage_model'):
        #      del ldm_model.cond_stage_model # Save memory if text cond not used
        
        # Check for the custom sampler method
        sampler = DDIMSampler(ldm_model)
        if not hasattr(sampler, "MIMO_decide_starttimestep_ddim_sampling"):
            print("ERROR: Custom method 'MIMO_decide_starttimestep_ddim_sampling' not found in DDIMSampler.")
            print("Please ensure your 'ldm' library contains the custom modifications from ALLSU-MIMO.")
            # For fallback, standard sampling could be used, but user requested the specific method.
    except Exception as e:
        print(f"Failed to load LDM: {e}")
        return

    # 2. Load Images
    images = load_images(args.img_dir).to(device)
    if len(images) == 0:
        print("No images found.")
        return

    # 3. Transmission Loop
    print(f"Starting Transmission at SNR={args.snr}dB...")
    
    for i, img in enumerate(tqdm(images)):
        # --- A. Encode (VAE) ---
        # img: [C, H, W] -> [1, C, H, W]
        x_in = img.unsqueeze(0).to(device) * 2.0 - 1.0 # [-1, 1] range for VAE
        
        with torch.no_grad():
            z = ldm_model.encode_first_stage(x_in)
            z = ldm_model.get_first_stage_encoding(z).detach() # [1, 4, 32, 32]
        
        # Normalize logic (from ALLSU)
        eps = 1e-6
        z_variance = torch.var(z)
        z_norm = z / torch.sqrt(2 * (z_variance + eps))
        
        # Flatten and Complexify
        # z: [1, 4, 32, 32] -> 4096 elements
        # Map to complex symbols: 2 floats -> 1 complex
        # 4096 floats -> 2048 complex symbols
        z_flat = z_norm.view(-1)
        z_complex = torch.complex(z_flat[0::2], z_flat[1::2]) # 2048 symbols
        
        # --- B. Map to OFDM Slots ---
        # Capacity per slot:
        # Data REs per slot = num_data_re(approx 12*76) * num_streams(4)
        num_data_re_per_stream = ofdm_sys.num_data_re
        capacity_per_slot = num_data_re_per_stream * ofdm_sys.num_streams
        
        num_slots = int(np.ceil(len(z_complex) / capacity_per_slot))
        
        received_symbols_lin = []
        received_symbols_diff = []
        
        current_idx = 0
        
        for slot in range(num_slots):
            # 1. Prepare Data Chunk
            chunk_len = min(capacity_per_slot, len(z_complex) - current_idx)
            data_chunk = z_complex[current_idx : current_idx + chunk_len]
            
            # Padding if needed
            if chunk_len < capacity_per_slot:
                pad = torch.zeros(capacity_per_slot - chunk_len, device=device, dtype=torch.cfloat)
                data_chunk = torch.cat([data_chunk, pad])
            
            # Reshape to [Tx(4), Data_REs]
            x_slot = data_chunk.view(ofdm_sys.num_streams, num_data_re_per_stream)
            
            # Map to Resource Grid [Tx, T, F]
            # Initialize with zeros
            x_grid = torch.zeros((ofdm_sys.num_streams, ofdm_sys.num_symbols, ofdm_sys.fft_size), 
                                 dtype=torch.cfloat, device=device)
            
            # Fill Data
            # pilot_mask is False at pilot positions
            mask = torch.tensor(ofdm_sys.data_mask, device=device)
            # Assign data to masked positions
            # x_grid[:, mask] = x_slot # This indexing flattens T,F
            # Need to be careful with indexing. 
            # mask shape [T, F]. x_grid shape [S, T, F].
            for s in range(ofdm_sys.num_streams):
                 x_grid[s, mask] = x_slot[s]
            
            # Add Batch Dim -> [1, S, T, F]
            x_tx = x_grid.unsqueeze(0)
            
            # 2. Channel & Transmission
            # Generate Channel
            h_freq, x_gt_img = ofdm_sys.generate_channel(batch_size=1)
            
            # Apply Channel: Y = HX + N
            # H: [1, Rx(8), Tx(4), T, F]
            # X: [1, Tx(4), T, F]
            # Y: [1, Rx(8), T, F]
            
            # Matmul over Tx dimension
            # Permute for broadcast: H[..., Tx, T, F], X[..., Tx, T, F]
            # H*X -> sum over Tx
            y_clean = torch.einsum('brstf,bstf->brtf', h_freq, x_tx)
            
            # Add Noise
            sig_pwr = torch.mean(torch.abs(y_clean)**2)
            snr_linear = 10.0 ** (args.snr / 10.0)
            noise_pwr = sig_pwr / snr_linear
            noise_std = torch.sqrt(noise_pwr / 2.0)
            noise = (torch.randn_like(y_clean) + 1j*torch.randn_like(y_clean)) * noise_std
            y_rx = y_clean + noise
            
            # 3. Channel Estimation
            # A) Linear Interpolation
            h_lin, x_cond_img = ofdm_sys.estimate_channel_linear(h_freq, args.snr) # Note: simulates piloting
            
            # B) Diffusion Refinement
            # Blind SNR (-100) or True SNR? User's prev code used Blind logic for inference.
            # Using UNKNOWN_SNR_VALUE (-100.0) as per request for "Blind"
            snr_cond = torch.full((1,), UNKNOWN_SNR_VALUE, device=device)
            x_est_diff = sample_channel_ddpm(chan_est_model, x_cond_img, x_gt_img.shape, device, snr_cond)
            
            # Convert Image back to Complex Channel
            # x_est_diff: [1, 64, T, F] -> h_diff: [1, Rx, Tx, T, F]
            h_diff_real = x_est_diff[:, :32, :, :]
            h_diff_imag = x_est_diff[:, 32:, :, :]
            h_diff = torch.complex(h_diff_real, h_diff_imag).view(1, 8, 4, ofdm_sys.num_symbols, ofdm_sys.fft_size)
            
            # 4. Equalization (ZF)
            # A) Using Linear H
            x_hat_lin = ofdm_sys.zf_equalize(y_rx, h_lin)
            
            # B) Using Diff H
            x_hat_diff = ofdm_sys.zf_equalize(y_rx, h_diff)
            
            # 5. Extract Data
            # x_hat: [1, Tx, T, F]
            for s in range(ofdm_sys.num_streams):
                # Extract using mask
                d_lin = x_hat_lin[0, s, mask]
                d_diff = x_hat_diff[0, s, mask]
                
                # Append to list (only valid data length)
                # Need to verify if this is the last chunk
                pass # logic handled by flattening
                
            # Simply flatten masked area and concat
            # [1, S, T, F] -> select mask -> [S, Data_REs] -> flatten
            x_lin_flat = x_hat_lin[0, :, mask].view(-1)
            x_diff_flat = x_hat_diff[0, :, mask].view(-1)
            
            if chunk_len < capacity_per_slot:
                 # Trim padding
                 x_lin_flat = x_lin_flat[:chunk_len]
                 x_diff_flat = x_diff_flat[:chunk_len]
            
            received_symbols_lin.append(x_lin_flat)
            received_symbols_diff.append(x_diff_flat)
            
            current_idx += chunk_len
        
        # --- C. Reconstruct Latent ---
        z_rx_lin = torch.cat(received_symbols_lin)
        z_rx_diff = torch.cat(received_symbols_diff)
        
        # Complex to Real
        # 2048 complex -> 4096 real
        def complex_to_real_flat(c):
            r = torch.zeros(len(c)*2, device=device)
            r[0::2] = c.real
            r[1::2] = c.imag
            return r
            
        z_real_lin = complex_to_real_flat(z_rx_lin).view(1, 4, 32, 32)
        z_real_diff = complex_to_real_flat(z_rx_diff).view(1, 4, 32, 32)
        
        # Denormalize (Approximate, using Tx stats)
        z_est_lin = z_real_lin * torch.sqrt(2 * (z_variance + eps))
        z_est_diff = z_real_diff * torch.sqrt(2 * (z_variance + eps))
        
        # --- D. Denoising (LDM) ---
        # Calculate approximate SINR/Noise variance for the sampler
        # MSE between sent z and received z
        mse_lin = torch.mean((z_est_lin - z)**2)
        mse_diff = torch.mean((z_est_diff - z)**2)
        
        # Assuming signal power approx 1 (due to normalization), noise_var ~ MSE
        # Or strictly: 1/SINR
        noise_var_lin = mse_lin.item()
        noise_var_diff = mse_diff.item()
        
        print(f"Image {i}: SNR={args.snr}dB | MSE(Lin)={mse_lin:.4f} | MSE(Diff)={mse_diff:.4f}")

        # Conditioning (Empty for unconditional generation/refinement)
        cond = ldm_model.get_learned_conditioning([""])
        
        # Run Sampler (Linear)
        print("  Denoising Linear Result...")
        try:
            samples_lin, _ = sampler.MIMO_decide_starttimestep_ddim_sampling(
                S=args.ddim_steps, # ddim_steps
                batch_size=1,
                shape=(4, 32, 32),
                x_T=z_est_lin,
                conditioning=cond,
                noise_variance=noise_var_lin
            )
            img_lin = ldm_model.decode_first_stage(samples_lin)
        except AttributeError:
             # Fallback if custom method missing
             img_lin = ldm_model.decode_first_stage(z_est_lin)

        # Run Sampler (Diff)
        print("  Denoising Diffusion Result...")
        try:
            samples_diff, _ = sampler.MIMO_decide_starttimestep_ddim_sampling(
                S=args.ddim_steps,
                batch_size=1,
                shape=(4, 32, 32),
                x_T=z_est_diff,
                conditioning=cond,
                noise_variance=noise_var_diff
            )
            img_diff = ldm_model.decode_first_stage(samples_diff)
        except AttributeError:
            img_diff = ldm_model.decode_first_stage(z_est_diff)
            
        # --- E. Save ---
        # Original
        img_gt = ldm_model.decode_first_stage(z)
        
        def save_tensor(t, name):
            t = torch.clamp((t + 1.0) / 2.0, min=0.0, max=1.0)
            vutil_save(t, name)

        from torchvision.utils import save_image as vutil_save
        
        save_path = os.path.join(args.out_dir, f"img_{i}_snr{int(args.snr)}")
        save_tensor(img_gt, f"{save_path}_gt.png")
        save_tensor(img_lin, f"{save_path}_linear.png")
        save_tensor(img_diff, f"{save_path}_diff.png")
        
        # Grid View
        grid = torch.cat([img_gt, img_lin, img_diff], dim=0)
        save_tensor(grid, f"{save_path}_combined.png")

if __name__ == "__main__":
    main()