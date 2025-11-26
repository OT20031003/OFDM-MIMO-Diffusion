# common_utils.py
import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf

# ★高速化: DLPackを使ってGPUメモリ間でデータを渡すためのインポート
from torch.utils.dlpack import from_dlpack

# ResourceGrid は ofdm モジュールにあります
from sionna.phy.ofdm import ResourceGrid
from sionna.phy.channel.tr38901 import AntennaArray, CDL
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel

# 定数: SNRが不明であることを示す値 (実際のSNR(-5~30dB)と被らない値にする)
UNKNOWN_SNR_VALUE = -100.0

# GPU設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TensorFlowのGPUメモリ確保設定
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # TFがメモリを食いつぶさないようにGrowthモードにする
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# ==========================================
# 1. データ生成クラス (DLPack高速化版)
# ==========================================
class SionnaChannelGeneratorGPU:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.pilot_indices = [2, 11] # パイロット位置
        self.num_symbols = 14
        
        # --- Sionna設定 ---
        self.rg = ResourceGrid(
            num_ofdm_symbols=self.num_symbols,
            fft_size=76, subcarrier_spacing=15e3,
            num_tx=1, num_streams_per_tx=4,
            cyclic_prefix_length=6, pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=self.pilot_indices
        )
        carrier_freq = 2.6e9
        ut_array = AntennaArray(num_rows=1, num_cols=2, polarization="dual", polarization_type="cross", antenna_pattern="38.901", carrier_frequency=carrier_freq)
        bs_array = AntennaArray(num_rows=1, num_cols=2, polarization="dual", polarization_type="cross", antenna_pattern="38.901", carrier_frequency=carrier_freq)
        
        # CDLチャネルモデル
        self.cdl = CDL("B", 300e-9, carrier_freq, ut_array, bs_array, "uplink", min_speed=10)
        self.frequencies = subcarrier_frequencies(self.rg.fft_size, self.rg.subcarrier_spacing)
        
        # 線形補間用の時間グリッド (GPU)
        self.t_grid = torch.arange(self.num_symbols, device=device).view(1, 1, -1, 1).float()

    def get_batch(self, snr_db=10.0):
        # 1. Sionnaで真のチャネル生成
        a, tau = self.cdl(batch_size=self.batch_size, 
                          num_time_steps=self.rg.num_ofdm_symbols, 
                          sampling_frequency=1/self.rg.ofdm_symbol_duration)
        
        h_freq = cir_to_ofdm_channel(self.frequencies, a, tau, normalize=True)
        
        # DLPackによるゼロコピー転送
        try:
            h_torch_complex = from_dlpack(tf.experimental.dlpack.to_dlpack(h_freq))
            h_torch_complex = h_torch_complex.cfloat() # 型変換
        except Exception:
            h_torch_complex = torch.from_numpy(h_freq.numpy()).cfloat().to(device)

        if h_torch_complex.device != device:
             h_torch_complex = h_torch_complex.to(device)
        
        # Reshape: [Batch, Rx, RxAnt, Tx, TxAnt, T, F] -> [Batch, Channels, T, F]
        b, rx, rx_ant, tx, tx_ant, t, f = h_torch_complex.shape
        h_gt_complex = h_torch_complex.view(b, rx*rx_ant*tx*tx_ant, t, f)

        # 2. パイロット抽出とノイズ付加
        t1, t2 = self.pilot_indices[0], self.pilot_indices[1]
        val_t1_clean = h_gt_complex[:, :, t1:t1+1, :]
        val_t2_clean = h_gt_complex[:, :, t2:t2+1, :]
        
        sig_pwr = torch.mean(torch.abs(h_gt_complex)**2)
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_pwr = sig_pwr / snr_linear
        noise_std = torch.sqrt(noise_pwr / 2.0)
        
        noise_n1 = (torch.randn_like(val_t1_clean) + 1j * torch.randn_like(val_t1_clean)) * noise_std
        noise_n2 = (torch.randn_like(val_t2_clean) + 1j * torch.randn_like(val_t2_clean)) * noise_std
        
        val_t1_noisy = val_t1_clean + noise_n1
        val_t2_noisy = val_t2_clean + noise_n2
        
        # 3. GPU上での高速線形補間
        slope = (val_t2_noisy - val_t1_noisy) / (t2 - t1)
        h_cond_complex = val_t1_noisy + slope * (self.t_grid - t1)
        
        # 4. 実部・虚部の結合
        x_gt = torch.cat([h_gt_complex.real, h_gt_complex.imag], dim=1).float()
        x_cond = torch.cat([h_cond_complex.real, h_cond_complex.imag], dim=1).float()
        
        return x_gt, x_cond

# ==========================================
# 2. モデル定義 (SNR条件付き)
# ==========================================
class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=64, cond_channels=64, time_emb_dim=32):
        super().__init__()
        total_in_channels = in_channels + cond_channels
        
        # Time Embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.time_proj = nn.Linear(time_emb_dim, 512)

        # ★SNR Embedding
        # SNR値をTimeと同じ次元へ埋め込み
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
        
        # Time Embedding
        t = t.float().view(-1, 1) / 1000.0
        t_emb = self.time_mlp(t)
        t_emb = self.time_proj(t_emb).view(-1, 512, 1, 1)

        # ★SNR Embedding
        # 正規化: 
        #   有効範囲(-5~30) -> -0.16 ~ 1.0 程度
        #   Blind(-100)     -> -3.33 程度
        #   MLPはこれらを十分に区別して学習可能
        s = snr.float().view(-1, 1) / 30.0 
        s_emb = self.snr_mlp(s)
        s_emb = self.snr_proj(s_emb).view(-1, 512, 1, 1)
        
        x1 = self.act(self.down1(x_in))
        x2 = self.act(self.down2(self.pool(x1)))
        
        # TimeとSNRの情報をBottleneckで注入
        x_bot = self.act(self.bot1(x2) + t_emb + s_emb)
        x_bot = self.act(self.bot2(x_bot))
        
        x_up = self.up1(x_bot)
        if x_up.shape != x1.shape:
             x_up = torch.nn.functional.interpolate(x_up, size=x1.shape[2:])
        
        x_dec = torch.cat([x_up, x1], dim=1)
        x_dec = self.act(self.conv1(x_dec))
        out = self.conv2(x_dec)
        
        return out

# common_utils.py の末尾に追加

@torch.no_grad()
def sample_ddpm(model, x_cond, snr_val, shape, device):
    """
    DDPMの逆拡散プロセス（推論）
    x_cond: 線形補間されたチャネル (条件)
    snr_val: 推論時のSNR設定値 (float)
    """
    model.eval()
    b = shape[0]
    
    # DDPMパラメータ (train.pyと同じ設定である必要があります)
    T = 1000
    beta = torch.linspace(1e-4, 0.02, T).to(device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    
    # SNR条件
    snr_tensor = torch.full((b,), snr_val, device=device)

    # 1. 完全なノイズからスタート
    x = torch.randn(shape, device=device)
    
    # 2. 逆拡散ループ (T -> 0)
    for i in reversed(range(0, T)):
        t_tensor = torch.full((b,), i, device=device, dtype=torch.long)
        
        # 現在のノイズ予測
        predicted_noise = model(x, x_cond, t_tensor, snr_tensor)
        
        # 係数計算
        curr_alpha = alpha[i]
        curr_alpha_bar = alpha_bar[i]
        
        # x_{t-1} の平均値を計算
        # mu = (1 / sqrt(alpha)) * (x_t - (beta / sqrt(1 - alpha_bar)) * noise)
        coef1 = 1 / torch.sqrt(curr_alpha)
        coef2 = beta[i] / torch.sqrt(1 - curr_alpha_bar)
        
        mean = coef1 * (x - coef2 * predicted_noise)
        
        if i > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(beta[i]) # シンプルなsigma設定
            x = mean + sigma * noise
        else:
            x = mean # 最後のステップはノイズを加えない

    return x