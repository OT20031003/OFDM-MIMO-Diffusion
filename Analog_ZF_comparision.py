import os
import torch
import torch.nn.functional as F # 追加: パディング用
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

# Sionna imports
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, OFDMModulator, OFDMDemodulator, ResourceGridDemapper
from sionna.phy.channel import ApplyOFDMChannel, AWGN
from sionna.phy.mimo import StreamManagement

# 自作モジュールのインポート
from common_utils import SionnaChannelGeneratorGPU, ConditionalUNet, sample_ddpm, device

# 環境設定
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def manual_zf_equalization(y_received_grid, h_est_tf, rg):
    """
    手動ZF等化を行う関数
    """
    # 軸の整理 [batch, Time, Freq, RxAnt, Stream]
    h_sq = tf.squeeze(h_est_tf, axis=[1, 3]) 
    h_perm = tf.transpose(h_sq, perm=[0, 3, 4, 1, 2]) 

    # Y -> [batch, Time, Freq, RxAnt, 1]
    y_sq = tf.squeeze(y_received_grid, axis=1) 
    y_perm = tf.transpose(y_sq, perm=[0, 2, 3, 1]) 
    y_perm = tf.expand_dims(y_perm, axis=-1)

    # ZF計算: (H^H * H)^(-1) * H^H * y
    h_herm = tf.linalg.adjoint(h_perm)
    gram = tf.matmul(h_herm, h_perm)
    
    # 逆行列
    gram_inv = tf.linalg.inv(gram) 
    
    h_inv_manual = tf.matmul(gram_inv, h_herm)
    x_hat_perm = tf.matmul(h_inv_manual, y_perm)

    # 形状を戻す [B, 1, Stream, T, F]
    x_hat_perm = tf.squeeze(x_hat_perm, axis=-1)
    x_hat_reshaped = tf.transpose(x_hat_perm, perm=[0, 3, 1, 2])
    x_hat = tf.expand_dims(x_hat_reshaped, axis=1)
    
    return x_hat

def torch_to_tf_channel(h_torch_complex, batch_size, rg):
    """PyTorchチャネル -> TensorFlowチャネル変換"""
    h_np = h_torch_complex.detach().cpu().numpy()
    h_reshaped = h_np.reshape(batch_size, 1, 4, 1, 4, rg.num_ofdm_symbols, rg.fft_size)
    return tf.constant(h_reshaped, dtype=tf.complex64)

def reconstruct_z(q_hat_tf, z_variance, transmit_data_size, batch_size, channel_size, h_size, w_size):
    """
    復調されたシンボルから元の信号zを復元する関数
    (test.pyの受信側ロジックに相当)
    """
    # TF -> Torch
    q_hat_torch = torch.from_numpy(q_hat_tf.numpy()).to(device)
    q_hat_flat = q_hat_torch.view(batch_size, -1)

    # Remove Padding
    # test.pyのロジック: if total_capacity >= transmit_data_size...
    q_hat_valid = q_hat_flat[:, :transmit_data_size]

    # Reconstruction (Complex -> Real concat -> Scale)
    real_part_hat = q_hat_valid.real
    imag_part_hat = q_hat_valid.imag
    q_view_hat = torch.cat((real_part_hat, imag_part_hat), dim=1)
    
    eps = 0.000001
    scale_factor = torch.sqrt(2 * (z_variance + eps)).view(-1, 1)

    z_hat_flat = q_view_hat * scale_factor
    z_hat = z_hat_flat.view(batch_size, channel_size, h_size, w_size)
    
    return z_hat

def main():
    # --- 設定 ---
    BATCH_SIZE = 32
    SNR_DB = 25.0 # 比較しやすいように少し上げても良いです
    CHECKPOINT_PATH = "ckpt_step_135000.pth"
    
    # zデータの次元設定 (test.pyと同じ)
    CHANNEL_SIZE = 4
    H_SIZE = 32
    W_SIZE = 32

    # ジェネレータとモデル
    generator = SionnaChannelGeneratorGPU(batch_size=BATCH_SIZE)
    model = ConditionalUNet(in_channels=32, cond_channels=32).to(device)
    
    # ロード
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading model from {CHECKPOINT_PATH}...")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        print("Checkpoint not found! Please run train.py first.")
        return

    # --- 1. チャネル生成 ---
    print("Generating Channel Data...")
    x_gt, x_cond = generator.get_batch(snr_db=SNR_DB)
    
    # --- 2. Diffusion推論 ---
    print("Running Diffusion Inference...")
    x_est_diff = sample_ddpm(model, x_cond, SNR_DB, x_gt.shape, device)
    
    # --- 3. チャネル整形 (PyTorch) ---
    def to_complex(x):
        c = x.shape[1] // 2
        return torch.complex(x[:, :c, ...], x[:, c:, ...])

    h_perfect_torch = to_complex(x_gt)
    h_linear_torch = to_complex(x_cond)
    h_diff_torch = to_complex(x_est_diff)
    
    # チャネルMSE
    mse_linear = torch.mean(torch.abs(h_perfect_torch - h_linear_torch)**2).item()
    mse_diff = torch.mean(torch.abs(h_perfect_torch - h_diff_torch)**2).item()
    print(f"\nChannel MSE (Linear): {mse_linear:.5f}")
    print(f"\nChannel MSE (Diffusion): {mse_diff:.5f}")

    # --- 4. 送信データ生成 (Analog z generation: test.py style) ---
    print("Generating Analog Data (z)...")
    rg = generator.rg
    
    # (A) z生成
    z = torch.randn((BATCH_SIZE, CHANNEL_SIZE, H_SIZE, W_SIZE)).to(device)
    z_variance = torch.var(z, dim=(1,2,3)) # 受信側でのスケーリング復元に使用
    eps = 0.000001
    
    # (B) 正規化と複素数化
    q_real_data = z / torch.sqrt(2*(z_variance + eps)).view(-1, 1, 1, 1)
    q_view = q_real_data.view(BATCH_SIZE, -1)
    real_part, imag_part = torch.chunk(q_view, 2, dim=1)
    q = torch.complex(real_part, imag_part)
    
    # (C) パディング
    n_data_per_stream = rg.num_data_symbols
    num_streams = rg.num_streams_per_tx
    total_capacity = n_data_per_stream * num_streams
    transmit_data_size = q.shape[1]

    if total_capacity >= transmit_data_size:
        padding_size = total_capacity - transmit_data_size
        q_padded = F.pad(q, (0, padding_size), mode='constant', value=0)
    else:
        raise ValueError("Data size exceeds grid capacity.")

    # [Batch, 1, Stream, Data]
    q_reshaped = q_padded.view(BATCH_SIZE, 1, num_streams, n_data_per_stream)
    
    # (D) Sionnaへ渡すためにNumpy化
    q_np = q_reshaped.detach().cpu().numpy()
    
    # Mapper
    mapper = ResourceGridMapper(rg)
    x_rg_tf = mapper(q_np) # [B, 1, 4, T, F]

    # --- 5. 通信路適用 (Perfect CSIで受信信号作成) ---
    h_perfect_tf = torch_to_tf_channel(h_perfect_torch, BATCH_SIZE, rg)
    channel_applier = ApplyOFDMChannel(add_awgn=True)
    no = 10**(-SNR_DB / 10)
    y_rg_tf = channel_applier(x_rg_tf, h_perfect_tf, no)

    # --- 6. 比較実行 ---
    
    # 準備
    h_linear_tf = torch_to_tf_channel(h_linear_torch, BATCH_SIZE, rg)
    h_diff_tf = torch_to_tf_channel(h_diff_torch, BATCH_SIZE, rg)
    rx_tx_association = np.array([[1]])
    sm = StreamManagement(rx_tx_association, num_streams_per_tx=4)
    demapper = ResourceGridDemapper(rg, stream_management=sm)

    results = {}
    
    # 共通処理関数
    def evaluate_method(name, h_est_tf):
        # 1. ZF Equalization
        x_hat = manual_zf_equalization(y_rg_tf, h_est_tf, rg)
        # 2. Demapping
        q_hat_tf = demapper(x_hat)
        # 3. Reconstruction z
        z_hat = reconstruct_z(q_hat_tf, z_variance, transmit_data_size, 
                              BATCH_SIZE, CHANNEL_SIZE, H_SIZE, W_SIZE)
        # 4. MSE Calculation
        mse_val = F.mse_loss(z, z_hat).item()
        return mse_val, z_hat

    # Case 1: Perfect CSI
    mse_perf, z_hat_perf = evaluate_method("Perfect CSI", h_perfect_tf)
    results['Perfect CSI'] = mse_perf

    # Case 2: Linear Interp
    mse_lin, z_hat_lin = evaluate_method("Linear Interp", h_linear_tf)
    results['Linear Interp'] = mse_lin

    # Case 3: Diffusion
    mse_diff, z_hat_diff = evaluate_method("Diffusion", h_diff_tf)
    results['Diffusion'] = mse_diff

    # --- 7. 結果表示 ---
    print("-" * 40)
    print(f"Reconstructed z MSE (SNR={SNR_DB}dB)")
    print("-" * 40)
    for method, val in results.items():
        print(f"{method:<20}: {val:.5f}")
    print("-" * 40)
    
    # --- 8. 可視化 ---
    # アナログ信号なのでコンスタレーションはガウス分布の雲のようになりますが、
    # 復元誤差の傾向を見るために散布図を作成します（zの特定チャネルの一部を表示）
    
    plt.figure(figsize=(15, 5))
    titles = ['Perfect CSI', 'Linear Interp', 'Diffusion']
    z_hats = [z_hat_perf, z_hat_lin, z_hat_diff]
    
    # 可視化用にデータをフラット化して一部を抽出
    z_flat = z.detach().cpu().numpy().flatten()[:1000] # 見やすくするため先頭1000点
    
    for i, (tit, z_h) in enumerate(zip(titles, z_hats)):
        plt.subplot(1, 3, i+1)
        z_h_flat = z_h.detach().cpu().numpy().flatten()[:1000]
        
        # 横軸: 送信z, 縦軸: 受信z (理想は y=x の直線)
        plt.scatter(z_flat, z_h_flat, alpha=0.3, s=5)
        
        # y=x 線
        min_val = min(z_flat.min(), z_h_flat.min())
        max_val = max(z_flat.max(), z_h_flat.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
        
        plt.title(f"{tit}\nMSE: {results[tit]:.4f}")
        plt.xlabel("Original z")
        plt.ylabel("Reconstructed z")
        plt.grid(True)

    plt.tight_layout()
    plt.savefig("z_reconstruction_comparison.png")
    print("Comparison plot saved to z_reconstruction_comparison.png")

if __name__ == "__main__":
    main()