# inference.py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# common_utils からクラスと定数をインポート
from common_utils import SionnaChannelGeneratorGPU, ConditionalUNet, device, UNKNOWN_SNR_VALUE

def compute_nmse(true, est):
    """Normalized Mean Square Error (dB)"""
    mse = torch.mean((true - est)**2, dim=[1, 2, 3])
    power = torch.mean(true**2, dim=[1, 2, 3])
    nmse = 10 * torch.log10(mse / power)
    return nmse.mean().item()

@torch.no_grad()
def sample_ddpm(model, x_cond, shape, device, snr_tensor):
    """DDPMの逆拡散プロセス"""
    T = 1000
    beta = torch.linspace(1e-4, 0.02, T).to(device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    
    model.eval()
    
    x = torch.randn(shape, device=device)
    
    # 逆拡散ループ
    for t in tqdm(reversed(range(T)), desc="Sampling", total=T, leave=False):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        
        # ★ここでSNRテンソルを渡す
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

def main():
    # --- 設定 ---
    # テストしたい物理的なSNR (シミュレーション用)
    TEST_SNR_DB = 5.0       
    BATCH_SIZE = 16
    MODEL_PATH = "checkpoints/ckpt_step_130000.pth"
    
    print(f"Running Inference (Blind Mode) on True SNR = {TEST_SNR_DB} dB...")

    # --- 1. モデルとジェネレータの準備 ---
    generator = SionnaChannelGeneratorGPU(batch_size=BATCH_SIZE)
    # 正解データ作成には実際のSNRを使う
    x_gt, x_cond = generator.get_batch(snr_db=TEST_SNR_DB)
    
    model = ConditionalUNet(in_channels=64, cond_channels=64).to(device)
    try:
        # 重みのみロードする（strict=Falseにはしない、構造は合っている前提）
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        # チェックポイント辞書か、state_dict直接かを判定
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: {MODEL_PATH} not found.")
        return

    # --- 2. 推論 (サンプリング) ---
    print("Starting Reverse Diffusion Process...")
    
    # ★Blind Inference: SNRがわからないものとして -100.0 を入力する
    blind_snr_tensor = torch.full((BATCH_SIZE,), UNKNOWN_SNR_VALUE, device=device)
    #blind_snr_tensor = torch.full((BATCH_SIZE,), TEST_SNR_DB, device=device)
    x_estimated = sample_ddpm(model, x_cond, x_gt.shape, device, blind_snr_tensor)

    # --- 3. 評価 (NMSE) ---
    nmse_linear = compute_nmse(x_gt, x_cond)
    nmse_diff = compute_nmse(x_gt, x_estimated)

    print("\n" + "="*30)
    print(f"Results (True SNR: {TEST_SNR_DB} dB)")
    print("="*30)
    print(f"Linear Interpolation NMSE : {nmse_linear:.4f} dB")
    print(f"Diffusion Model (Blind)   : {nmse_diff:.4f} dB")
    print(f"Improvement               : {nmse_linear - nmse_diff:.4f} dB")
    print("="*30)

    # --- 4. 可視化 ---
    idx = 32
    gt_img = x_gt[0, idx, :, :].cpu().numpy()
    cond_img = x_cond[0, idx, :, :].cpu().numpy()
    est_img = x_estimated[0, idx, :, :].cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    vmin = min(gt_img.min(), cond_img.min(), est_img.min())
    vmax = max(gt_img.max(), cond_img.max(), est_img.max())

    im1 = axes[0].imshow(gt_img, aspect='auto', vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth (Clean)")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(cond_img, aspect='auto', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Linear Interp (Input)\nNMSE: {nmse_linear:.2f}dB")
    fig.colorbar(im2, ax=axes[1])

    im3 = axes[2].imshow(est_img, aspect='auto', vmin=vmin, vmax=vmax)
    axes[2].set_title(f"Diff Model (Blind)\nNMSE: {nmse_diff:.2f}dB")
    fig.colorbar(im3, ax=axes[2])

    plt.suptitle(f"Blind Channel Estimation (True SNR={TEST_SNR_DB}dB)")
    plt.tight_layout()
    plt.savefig(f"result_snr{int(TEST_SNR_DB)}_blind.png")
    print(f"Result image saved.")

if __name__ == "__main__":
    main()