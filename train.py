# train.py
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# common_utils.py からクラスと定数をインポート
from common_utils import SionnaChannelGeneratorGPU, ConditionalUNet, device, UNKNOWN_SNR_VALUE

def save_checkpoint(model, optimizer, scheduler, step, loss_history, filepath):
    """チェックポイント保存用関数（スケジューラも保存）"""
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(), # 追加: スケジューラの状態
        'loss_history': loss_history
    }
    torch.save(checkpoint, filepath)

def train():
    # --- 引数設定 ---
    parser = argparse.ArgumentParser(description="Train DDPM for Channel Estimation with Resume & Blind SNR")
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # --- ハイパーパラメータ ---
    BATCH_SIZE = 256  
    TOTAL_STEPS = 200000 
    LR = 1e-4
    MIN_LR = 1e-6           # 追加: 最終的な最小学習率
    SNR_DROPOUT_PROB = 0.15  # 15%の確率でSNRを隠す(Blind学習)
    
    # 保存ディレクトリ作成
    os.makedirs("checkpoints", exist_ok=True)
    
    torch.backends.cudnn.benchmark = True
    
    # --- 準備 ---
    generator = SionnaChannelGeneratorGPU(batch_size=BATCH_SIZE)
    model = ConditionalUNet(in_channels=64, cond_channels=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss()

    # ★追加: 学習率スケジューラ (Cosine Annealing)
    # TOTAL_STEPS かけて、LR から MIN_LR まで徐々に下げる
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=TOTAL_STEPS, 
        eta_min=MIN_LR
    )
    
    # DDPM パラメータ
    T = 1000
    beta = torch.linspace(1e-4, 0.02, T).to(device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    
    loss_history = []
    start_step = 0

    # --- レジューム処理 ---
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}...")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 追加: スケジューラの復元（もし保存されていれば）
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            start_step = checkpoint['step']
            loss_history = checkpoint.get('loss_history', [])
            print(f"Resumed from step {start_step}.")
        else:
            print(f"Error: Checkpoint file {args.resume} not found.")
            return

    print(f"Starting Training on {device} from step {start_step} to {TOTAL_STEPS}...")
    model.train()
    
    # tqdmの設定（開始位置を調整）
    pbar = tqdm(range(start_step, TOTAL_STEPS), initial=start_step, total=TOTAL_STEPS)
    
    for step in pbar:
        # 1. SNRのランダム決定 
        current_snr_val = np.random.uniform(-5.0, 30.0)
        
        # データ生成
        x_gt, x_cond = generator.get_batch(snr_db=current_snr_val)
        
        # 2. ノイズ付加
        t = torch.randint(0, T, (BATCH_SIZE,)).to(device)
        epsilon = torch.randn_like(x_gt)
        
        sqrt_ab = torch.sqrt(alpha_bar[t]).view(-1, 1, 1, 1)
        sqrt_onem_ab = torch.sqrt(1 - alpha_bar[t]).view(-1, 1, 1, 1)
        
        x_noisy = sqrt_ab * x_gt + sqrt_onem_ab * epsilon
        
        # 3. SNR条件の作成 (Dropout適用)
        snr_tensor = torch.full((BATCH_SIZE,), current_snr_val, device=device)
        if np.random.rand() < SNR_DROPOUT_PROB:
            snr_tensor.fill_(UNKNOWN_SNR_VALUE)

        # 4. 予測 & ロス計算
        pred_noise = model(x_noisy, x_cond, t, snr_tensor)
        loss = mse(pred_noise, epsilon)
        
        # 5. 更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # ★追加: スケジューラのステップ更新
        scheduler.step()
        
        # ログ
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        # 現在の学習率を取得して表示
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_description(f"Loss: {loss_val:.5f} | LR: {current_lr:.2e}")
        
        # 定期保存
        if (step + 1) % 5000 == 0:
            save_path = f"checkpoints/ckpt_step_{step+1}.pth"
            save_checkpoint(model, optimizer, scheduler, step + 1, loss_history, save_path)
            
            plt.figure()
            plt.plot(loss_history)
            plt.title("Training Loss")
            plt.yscale('log') # Lossが見やすいように対数表示も検討（任意）
            plt.savefig("loss_curve.png")
            plt.close()

    # 最終保存
    save_checkpoint(model, optimizer, scheduler, TOTAL_STEPS, loss_history, "diff_model_final.pth")
    print("Training Finished. Model saved.")
    
    plt.figure()
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.savefig("loss_curve.png")

if __name__ == "__main__":
    train()