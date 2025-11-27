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
import lpips
from common_utils import SionnaChannelGeneratorGPU, ConditionalUNet, sample_ddpm, device
import torch.nn.functional as F
import tensorflow as tf
import matplotlib.pyplot as plt
import gc # メモリ解放用

# Sionna imports
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, OFDMModulator, OFDMDemodulator, ResourceGridDemapper
from sionna.phy.channel import ApplyOFDMChannel, AWGN
from sionna.phy.mimo import StreamManagement

# 環境設定
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def get_image_paths(dir_path):
    """ディレクトリから画像パスのリストのみを取得する"""
    image_paths = []
    supported_formats = ["*.jpg", "*.jpeg", "*.png"]
    for fmt in supported_formats:
        image_paths.extend(glob.glob(os.path.join(dir_path, fmt)))
    
    if not image_paths:
        print(f"警告: ディレクトリ '{dir_path}' にサポートされている画像ファイルが見つかりません。")
        return []
    
    # 順序を保証するためにソート
    image_paths.sort()
    return image_paths

def load_images_from_paths(paths, image_size=(256, 256)):
    """指定されたパスのリストから画像を読み込んでテンソル化する"""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    
    tensors_list = []
    # バッチごとの読み込みなのでtqdmは不要または軽量化
    for path in paths:
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

def save_img_individually(img, path, start_index=0):
    """
    画像を保存する。バッチ処理に対応し、通し番号(start_index)を加算する。
    """
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    dirname = os.path.dirname(path)
    basename = os.path.splitext(os.path.basename(path))[0]
    ext = os.path.splitext(path)[1]
    os.makedirs(dirname, exist_ok=True)
    batch_size = img.shape[0]
    
    for i in range(batch_size):
        # グローバルなインデックスを使用してファイル名を作成
        global_idx = start_index + i
        individual_path = os.path.join(dirname, f"{basename}_{global_idx}{ext}")
        vutil.save_image(img[i], individual_path)
    # print(f"Saved {batch_size} images to {dirname}/ starting from index {start_index}")

def remove_png(path):
    png_files = glob.glob(f'{path}/*.png')
    for file in png_files:
        try:
            os.remove(f"{file}")
        except OSError as e:
            print(f"Error: {e.strerror} - {e.filename}")
    print(f"remove_png complete")

def manual_zf_equalization_with_error(y_received_grid, h_est_tf, h_true_tf, rg, no):
    # (既存のコードと同じため省略なしで記述)
    # 1. Arrange axes
    h_est_sq = tf.squeeze(h_est_tf, axis=[1, 3]) 
    h_est_perm = tf.transpose(h_est_sq, perm=[0, 3, 4, 1, 2]) 
    
    h_true_sq = tf.squeeze(h_true_tf, axis=[1, 3])
    h_true_perm = tf.transpose(h_true_sq, perm=[0, 3, 4, 1, 2])

    y_sq = tf.squeeze(y_received_grid, axis=1) 
    y_perm = tf.transpose(y_sq, perm=[0, 2, 3, 1]) 
    y_perm = tf.expand_dims(y_perm, axis=-1)

    # 2. Compute ZF Combiner W using Estimated Channel
    h_est_herm = tf.linalg.adjoint(h_est_perm)
    gram = tf.matmul(h_est_herm, h_est_perm)
    gram_inv = tf.linalg.inv(gram) 
    W = tf.matmul(gram_inv, h_est_herm) 

    # 3. Compute Effective Noise Variance
    h_err = h_true_perm - h_est_perm
    h_err_herm = tf.linalg.adjoint(h_err)
    cov_h_err = tf.matmul(h_err, h_err_herm)
    
    rx_dim = tf.shape(h_est_perm)[-2]
    eye_rx = tf.eye(rx_dim, batch_shape=tf.shape(h_est_perm)[:-2], dtype=cov_h_err.dtype)
    cov_noise = tf.cast(no, dtype=cov_h_err.dtype) * eye_rx
    
    cov_inner = cov_h_err + cov_noise
    
    w_herm = tf.linalg.adjoint(W)
    cov_eff = tf.matmul(tf.matmul(W, cov_inner), w_herm)
    
    diag_eff = tf.linalg.diag_part(cov_eff)
    effective_noise_var = tf.reduce_mean(tf.math.real(diag_eff), axis=[1, 2, 3])
    
    # 4. Equalization
    x_hat_perm = tf.matmul(W, y_perm)
    x_hat_perm = tf.squeeze(x_hat_perm, axis=-1)
    x_hat_reshaped = tf.transpose(x_hat_perm, perm=[0, 3, 1, 2])
    x_hat = tf.expand_dims(x_hat_reshaped, axis=1)
    
    return x_hat, effective_noise_var

def torch_to_tf_channel(h_torch_complex, batch_size, rg):
    h_np = h_torch_complex.detach().cpu().numpy()
    h_reshaped = h_np.reshape(batch_size, 1, 4, 1, 4, rg.num_ofdm_symbols, rg.fft_size)
    return tf.constant(h_reshaped, dtype=tf.complex64)

def reconstruct_z(q_hat_tf, z_variance, transmit_data_size, batch_size, channel_size, h_size, w_size):
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
    # Tはここで指定せず、ループ内で制御することも可能ですが、元のロジックを尊重
    T = None 
    parser.add_argument("--prompt", type=str, nargs="?", default="a painting of a virus monster playing guitar")
    # outdirはmodeによって動的に変更するため、デフォルト値はプレースホルダー的に扱います
    parser.add_argument("--base_outdir", type=str, default="outputs/OFDM")
    parser.add_argument("--sentimgdir", type=str, nargs='?', default="./sentimg")
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--plms", action='store_true')
    parser.add_argument("--input_path", type=str, default="input_img")
    parser.add_argument("--intermediate_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for processing")
    
    opt = parser.parse_args()
    if opt.intermediate_path != None:
        os.makedirs(opt.intermediate_path, exist_ok=True)
    
    # モデルのロードはループの外で一度だけ行う（時間短縮）
    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml") 
    model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt") 

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    # チャネル推定モデルのロードもループの外で行う
    model_channel = ConditionalUNet(in_channels=32, cond_channels=32).to(device)
    CHECKPOINT_PATH = "ckpt_step_190000.pth"
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        model_channel.load_state_dict(ckpt['model_state_dict'])
    else:
        print("Checkpoint not found! Please run train.py first.")
        raise ValueError

    # 全画像のパスを取得
    all_image_paths = get_image_paths(opt.input_path)
    total_images = len(all_image_paths)
    print(f"Total images found: {total_images}")

    # モードのリスト
    MODES = ["Perfect_CSI", "Linear_Interp", "Diffusion"]

    # --- Mode Loop ---
    for mode in MODES:
        print(f"\n{'='*20} Starting Mode: {mode} {'='*20}")
        
        # モードごとの出力ディレクトリ設定
        if T == None or T < 0:
            current_outdir = f"{opt.base_outdir}/{mode}_dynamic"
            current_nosample_outdir = f"{opt.base_outdir}/{mode}_dynamic/nodiffusion"
        else:
            current_outdir = f"{opt.base_outdir}/{mode}_T={T}"
            current_nosample_outdir = f"{opt.base_outdir}/{mode}_nodiffusion"

        os.makedirs(current_outdir, exist_ok=True)
        os.makedirs(opt.sentimgdir, exist_ok=True)
        os.makedirs(current_nosample_outdir, exist_ok=True)

        # --- Batch Loop ---
        # 画像パスリストをバッチサイズごとに分割して処理
        for i in range(0, total_images, opt.batch_size):
            batch_paths = all_image_paths[i : i + opt.batch_size]
            current_batch_idx = i # 現在のバッチの開始インデックス
            
            print(f"\nProcessing batch {current_batch_idx} to {current_batch_idx + len(batch_paths)} (Total: {total_images})")
            
            # 画像読み込み
            img = load_images_from_paths(batch_paths)
            if img.shape[0] == 0:
                continue

            BATCH_SIZE = img.shape[0] # 最後のバッチは20未満になる可能性があるため動的に取得
            
            # --- 画像保存処理の修正案 ---
            # 最初のモードの時だけ、元画像を保存する（無駄な上書きを防ぐため）
            if mode == MODES[0]: 
                save_img_individually(img, opt.sentimgdir + "/sentimg.png", start_index=current_batch_idx)
            img = img.to(device=device)
            
            # Encode
            with torch.no_grad():
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
            
            # Generator must be initialized with current batch size
            generator = SionnaChannelGeneratorGPU(batch_size=BATCH_SIZE)
            
            # --- SNR Loop ---
            for snr in range(-5, 10, 1):
                print(f"--- Mode: {mode} | Batch: {current_batch_idx//opt.batch_size + 1} | SNR = {snr} ---")
                
                # Channel Simulation
                x_gt, x_cond = generator.get_batch(snr_db=snr)
                
                # Diffusion based channel estimation (using model_channel loaded outside)
                with torch.no_grad():
                    x_est_diff = sample_ddpm(model_channel, x_cond, snr, x_gt.shape, device)
                
                def to_complex(x):
                    c = x.shape[1] // 2
                    return torch.complex(x[:, :c, ...], x[:, c:, ...])
                
                h_perfect_torch = to_complex(x_gt)
                h_linear_torch = to_complex(x_cond)
                h_diff_torch = to_complex(x_est_diff)
                
                # MSE Calculation (logging only)
                mse_linear = torch.mean(torch.abs(h_perfect_torch - h_linear_torch)**2).item()
                mse_diff = torch.mean(torch.abs(h_perfect_torch - h_diff_torch)**2).item()
                
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

                def evaluate_method(name, h_est_tf, h_true_tf):
                    x_hat, eff_noise_var_batch = manual_zf_equalization_with_error(y_rg_tf, h_est_tf, h_true_tf, rg, no)
                    q_hat_tf = demapper(x_hat)
                    z_hat = reconstruct_z(q_hat_tf, z_variance, transmit_data_size, 
                                        BATCH_SIZE, CHANNEL_SIZE, H_SIZE, W_SIZE)
                    mse_val = F.mse_loss(z, z_hat).item()
                    
                    eff_noise_var_np = eff_noise_var_batch.numpy()
                    sinr_batch = 1.0 / (eff_noise_var_np + 1e-12)
                    scale_sq_batch = 2 * z_variance.detach().cpu().numpy()
                    z_noise_var_batch = eff_noise_var_np * scale_sq_batch
                    
                    return mse_val, z_hat, sinr_batch, z_noise_var_batch

                # Evaluate all methods to get stats, but only use 'mode' for reconstruction
                mse_perf, z_hat_perf, sinr_perf_batch, var_perf_batch = evaluate_method("Perfect CSI", h_perfect_tf, h_perfect_tf)
                mse_lin, z_hat_lin, sinr_lin_batch, var_lin_batch = evaluate_method("Linear Interp", h_linear_tf, h_perfect_tf)
                mse_diff, z_hat_diff, sinr_diff_batch, var_diff_batch = evaluate_method("Diffusion", h_diff_tf, h_perfect_tf)
                
                # Select target based on current 'mode' loop
                if mode == "Perfect_CSI":
                    target_z_hat = z_hat_perf
                    target_noise_var = var_perf_batch
                elif mode == "Linear_Interp":
                    target_z_hat = z_hat_lin
                    target_noise_var = var_lin_batch
                elif mode == "Diffusion":
                    target_z_hat = z_hat_diff
                    target_noise_var = var_diff_batch
                
                # Decode
                with torch.no_grad():
                    recoverd_img_no_samp = model.decode_first_stage(target_z_hat)
                    cond = model.get_learned_conditioning(z.shape[0] * [""])
                    
                    target_noise_var_torch = torch.from_numpy(target_noise_var).float().to(device)
                    
                    samples = sampler.MIMO_decide_starttimestep_ddim_sampling(S=opt.ddim_steps, batch_size=BATCH_SIZE,
                                    shape= z.shape[1:4], x_T=target_z_hat,
                                    conditioning=cond, starttimestep=T, noise_variance=target_noise_var_torch)

                    recoverd_img = model.decode_first_stage(samples)
                
                # Save Images with global index
                # output_{snr}.png という名前のプレースホルダーを渡し、関数内でインデックスが付与される
                # 例: output_-5_0.png, output_-5_1.png ... (次のバッチ) output_-5_20.png ...
                save_img_individually(recoverd_img, f"{current_outdir}/output_{snr}.png", start_index=current_batch_idx)
                save_img_individually(recoverd_img_no_samp, f"{current_nosample_outdir}/output_{snr}.png", start_index=current_batch_idx)

            # --- End of SNR Loop ---
            
            # メモリ解放 (バッチ終了時)
            del img, z, q, generator, x_gt, x_cond, x_est_diff, h_perfect_torch, h_linear_torch, h_diff_torch
            del h_perfect_tf, h_linear_tf, h_diff_tf, y_rg_tf, x_rg_tf
            del recoverd_img, recoverd_img_no_samp, samples, cond
            del target_z_hat, z_hat_perf, z_hat_lin, z_hat_diff
            
            torch.cuda.empty_cache()
            gc.collect()

        # --- End of Batch Loop ---
    # --- End of Mode Loop ---
    
    print("\nAll simulations completed.")