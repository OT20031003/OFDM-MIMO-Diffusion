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
import glob # ファイルパスのリストを正規表現で取得するために使用
import lpips
# 自作モジュールのインポート
from common_utils import SionnaChannelGeneratorGPU, ConditionalUNet, sample_ddpm, device
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

def load_images_as_tensors(dir_path, image_size=(256, 256)):
    """
    指定されたディレクトリ内のすべての画像ファイルを読み込み、
    PyTorchテンソルのリストとして返す。

    Args:
        dir_path (str): 画像が格納されているディレクトリのパス。
        image_size (tuple): リサイズ後の画像サイズ (高さ, 幅)。

    Returns:
        list: 画像テンソルのリスト。
              画像が1枚もなかった場合は空のリストを返す。
    """
    # 1. 変換処理の定義
    #    - 画像を (image_size) にリサイズ
    #    - テンソルに変換（値が [0.0, 1.0] に正規化され、次元が [C, H, W] になる）
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    image_paths = []
    supported_formats = ["*.jpg", "*.jpeg", "*.png"]
    for fmt in supported_formats:
        # glob.globは条件に合うファイルパスのリストを返す
        image_paths.extend(glob.glob(os.path.join(dir_path, fmt)))

    if not image_paths:
        print(f"警告: ディレクトリ '{dir_path}' にサポートされている画像ファイルが見つかりません。")
        return []

    # 3. 各画像をループで読み込み、テンソルに変換してリストに格納
    tensors_list = []
    for t in trange(len(image_paths), desc="Loading Image"):
        path = image_paths[t]
        try:
            # 画像を開き、RGB形式に統一
            img = Image.open(path).convert("RGB")
            # 変換処理を適用
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

def save_img(img, path):
    if len(img.shape) == 3:
        # [3,256,256]のとき
        img.unsqueeze_(0)

    # (batch, channel, h, w)をまとめて一枚の画像に保存
    vutil.save_image(img, path, nrow=4)
    print(f"images are saved in {path}")

def save_img_individually(img, path):
    """
    バッチ画像を個別のファイルとして1枚ずつ保存する関数。

    Args:
        img (torch.Tensor): 保存する画像のテンソル (B, C, H, W) or (C, H, W)
        path (str): 保存先のパス。例: "output/result.png"
    """
    # 3次元テンソル [C, H, W] の場合は、バッチ次元 [B] を追加して4次元に統一
    if len(img.shape) == 3:
        img = img.unsqueeze(0)

    # 保存先ディレクトリ、ベースとなるファイル名、拡張子を取得
    # 例: path="output/result.png" -> dirname="output", basename="result", ext=".png"
    dirname = os.path.dirname(path)
    basename = os.path.splitext(os.path.basename(path))[0]
    ext = os.path.splitext(path)[1]

    # 保存先ディレクトリが存在しない場合は作成
    os.makedirs(dirname, exist_ok=True)

    # バッチ内の画像を1枚ずつループして保存
    batch_size = img.shape[0]
    for i in range(batch_size):
        # 連番付きの新しいファイルパスを生成
        # 例: "output/result_0.png", "output/result_1.png", ...
        individual_path = os.path.join(dirname, f"{basename}_{i}{ext}")

        # i番目の画像テンソルを取得して保存
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

def caluc_lpips(x,y):
    loss_fn = lpips.LPIPS(net='alex')
    d = loss_fn(x, y)
    return d.item()
def make_pilot(tau_p, K, device, dtype):
    """
    (tau x K) のユニタリ行列 Phi をPyTorchで生成
    """
    assert(tau_p >= K)
    # torch.randnで実部と虚部を生成
    A_real = torch.randn(tau_p, K, device=device)
    A_imag = torch.randn(tau_p, K, device=device)
    # torch.complexで複素数テンソルに
    A = torch.complex(A_real, A_imag).to(dtype)
    # torch.linalg.qrでQR分解
    Q_pilot, R = torch.linalg.qr(A, mode='reduced')
    return Q_pilot
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    T = 400 # 固定タイムステップ
    #mode = "Linear_Interp"
    mode = "Diffusion"
    # python -m scripts.SU-MIMO > log_SU-MIMO_2_2_perfect.txt
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default=f"outputs/OFDM/{mode}_T={T}"
    )

    parser.add_argument(
        "--nosample_outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default=f"outputs/OFDM/{mode}_nodiffusion"
    )

    parser.add_argument(
        "--sentimgdir",
        type=str,
        nargs='?',
        help="sent img dir path",
        default="./sentimg"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--input_path",
        type=str,
        default="input_img",
        help="input image path"
    )
    parser.add_argument(
        "--intermediate_path",
        type=str,
        default=None,
        help="intermediate path"
    )
    parser.add_argument(
        "--intermediate_skip",
        type=int,
        default=1,
        help="intermediate path"
    )
    opt = parser.parse_args()
    if opt.intermediate_path != None:
        os.makedirs(opt.intermediate_path, exist_ok=True)
        print(f"{opt.intermediate_path} is created new")
    
    if T == None or T <0:
        opt.outdir = f"outputs/OFDM/{mode}_dynamic"
        opt.nosample_outdir = f"outputs/OFDM/{mode}_dynamic/nodiffusion"
        
    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    # ldm.modules.diffusion.ddpmをロード
    model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt")  # TODO: check path

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    os.makedirs(opt.sentimgdir, exist_ok=True)
    os.makedirs(opt.nosample_outdir, exist_ok=True)
    outpath = opt.outdir

    prompt = opt.prompt
    cdtype = torch.complex64
    fdtype = torch.float32

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    all_samples=list()
    # 画像をロード
    remove_png(opt.outdir)
    eps = 0.0000001
    img = load_images_as_tensors(opt.input_path)


    BATCH_SIZE = img.shape[0]

    print(f"img shape = {img.shape}")
    save_img_individually(img, opt.sentimgdir + "/sentimg.png")
    img = img.to(device=device)
    z = model.encode_first_stage(img)
    # detachはVAEの重みを固定するため
    print(f"encode start = ")
    z = model.get_first_stage_encoding(z).detach()
    z_variance = torch.var(z, dim=(1, 2, 3))
    print(f"z_variance before normalization = {z_variance}")
    z_channel = z.shape[1]
    CHANNEL_SIZE = z.shape[1]
    z_w_size = z.shape[3]
    W_SIZE = z.shape[3]
    H_SIZE = z.shape[2]
    z_h_size = z.shape[2]
    # 複素数するのでsqrt(2)でも
    q_real_data = z/ torch.sqrt(2*(z_variance + eps)).view(-1, 1, 1, 1)
    q_view = q_real_data.view(BATCH_SIZE, -1)
    real_part, imag_part = torch.chunk(q_view, 2, dim=1)
    q = torch.complex(real_part, imag_part)
    # ジェネレータとモデル
    generator = SionnaChannelGeneratorGPU(batch_size=BATCH_SIZE)
    model_channel = ConditionalUNet(in_channels=32, cond_channels=32).to(device)
    # ロード
    CHECKPOINT_PATH = "ckpt_step_135000.pth"
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading model from {CHECKPOINT_PATH}...")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        model_channel.load_state_dict(ckpt['model_state_dict'])
    else:
        print("Checkpoint not found! Please run train.py first.")
        raise ValueError
    

    for snr in range(-5, 10, 1):
        print(f"--------SNR = {snr}-----------")
        # チャネル生成
        x_gt, x_cond = generator.get_batch(snr_db=snr)

        # Diffusion 推論
        x_est_diff = sample_ddpm(model_channel, x_cond, snr, x_gt.shape, device)
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
        rg = generator.rg
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
        no = 10**(-snr / 10)
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
        print(f"Reconstructed z MSE (SNR={snr}dB)")
        print("-" * 40)
        for method, val in results.items():
            print(f"{method:<20}: {val:.5f}")
        print("-" * 40)
        
        if mode == "Perfect_CSI":
            recoverd_img_no_samp = model.decode_first_stage(z_hat_diff)
            #save_img(recoverd_img_no_samp, f"outputs/nosample_{snr}.png")
            cond = model.get_learned_conditioning(z.shape[0] * [""])
            print(f"####cond finisihed #####")
            samples = sampler.MIMO_decide_starttimestep_ddim_sampling(S=opt.ddim_steps, batch_size=BATCH_SIZE,
                            shape= z.shape[1:4],x_T=z_hat_perf,
                            conditioning=cond,starttimestep=T, noise_variance = None)

        if mode == "Linear_Interp":
            recoverd_img_no_samp = model.decode_first_stage(z_hat_lin)
            #save_img(recoverd_img_no_samp, f"outputs/nosample_{snr}.png")
            cond = model.get_learned_conditioning(z.shape[0] * [""])
            print(f"####cond finisihed #####")
            samples = sampler.MIMO_decide_starttimestep_ddim_sampling(S=opt.ddim_steps, batch_size=BATCH_SIZE,
                            shape= z.shape[1:4],x_T=z_hat_lin,
                            conditioning=cond,starttimestep=T, noise_variance = None)

        if mode == "Diffusion":
            recoverd_img_no_samp = model.decode_first_stage(z_hat_diff)
            #save_img(recoverd_img_no_samp, f"outputs/nosample_{snr}.png")
            cond = model.get_learned_conditioning(z.shape[0] * [""])
            print(f"####cond finisihed #####")
            samples = sampler.MIMO_decide_starttimestep_ddim_sampling(S=opt.ddim_steps, batch_size=BATCH_SIZE,
                            shape= z.shape[1:4],x_T=z_hat_diff,
                            conditioning=cond,starttimestep=T, noise_variance = None)

        
        



        print(f"d = {samples.shape}")
        recoverd_img = model.decode_first_stage(samples)
        #print(f"LPIPS = {caluc_lpips(recoverd_img, img.to(device))}")
        print(f"recoverd_img = {recoverd_img.shape}")
        save_img_individually(recoverd_img, f"{opt.outdir}/output_{snr}.png")
        save_img_individually(recoverd_img_no_samp, f"{opt.nosample_outdir}/output_{snr}.png")




    # additionally, save as grid
    # grid = torch.stack(all_samples, 0)
    # grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    # grid = make_grid(grid, nrow=opt.n_samples)

    # # to image
    # grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    # Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'{prompt.replace(" ", "-")}.png'))

    # print(f"Your samples are ready and waiting four you here: \n{outpath} \nEnjoy.")