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
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    T = None # 固定タイムステップ
    t = 2
    N = t
    r = 2
    P_power = 1.0
    # python -m scripts.SU-MIMO > log_SU-MIMO_2_2.txt
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
        default=f"outputs/SU-MIMO/t={t}_r={r}_t={t}/T={T}"
    )

    parser.add_argument(
        "--nosample_outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default=f"outputs/SU-MIMO/t={t}_r={r}_t={t}/nonoisenosample"
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
        opt.outdir = f"outputs/SU-MIMO/t={t}_r={r}_t={t}/dynamic"
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


    batch_size = img.shape[0]

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
    z_w_size = z.shape[3]
    z_h_size = z.shape[2]
    # 複素数するのでsqrt(2)でも
    q_real_data = z/ torch.sqrt(2*(z_variance + eps)).view(-1, 1, 1, 1)
    # チャネル符号化
    q_view = q_real_data.view(batch_size, t, -1) # (B, K, Nsym)
    N_sym_total = q_view.shape[2]
    assert(N_sym_total % 2 == 0)
    l = N_sym_total // 2 # 送信データ長 (複素数)
    real_part, imag_part = torch.chunk(q_view, 2, dim=2)
    q = torch.complex(real_part, imag_part).to(device)  # (B, K, N_sym)
    # パイロット信号 P の生成 (フーリエ変換行列でバッチ共通)
    t_vec = torch.arange(t, device=device)
    N_vec = torch.arange(N, device=device)
    tt, NN = torch.meshgrid(t_vec, N_vec)
    P = torch.sqrt(torch.tensor(P_power/(N*t)))* torch.exp(1j*2*torch.pi*tt*NN/N)


    for snr in range(-5, 10, 1):
        print(f"--------SNR = {snr}-----------")
        # Noise = t * Noise/Signal
        noise_variance = t/(10**(snr/10))
        # チャネルH設定
        X = q # 送信信号 (B, t, l)
        H_real = torch.randn(batch_size, r, t) * torch.sqrt(torch.tensor(0.5))
        H_imag = torch.randn(batch_size, r, t) * torch.sqrt(torch.tensor(0.5))
        H = H_real + H_imag * 1j # (B, r, t)
        H = H.to(device)
        # ガウス雑音
        V_real = torch.randn(batch_size, r, N) * torch.sqrt(torch.tensor(0.5*noise_variance))
        V_imag = torch.randn(batch_size, r, N) * torch.sqrt(torch.tensor(0.5*noise_variance))
        V = V_real + V_imag * 1j # (B, r, N)
        V = V.to(device)
        S = H @ P + V #(B, r, N)

        # チャネル推定
        H_hat = S @ (P.mH @ torch.inverse(P@P.mH))
        H_tilde = H_hat - H
        

        # ここから実際の信号伝送

        # 通信ノイズ
        W_real = torch.randn(batch_size, r, l) * torch.sqrt(torch.tensor(0.5*noise_variance))
        W_imag = torch.randn(batch_size, r, l) * torch.sqrt(torch.tensor(0.5*noise_variance))
        W = W_real + W_imag * 1j # (r, l)
        W = W.to(device)

        #受信信号
        Y = H @ X + W
        #print(f"H@X var = {torch.var(H@X, dim=(1, 2))}")
        print(f"Real SNR = {10*torch.log10(torch.var(H@X, dim=(1, 2))/ torch.var(W,dim=(1, 2)))}")
        # ZFフィルタ
        A = torch.inverse(H_hat.mH@H_hat) @ H_hat.mH
        SINR = torch.var(X, dim=(1, 2)) / torch.var(A@(W-H_tilde@X), dim=(1, 2))
        # 完全な推定ができた場合
        A = torch.inverse(H.mH@H) @ H.mH
        SINR = torch.var(X, dim=(1, 2)) / torch.var(A@(W), dim=(1, 2))
        AY = A @ Y #(B, t, l)
        E = X - AY
        print(f"torch.linalg.det(H_hat.mH@H_hat) = {torch.linalg.det(H_hat.mH@H_hat)}")
        print(f"torch.linalg.det(H.mH@H) = {torch.linalg.det(H.mH@H)}")
        
        
        print(f"SINR {SINR.shape} = {10*torch.log10(SINR)}")
        print(f"E[SINR] = {1/(r - t+eps)}")
        E_mse = torch.mean(torch.abs(E)**2)
        print(f"E_MSE (Error Power) = {E_mse}")
        # --逆符号化--
        # 実部と虚部に分離
        # --逆符号化 (修正版)--
        # AY の形状: (B, t, l)

        # 1. 複素数を実数のペアに変換 -> 形状: (B, t, l, 2)
        AY_real_imag = torch.view_as_real(AY)

        # 2. 最後の次元(dim=3)から実部と虚部を正しく抽出
        #    real_part_restored の形状: (B, t, l)
        real_part_restored = AY_real_imag[..., 0]
        #    imag_part_restored の形状: (B, t, l)
        imag_part_restored = AY_real_imag[..., 1]

        # 3. 符号化時と「逆」の操作で、実部と虚部を dim=2 で結合
        #    q_view_restored の形状: (B, t, 2*l)  <- 元の q_view と同じ形状
        q_view_restored = torch.cat([real_part_restored, imag_part_restored], dim=2)

        # 4. 元の
        q_real_data_restored = q_view_restored.view(batch_size, z_channel, z_h_size, z_w_size)

        # 正規化を元に戻す
        z = q_real_data_restored * torch.sqrt(2*(z_variance + eps)).view(-1, 1, 1, 1)

        recoverd_img_no_samp = model.decode_first_stage(z)
        #save_img(recoverd_img_no_samp, f"outputs/nosample_{snr}.png")
        cond = model.get_learned_conditioning(z.shape[0] * [""])
        print(f"####cond finisihed #####")
        samples = sampler.MIMO_decide_starttimestep_ddim_sampling(S=opt.ddim_steps, batch_size=batch_size,
                        shape= z.shape[1:4],x_T=z,
                        conditioning=cond,starttimestep=T, noise_variance = 1/SINR)




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