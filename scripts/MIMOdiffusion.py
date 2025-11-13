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
    t = None # 固定タイムステップ
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
        default=f"outputs/MIMOdiffusion/t={t}"
    )
    
    parser.add_argument(
        "--nosample_outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default=f"outputs/MIMOdiffusion/nonoisenosample"
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
    if t == None or t <0:
        opt.outdir = "outputs/MIMOdiffusion/dynamic"
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
    K = 4
    M = 10
    tau_p = 20 # taup >= Kは必須。M >= にすると推定制度アップ

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
    q_view = q_real_data.view(batch_size, K, -1) # (B, K, Nsym)
    N_sym_total = q_view.shape[2]
    assert(N_sym_total % 2 == 0)
    N_sym = N_sym_total // 2
    real_part, imag_part = torch.chunk(q_view, 2, dim=2)
    q = torch.complex(real_part, imag_part).to(device)  # (B, K, N_sym)
    # eta (電力割り当て) - (B, K)
    # これをランダムにすると、性能が落ちる
    eta = torch.ones(batch_size, K, device=device, dtype=fdtype)
    # オプションA: 全要素に対する分散 (スカラー値が出力)
    real_var_all = torch.var(q.real)
    imag_var_all = torch.var(q.imag)

    print(f"Variance (Real, all elements): {real_var_all.item()}")
    print(f"Variance (Imag, all elements): {imag_var_all.item()}")
    for snr in range(-5, 10, 1):
        # SNR 15dBのときのノイズを乗せる snr = signal/noise
        
        rho_ul = 10.0 ** (snr / 10.0)  # 総受信SNR 
        rho_ul /=K
        print(f"--------SNR = {snr},  rho_ul = {rho_ul}-----------")
        # パイロット信号の生成 どのようなパイロットを使うかで性能が変わる
        # 全バッチで結合
        Phi = make_pilot(tau_p, K, device=device, dtype=cdtype)  # (tau_p, K)
        Phi_H = Phi.mH
        # beta 大規模フェージング
        beta_vals = [1.0 for x in range(K)]
        beta = torch.tensor(beta_vals, device=device, dtype=fdtype).view(1, 1, K).to(device)
        # G (チャネル行列) - (B, M, K)
        G_real = torch.sqrt(beta / 2) * torch.randn(batch_size, M, K, device=device)
        G_imag = torch.sqrt(beta / 2) * torch.randn(batch_size, M, K, device=device)
        G = torch.complex(G_real, G_imag).to(cdtype) # (B, M, K)
        G_variance = torch.var(G, dim=(1, 2))
        print(f"G_variance = {G_variance}")
        # W_p (パイロットノイズ) - (B, M, tau_p)
        W_p_real = torch.randn(batch_size, M, tau_p, device=device) * torch.sqrt(torch.tensor(0.5))
        W_p_imag = torch.randn(batch_size, M, tau_p, device=device) * torch.sqrt(torch.tensor(0.5))
        W_p = torch.complex(W_p_real, W_p_imag).to(cdtype) # (B, M, tau_p)
        # X_p (送信パイロット) - (K, tau_p)
        X_p = torch.sqrt(torch.tensor(tau_p, dtype=fdtype)) * Phi_H # (K, tau_p)
        # Y_p (受信パイロット) - (B, M, tau_p)
        # (B, M, K) @ (K, tau_p) = (B, M, tau_p)
        Y_p = torch.sqrt(torch.tensor(rho_ul)) * (G @ X_p) + W_p

        # Y_p_dash (受信機処理) - (B, M, K)
        # (B, M, tau_p) @ (tau_p, K) = (B, M, K)
        Y_p_dash = Y_p @ Phi
        numerator = torch.sqrt(torch.tensor(tau_p * rho_ul)) * beta
        denominator = 1 + tau_p * rho_ul * beta
        scaling_factor = (numerator / denominator).to(cdtype) # (1, 1, K)
        # ここで推定
        G_hat = scaling_factor * Y_p_dash # (B, M, K)
        G_hat_variance = torch.var(G_hat, dim=(1, 2))
        print(f"G_hat_variance = {G_hat_variance}")
        # gamma (推定チャネルの平均電力)
        gamma_numerator = tau_p * rho_ul * beta.pow(2)
        gamma_denominator = 1 + tau_p * rho_ul * beta
        gamma = (gamma_numerator / gamma_denominator) # (1, 1, K)
        G_tilde = G_hat - G # 実際のチャネル推定誤差

        print(f"G_tilde_var = {torch.var(G_tilde, dim=(1, 2))}")
        print(f"gamma = {gamma}")
        # Z (正規化チャネル推定値)
        sqrt_gamma = torch.sqrt(gamma).to(cdtype) # (1, 1, K)
        Z = G_hat / sqrt_gamma # (B, M, K)
        Z_variance = torch.var(Z, dim=(1,2))
        Z_mean = torch.mean(Z, dim=(1, 2))
        print(f"Z_mean = {Z_mean}")
        print(f"Z_variance ~CN(0, 1)= {Z_variance}")
        # ここからデータ送信

        # (B, K, K) の対角行列を作成
        D_eta_sqrt = torch.diag_embed(torch.sqrt(eta)).to(cdtype) # (B, K, K)

        # w (データノイズ) - (B, M, N_sym)
        w_real = torch.randn(batch_size, M, N_sym, device=device) / torch.sqrt(torch.tensor(2.0))
        w_imag = torch.randn(batch_size, M, N_sym, device=device) / torch.sqrt(torch.tensor(2.0))
        w = torch.complex(w_real, w_imag) # (B, M, N_sym)
        # y (受信データ) - (B, M, N_sym)
        # q (B, K, N_sym)
        # (B, M, K) @ (B, K, N_sym) = (B, M, N_sym)
        signal = torch.sqrt(torch.tensor(rho_ul)) * (G @ (D_eta_sqrt @ q))
        signal_var = torch.var(signal, dim=(1, 2))
        w_var = torch.var(w, dim=(1, 2))
        Real_SNR = 10.0 * torch.log10(signal_var/w_var)
        print(f"Real_SNR = {Real_SNR}")
        y = signal + w
        y_var = torch.var(y, dim=(1, 2))
        print(f"q_var = {torch.var(q, dim=(1, 2))}")
        
        print(f"(D_eta_sqrt @ q).var = {torch.var((D_eta_sqrt @ q), dim=(1, 2))}")
        print(f"(G @ (D_eta_sqrt @ q)).var = {torch.var((G @ (D_eta_sqrt @ q)), dim=(1, 2))}")
        print(f"signal_var = {signal_var}")
        print(f"w_var = {w_var}")
        print(f"y_var = {y_var}")

        # -- Zero-Forcing  --
        # A (ZFフィルタ): (B, M, K)
        Z_H = Z.mH # (B, K, M)
        Z_H_Z = Z_H @ Z # (B, K, K)
        inv_Z_H_Z = torch.linalg.inv(Z_H_Z) # (B, K, K)
        A = Z @ inv_Z_H_Z # (B, M, K)
        # AHy (等化後信号): (B, K, N_sym)
        AHy = A.mH @ y # (B, K, M) @ (B, M, N_sym)

        # q_receiver (復元データ)
        # (B, K, 1) に形状を整えてブロードキャスト
        gamma_scaled = gamma.permute(0, 2, 1).to(cdtype) # (1, 1, K) -> (1, K, 1)
        eta_scaled = eta.unsqueeze(-1).to(cdtype)      # (B, K) -> (B, K, 1)

        scaling_factor = torch.sqrt(rho_ul * gamma_scaled * eta_scaled) # (B, K, 1)
        q_receiver = AHy / scaling_factor # (B, K, N_sym)

        # --逆符号化--
        # --- 1. 実部と虚部を分離 ---
        # q の形状: (B, K, N_sym) = (10, 4, 512)
        real_part_restored = torch.real(q_receiver)
        imag_part_restored = torch.imag(q_receiver)
        q_view_restored = torch.cat([real_part_restored, imag_part_restored], dim=2)
        # q_view_restores = (B, K, 2*N_sym)
        q_real_data_restored = q_view_restored.view(batch_size, z_channel, z_h_size, z_w_size)


        # 非正規化
        z = q_real_data_restored * torch.sqrt(2*(z_variance + eps)).view(-1, 1, 1, 1)

        

        recoverd_img_no_samp = model.decode_first_stage(z)
        #save_img(recoverd_img_no_samp, f"outputs/nosample_{snr}.png")
        cond = model.get_learned_conditioning(z.shape[0] * [""])
        print(f"####cond finisihed #####")
        #Z_H_Z の形状: (B, K, K)
        # 各バッチの対角成分(K, K)を取得し、絶対値をとる
        # Z_H_Z_k の形状: (B, K)
        inv_Z_H_Z_k = torch.diagonal(inv_Z_H_Z, dim1=-2, dim2=-1).abs()

        # beta, gamma の形状: (1, 1, K)
        # squeeze() で不要な次元を削除 -> (K,)
        beta_sq = beta.squeeze()   # (K,)
        gamma_sq = gamma.squeeze() # (K,)
        
        # eta の形状: (B, K)

        # 1. (beta_k - gamma_k) を計算
        # diff の形状: (K,)
        diff = beta_sq - gamma_sq
        
        # 2. (beta_k - gamma_k) * eta_bk を計算
        # (K,) * (B, K) -> (B, K) (diff が (B, K) にブロードキャストされる)
        prod = diff * eta
        
        # 3. Kの次元で総和を取る: sum_{k=0}^{K-1} ...
        # sum_term の形状: (B,)
        sum_term = torch.sum(prod, dim=1)
        print(f"sum_betak_gammak = {sum_term}")
        # 4. 1.0 を足す (term の計算)
        # term の形状: (B,)
        term = 1.0 + rho_ul*sum_term
        
        # Var の計算
        # term を (B, 1) に変形してブロードキャスト
        # (B, 1) * (B, K) -> (B, K)
        # Var の形状: (B, K)
        Var = term.unsqueeze(-1) * inv_Z_H_Z_k

        print(f"************")
        # Z_H_Z_k は (B, K) のテンソルとして表示
        print(f"inv_Z_H_Z_k (shape: {inv_Z_H_Z_k.shape}) = {inv_Z_H_Z_k}")
        # Var は (B, K) のテンソルとして表示
        print(f"Var (shape: {Var.shape}) = {Var}")
        print(f"***********")
        signal_power = rho_ul * gamma.squeeze() * eta
        Var = signal_power / Var
        SINR_linear = torch.mean(Var, dim=1) # ユーザ間で平均（線形値）
        SINR = 10.0 * torch.log10(SINR_linear) # dBに変換
        

        # Real SNRの計算
        effective_noise = A.mH@(w - torch.sqrt(torch.tensor(rho_ul))*G_tilde@(D_eta_sqrt@q))
        desired_signal = torch.sqrt(torch.tensor(rho_ul))*A.mH@G_hat @ D_eta_sqrt @ q
        effective_noise_var = torch.var(effective_noise, dim=(1, 2))
        desired_signal_var = torch.var(desired_signal, dim=(1, 2))
        Real_SINR = 10.0 * torch.log10(desired_signal_var / effective_noise_var)
        
        #print(f"mean_perbatch {mean_per_batch.shape} = {mean_per_batch}")
        print(f"SINR = {SINR}")
        print(f"Real_SINR = {Real_SINR}")
        Real_SINR_noise = effective_noise_var / desired_signal_var
        print(f"Real_SINR_noise.shape = {Real_SINR_noise.shape}")
       

        



        samples = sampler.MIMO_decide_starttimestep_ddim_sampling(S=opt.ddim_steps, batch_size=batch_size,
                        shape= z.shape[1:4],x_T=z,
                        conditioning=cond,starttimestep=t, noise_variance = 1.0/SINR_linear)




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