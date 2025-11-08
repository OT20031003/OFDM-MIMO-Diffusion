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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    t = 0 # 意図的に与えるタイムステップ
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
        default=f"outputs/jointdiffusion/t={t}"
    )
    parser.add_argument(
        "--nosample_outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default=f"outputs/jointdiffusion/nonoisenosample/t={t}"
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


    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    all_samples=list()
    # 画像をロード
    remove_png(opt.outdir)
    
    img = load_images_as_tensors(opt.input_path)
    print(f"img shape = {img.shape}")
    save_img_individually(img, opt.sentimgdir + "/sentimg.png")
    img = img.to(device="cuda")
    z = model.encode_first_stage(img)
    # detachはVAEの重みを固定するため
    print(f"encode start = ")
    z = model.get_first_stage_encoding(z).detach()
    print(f"z = {z.shape}, z_max = {z.max()}, z_min = {z.min()}")
    z_variances_original = torch.var(z, dim=(1, 2, 3))
    print(f"z_variance_original = {z_variances_original}")
    z = sampler.forward_diffusion(S=opt.ddim_steps, batch_size=z.shape[0], timestep=t, x = z)
    
    # チャネル符号化 (複素数)
    assert(z.shape[1] % 2 == 0)
    real_part, img_part = torch.chunk(z, 2, dim = 1)

    complex_z = torch.complex(real_part, img_part)
    # 正規化
    z_mean = complex_z.mean(dim=(1, 2, 3), keepdim=True)
    z_variances_with_noise = torch.var(complex_z, dim=(1, 2, 3)) 
    eps = 0.0000001
    complex_z = complex_z - z_mean
    normalized_z = complex_z / torch.sqrt((z_variances_with_noise + eps)).view(-1, 1, 1, 1)
    print(f"complex_z = {torch.var(normalized_z, dim=(1, 2, 3))}")
    z_copy = normalized_z
    h = torch.randn(size=()) + torch.randn(size=())*1j
    h = h.to(device)
    for snr in range(-5, 10, 1):
        # SNR 15dBのときのノイズを乗せる snr = signal/noise
        print(f"--------SNR = {snr}-----------")
        z = z_copy
        snrp = pow(10, snr/10) 
        z_variances = torch.var(z, dim=(1, 2, 3))
        noise_variances = z_variances/snrp # ノイズ分散 正解の分散
        scalar_variance = 1 / snrp # = Noise/Signal
        # 9. 「複素ノイズ」を生成します
        #    Var(Real) = Var(Imag) = noise_variances / 2 となるようにスケーリング
        noise_std = torch.sqrt(noise_variances / 2.0).view(-1, 1, 1, 1) # (B, 1, 1, 1)
        
        # z.real と同じ形状の実数・虚数ノイズを生成
        noise_real = torch.randn_like(z.real) * noise_std
        noise_imag = torch.randn_like(z.real) * noise_std
        noise = torch.complex(noise_real, noise_imag)
        
        print(f"noise_variace = {noise_variances}, z_variance = {z_variances}")
        z = z + noise *1/h
        #save_img(z, f"outputs/z_{snr}.png")
        # 非正規化
        z = torch.sqrt(z_variances_with_noise.view(-1, 1, 1, 1)) * z + z_mean
        # channel decoding
        z = torch.cat([z.real, z.imag], dim=1)
        
        recoverd_img_no_samp = model.decode_first_stage(z)
        #save_img(recoverd_img_no_samp, f"outputs/nosample_{snr}.png")  
        cond = model.get_learned_conditioning(z.shape[0] * [""])
        print(f"####cond finisihed #####")
        

        

        samples = sampler.jointdiffusion_ddim_sampling(S=opt.ddim_steps, batch_size=z.shape[0], 
                        shape= z.shape[1:4],x_T=z,noise_sigma_predict = noise_variances,added_timestep= t,
                        conditioning=cond,intermediate_path=opt.intermediate_path, intermediate_skip=opt.intermediate_skip, snr = snr)

    
        
    
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
