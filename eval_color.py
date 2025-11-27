import os
import argparse
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- New Imports for LPIPS ---
try:
    import torch
    import lpips
except ImportError:
    print("Warning: 'torch' or 'lpips' libraries not found.")
    print("To use the LPIPS metric, please install them: pip install torch lpips")
    torch = None
    lpips = None
# -------------------------------


def np_to_torch(img_np):
    """
    Converts a NumPy image (H, W, C) in range [0, 255]
    to a PyTorch tensor (N, C, H, W) in range [-1, 1].
    """
    img_tensor = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    img_tensor = (img_tensor / 127.5) - 1.0
    return img_tensor

def compute_metric(x, y, metric='ssim', lpips_model=None, device=None):
    """
    Computes the similarity/error between image pair x, y.
    """
    if metric == 'ssim':
        data_range = float(x.max() - x.min())
        if data_range == 0:
            return 1.0
        return ssim(x, y, channel_axis=-1, data_range=data_range)

    xd = x.astype(np.float64)
    yd = y.astype(np.float64)
    mse = float(np.mean((xd - yd) ** 2))

    if metric == 'mse':
        return mse
    
    elif metric == 'psnr':
        if mse == 0:
            return np.inf
        max_pixel = 255.0
        psnr = 10 * np.log10((max_pixel ** 2) / mse)
        return float(psnr)
        
    elif metric == 'lpips':
        if lpips_model is None or device is None:
            raise ValueError("lpips_model and device must be provided for LPIPS metric.")
        
        tensor_x = np_to_torch(x).to(device)
        tensor_y = np_to_torch(y).to(device)
        
        with torch.no_grad():
            dist = lpips_model(tensor_x, tensor_y)
        return float(dist.item())

    else:
        raise ValueError("Metric must be 'ssim', 'mse', 'psnr', or 'lpips'.")

def calculate_snr_vs_metric(sent_path, received_path, metric='ssim', resize=(256,256), lpips_model=None, device=None):
    """
    Compares images in the sent and received directories.
    """
    dic_sum = {}
    dic_num = {}

    if not os.path.isdir(sent_path):
        print(f"Error: Directory not found: {sent_path}")
        return [], []
    if not os.path.isdir(received_path):
        print(f"Error: Directory not found: {received_path}")
        return [], []

    # print(f"Processing comparison between '{sent_path}' and '{received_path}'... (metric={metric})")

    for sp in os.listdir(sent_path):
        if not sp.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        img_id = "".join(filter(str.isdigit, sp))
        if not img_id:
            continue

        sent_image_path = os.path.join(sent_path, sp)

        for rp in os.listdir(received_path):
            if not rp.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            if img_id in rp:
                try:
                    parts = os.path.splitext(rp)[0].split('_')
                    if len(parts) < 2:
                        continue

                    rimg_id_part = "".join(filter(str.isdigit, parts[-1]))
                    snr_str = parts[-2]

                    if rimg_id_part == img_id:
                        sentimg = Image.open(sent_image_path).convert('RGB')
                        recimg = Image.open(os.path.join(received_path, rp)).convert('RGB')

                        if resize is not None:
                            sentimg = sentimg.resize(resize)
                            recimg = recimg.resize(resize)

                        sentarr = np.array(sentimg)
                        recarr = np.array(recimg)

                        try:
                            val = compute_metric(sentarr, recarr, metric=metric, lpips_model=lpips_model, device=device)
                        except Exception as e:
                            print(f"Warning: Error during metric calculation ({rp}): {e}")
                            continue

                        dic_sum[snr_str] = dic_sum.get(snr_str, 0.0) + val
                        dic_num[snr_str] = dic_num.get(snr_str, 0) + 1
                except Exception:
                    continue

    if not dic_sum:
        print(f"Warning: No matching images found in '{received_path}'.")
        return [], []

    xy = []
    for snr_key, total in dic_sum.items():
        try:
            snr_float = float("".join(filter(lambda c: c.isdigit() or c in '.-', snr_key)))
            count = dic_num[snr_key]
            avg = total / count
            xy.append((snr_float, avg))
        except (ValueError, ZeroDivisionError):
            continue
    
    xy.sort()
    x_vals = [item[0] for item in xy]
    y_vals = [item[1] for item in xy]
    return x_vals, y_vals

# -----------------------------------------------------------------
# --- ⭐ Path Parsing & Plotting Logic ---
# -----------------------------------------------------------------

def parse_path_info(path):
    """
    Parses the directory path to extract Category (Group) and Condition (Subgroup).
    Logic:
      - If path ends in 'nodiffusion', 
          Category = Parent Folder Name
          Condition = 'nodiffusion'
      - Otherwise,
          Category = Current Folder Name
          Condition = 'Proposed' (or 'default')
    """
    norm_path = os.path.normpath(path)
    base_name = os.path.basename(norm_path)
    
    if base_name == 'nodiffusion':
        # Parent directory is the category
        category = os.path.basename(os.path.dirname(norm_path))
        condition = 'nodiffusion'
    else:
        # The directory itself is the category
        category = base_name
        condition = 'Proposed' # Label for the standard/diffusion case
        
    return category, condition

def plot_results(results_data, title_suffix="", output_filename="snr_vs_metric.png"):
    """
    Plots the results with semantic coloring and markers.
    results_data: list of dicts {'x':[], 'y':[], 'label':str, 'category':str, 'condition':str}
    """
    
    # 1. Identify unique categories and conditions to assign consistent styles
    unique_categories = sorted(list(set(d['category'] for d in results_data)))
    unique_conditions = sorted(list(set(d['condition'] for d in results_data)))
    
    # 2. Define Style Maps
    # Colors for Categories
    base_colors = list(mcolors.TABLEAU_COLORS.values()) # 10 colors
    # Map each category string to a color
    cat_color_map = {cat: base_colors[i % len(base_colors)] for i, cat in enumerate(unique_categories)}
    
    # Markers/Lines for Conditions
    # Example: 'Proposed' -> Circle (o) + Solid Line (-)
    #          'nodiffusion' -> Cross (x) + Dashed Line (--)
    available_markers = ['o', 'x', 's', '^', 'D', 'v', '<', '>']
    available_linestyles = ['-', '--', '-.', ':']
    
    cond_style_map = {}
    for i, cond in enumerate(unique_conditions):
        # Specific overrides for better readability based on user request
        if cond == 'nodiffusion':
            marker = 'x'
            linestyle = '--' # Dashed for nodiffusion
        elif cond == 'Proposed':
            marker = 'o'
            linestyle = '-'  # Solid for proposed
        else:
            # Fallback for other potential folder names
            marker = available_markers[i % len(available_markers)]
            linestyle = available_linestyles[i % len(available_linestyles)]
            
        cond_style_map[cond] = {'marker': marker, 'linestyle': linestyle}

    plt.figure(figsize=(10, 6))
    
    for data in results_data:
        x_vals = data['x']
        y_vals = data['y']
        cat = data['category']
        cond = data['condition']
        
        if not x_vals:
            continue
            
        # Get styles
        color = cat_color_map.get(cat, 'black')
        style = cond_style_map.get(cond, {'marker':'o', 'linestyle':'-'})
        
        # Construct Label: "Category" or "Category (Condition)"
        # If there is only one condition total, maybe just show Category. 
        # But here we likely have mixed, so show both.
        label_text = f"{cat} ({cond})"
        
        plt.plot(x_vals, y_vals, 
                 marker=style['marker'], 
                 linestyle=style['linestyle'], 
                 label=label_text, 
                 color=color, 
                 markersize=6, 
                 linewidth=1.5)
    
    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel(f"Metric value {title_suffix}", fontsize=12)
    plt.title(f"SNR vs. Metric Comparison {title_suffix}", fontsize=14)
    
    # Legend handling
    if len(results_data) > 6:
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize='small')
    else:
        plt.legend()
         
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"\nPlot saved as '{output_filename}'.")

# -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SNR vs SSIM/MSE/PSNR/LPIPS comparison script")
    parser.add_argument("--sent", "-s", default="./sentimg", help="Directory for 'sent' (original) images")
    parser.add_argument("--metric", "-m", choices=["ssim","mse","psnr","lpips","all"], default="ssim", help="Metric to use (ssim, mse, psnr, lpips, or all)")
    parser.add_argument("--resize", type=int, nargs=2, metavar=('W','H'), default=(256,256), help="Resize dimensions for comparison (W H)")
    parser.add_argument("recv_dirs", nargs='+', metavar='RECV_DIR', help="One or more directories for 'received' images to compare against 'sent'")

    args = parser.parse_args()
    recv_paths = args.recv_dirs
    
    if not recv_paths:
        print("Error: No received directories specified.")
        return

    # Parse inputs to get metadata for plotting later
    # We do this before the loop to establish categories/conditions if needed globally, 
    # but here just verifying paths is enough.
    
    metrics_to_run = ["ssim", "mse", "psnr", "lpips"] if args.metric == "all" else [args.metric]

    lpips_model = None
    device = None
    if "lpips" in metrics_to_run:
        if lpips is None or torch is None:
            print("Error: LPIPS requested but libraries missing.")
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nInitializing LPIPS model (AlexNet) on device: {device}")
        lpips_model = lpips.LPIPS(net='alex').to(device).eval()

    for metric in metrics_to_run:
        print(f"\n==========================================")
        print(f" PROCESSING METRIC: {metric.upper()} ")
        print(f"==========================================")
        
        plot_data_list = [] 

        for recv_path in recv_paths:
            # Parse directory structure to get label info
            category, condition = parse_path_info(recv_path)
            label = f"{category} ({condition})"
            
            print(f"\n--- Calculating {metric.upper()} for {label} ---")
            print(f"    Path: {recv_path}")
            
            x_vals, y_vals = calculate_snr_vs_metric(
                args.sent, recv_path, metric=metric, resize=tuple(args.resize), 
                lpips_model=lpips_model, device=device
            )
            
            if x_vals: 
                plot_data_list.append({
                    'x': x_vals, 
                    'y': y_vals, 
                    'label': label,
                    'category': category,
                    'condition': condition
                })
            else:
                print(f"Warning: No valid data found for {recv_path}.")

        if not plot_data_list:
            print(f"\nNo data to plot for metric '{metric}'.")
            continue

        outname = f"snr_vs_{metric}_comparison.png"
        plot_results(plot_data_list, title_suffix=f"({metric.upper()})", output_filename=outname)

    print("\nAll processing complete.")

if __name__ == "__main__":
    main()

"""
python eval_color.py --metric psnr outputs/OFDM/Perfect_CSI_dynamic/nodiffusion outputs/OFDM/Perfect_CSI_dynamic outputs/OFDM/Diffusion_dynamic/nodiffusion outputs/OFDM/Diffusion_dynamic  outputs/OFDM/Linear_Interp_dynamic/nodiffusion/ outputs/OFDM/Linear_Interp_dynamic/ 


"""