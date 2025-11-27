# Denoising System by Latent Diffusion Model

## Requirements
A suitable [conda](https://conda.io/) environment named `ldm` can be created
and activated with:

```
conda create --name ldm python=3.9
conda activate ldm
pip install -r requirements.txt
```


## Download the pre-trained weights (5.7GB)
```
mkdir -p models/ldm/text2img-large/
wget -O models/ldm/text2img-large/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
```
and sample with
```
python scripts/txt2img.py --prompt "a virus monster is playing guitar, oil on canvas" --ddim_eta 0.0 --n_samples 4 --n_iter 4 --scale 5.0  --ddim_steps 50
```
## Semantic Communication with Diffusion Model
The image you want to send should be put on "input_img" directory.
```
rm -r sentimg
```

```
python -m scripts.img2img
```

### Sent image
![original image](./figure_readme/sentimg_1.png)

### Latent space
The input image compressed by AutoEncoder.

### Received Image
The image with noise sampling. (channel noise = 0 dB, timestep = 400)

![image with sampling](./figure_readme/output_0_1.png)

The image by nosampling .

![image with nosampling](./figure_readme/output_0_1_nosampling.png)

## Result
I compared between total timestep.
![Result on varius total timestep](figure_readme/snr_vs_lpips_comparison.png)
I searched metric score during sampling
![Result on LPIPS](figure_readme/comparison_lpips_samplingeval.png)
![Result on PSNR](figure_readme/comparison_psnr_samplingeval.png)












