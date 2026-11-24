# Can All-in-One Restorers Adapt to Atmospheric Turbulence?
Sanjana Nandi

## Problem
All-in-one image restoration models (PromptIR, AdaIR, MoCE-IR, DOD, R2R) handle several
degradations with a single network, but none of them are trained on atmospheric turbulence.
Turbulence is different from the degradations they do see (noise, rain, haze): besides blurring
the image, it applies spatially varying geometric distortion (tilt), so pixels are displaced rather
than only changed in value.

This project asks two questions:
1. How well do pretrained all-in-one restorers handle turbulence they were never trained on?
2. Can a test-time adaptation method (DCTTA) close the gap without retraining?

A turbulence-specific model (DATUM) serves as a reference point for what a specialist achieves.

## Programming Language and Tools
- Python, PyTorch (MPS / CPU on Apple silicon), PyTorch Lightning
- [pixi](https://pixi.sh) for per-model environments (one workspace, one environment per model)
- A CUDA GPU is required for DATUM (custom CUDA kernel) and is expected for DCTTA at full scale

## Data
| Dataset | Type | Ground truth | Notes |
|---|---|---|---|
| [ATSyn-static](https://github.com/xg416/DATUM) | Synthetic | Clean image per scene | 50 turbulent frames per scene; separate tilt-only and blur-only versions; turbulence strength (D/r0) per scene |
| [ATSyn-dynamic](https://github.com/xg416/DATUM) | Synthetic video | Clean frames | Moving scenes |
| [OTIS](https://zenodo.org/records/161439) | Real turbulence | Clean chart patterns | Static chart targets imaged through real turbulence |

DATUM is trained on ATSyn, so OTIS is the fairer test of generalization.

### Pretrained Models to Download
| Model | Venue | Weights | Notes |
|---|---|---|---|
| [PromptIR](https://github.com/va1shn9v/PromptIR) | NeurIPS 2023 | 3-task all-in-one checkpoint ([Drive](https://drive.google.com/file/d/1j-b5Od70pGF7oaCqKAfUzmf-N-xEAjYl/view)) | Runs on MPS |
| [AdaIR](https://github.com/c-yn/AdaIR) | ICLR 2025 | 3-task checkpoint ([Drive](https://drive.google.com/drive/folders/1x2LN4kWkO3S65jJlH-1INUFiYt8KFzPH)) | CPU on Mac (MPS lacks complex FFT in torch 2.0) |
| [MoCE-IR](https://github.com/eduardzamfir/MoCE-IR) | CVPR 2025 | `MoCE_IR_AIO3/last.ckpt` ([Drive](https://drive.google.com/drive/folders/1pQBceb8cCPdIzbqbNNGqV5qNXzzqL4uK)) | |
| [DOD](https://github.com/tonia86/DOD) | AAAI 2026 | `stage1.pkl`, `stage2.pkl` ([Drive](https://drive.google.com/drive/folders/1O0BKy9qfWKwJrLwkIl3mG9xx_Z7NLkm-)); SD 2.1-base ([HuggingFace](https://huggingface.co/Manojb/stable-diffusion-2-1-base)); `256x256_diffusion_uncond.pt` ([OpenAI guided-diffusion](https://drive.google.com/file/d/1j-b5Od70pGF7oaCqKAfUzmf-N-xEAjYl/view)) | SD 2.1 was deprecated by Stability AI; uses a Hugging Face duplicate |
| [R2R](https://github.com/cscxwang/R2R) | CVPR 2026 | `train_ckpt_3D_finetune/` + `save_prompts_3D/` ([Drive](https://drive.google.com/drive/folders/1KAAafchWfLCYthhyRUcc1Xz4iip3kWqF)) | CPU on Mac (MPS adaptive-pooling limit) |
| [DATUM](https://github.com/xg416/DATUM) | CVPR 2024 | Static-scene model (repo README) | Turbulence specialist; needs CUDA |

Test-time adaptation: **[DCTTA](https://github.com/tonia86/DCTTA)** (CVPR 2026).

## Project Tasks
- [ ] Set up an environment for each model and run it on one turbulence image, plus a noise sanity check
- [ ] Build fixed evaluation sets (ATSyn-static subset; OTIS sequences)
- [ ] Zero-shot evaluation of the five restorers, with DATUM as a reference (GPU)
- [ ] Compute metrics: PSNR, SSIM, CW-SSIM, LPIPS (reference); NIQE, BRISQUE, MUSIQ (no reference)
- [ ] Apply DCTTA to each restorer and re-evaluate

## Next Steps
Run each of the models with the fixed evaluation sets.