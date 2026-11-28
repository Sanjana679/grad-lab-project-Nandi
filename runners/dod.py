import os
import sys

sys.path.append(os.getcwd())
import argparse
import glob

# torchvision >= 0.17 removed functional_tensor; basicsr 1.4.2 still imports it
import sys
import types

import torch
import torchvision.transforms.functional as F
from guided_diffusion.script_util import i_DDPM

try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except ImportError:
    import torchvision.transforms.functional as _TF
    _shim = types.ModuleType("torchvision.transforms.functional_tensor")
    _shim.rgb_to_grayscale = _TF.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = _shim
    
from osediff import OSEDiff_test
from PIL import Image
from torch import nn
from torchvision import transforms
from tqdm import tqdm

dev_name = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device(dev_name)


####调用neg_prompt####
def load_neg_prompt_embeds(load_path="prompt/neg_prompt_embeds77.pt", device=device):
    if os.path.exists(load_path):
        return torch.load(load_path, map_location=device)
    else:
        raise FileNotFoundError(f"Negative prompt embeddings file not found at {load_path}")


tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def denormalize(tensor, mean, std):
    """将图像从标准化状态恢复到 [0, 1] 范围"""
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean

def freeze_model_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

def get_validation_prompt(image, device=device):
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    return lq


IMG_EXTS = (".png", ".jpg", ".jpeg")

def list_inputs(input_path):
    """Accepts a glob ('.../*/turb/0.png'), a folder, or a single image."""
    if os.path.isdir(input_path):
        input_path = os.path.join(input_path, "*")
    paths = sorted(p for p in glob.glob(input_path, recursive=True)
                   if p.lower().endswith(IMG_EXTS))
    if not paths:
        raise FileNotFoundError(f"No images matched: {input_path}")
    # root = shared parent of all inputs; outputs keep the structure below it
    root = os.path.dirname(paths[0]) if len(paths) == 1 else os.path.commonpath(paths)
    return paths, root


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', '-i', type=str, default='../../datasets/input/', help='glob, directory, or single image (quote globs!)')
    parser.add_argument('--output_path', '-o', type=str, default='../../datasets/output/dod/', help='the directory to save the output')
    parser.add_argument("--nafnet_path", type=str, default='/tn/xmu/code/DOD/experience/osediff_stage2/checkpoints/model_NAFNet_1.pkl')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default="/tn/xmu/model/sd21/", help='sd model path')#sd2.1
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument("--process_size", type=int, default=256)    
    parser.add_argument("--upscale", type=int, default=1)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='wavelet')
    parser.add_argument("--osediff_path", type=str, default='/tn/xmu/code/DOD/experience/osediff/checkpoints/model_1.pkl') #/data/tn/work3_cl/osediff/experience/s3diff_ddpm_sft/model_200001.pkl
    parser.add_argument('--prompt', type=str, default='', help='user prompts')
    parser.add_argument('--save_prompts', type=bool, default=True)
    # precision setting (fp32 default: fp16 isn't supported for the CPU/MPS path)
    parser.add_argument("--mixed_precision", type=str, choices=['fp16', 'fp32'], default="fp32")
    # merge lora
    parser.add_argument("--merge_and_unload_lora", default=False) # merge lora weights before inference
    # tile setting
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224) 
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024) 
    parser.add_argument("--latent_tiled_size", type=int, default=96) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=32) 
    parser.add_argument("--lora_rank", default=8, type=int)

    args = parser.parse_args()

    print("device:", device)

    # initialize the model
    model = OSEDiff_test(args)

    # get all input images (glob, folder, or single file)
    image_names, root = list_inputs(args.input_path)

    ddpm_model, _ = i_DDPM()  # 初始化在 CPU
    torch.cuda.empty_cache()

    pretrained_state_dict = torch.load(
        './weights/256x256_diffusion_uncond.pt',
        map_location='cpu'
    )
    filtered_state_dict = {
        k: v for k, v in pretrained_state_dict.items()
        if 'output_blocks' not in k and ('out.0' not in k) and ('out.2' not in k)
    }
    ddpm_model.load_state_dict(filtered_state_dict, strict=True)

    # ddpm_model = ddpm_model.cuda().half()  # 转为 fp16 后再传入 CUDA
    ddpm_model = ddpm_model.to(device)
    def convert_norm_module_to_float(module):
        if isinstance(module, (torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.BatchNorm2d)):
            module.float()
        for child in module.children():
            convert_norm_module_to_float(child)

    convert_norm_module_to_float(ddpm_model)
    freeze_model_parameters(ddpm_model)

    context_processor = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(start_dim=1)
    ).to(device)

    # weight type
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16

    if args.save_prompts:
        txt_path = os.path.join(args.output_path, 'txt')
        os.makedirs(txt_path, exist_ok=True)
    
    # make the output dir
    os.makedirs(args.output_path, exist_ok=True)

    # same prompt embedding for every image, so load it once
    prompt_embeds = load_neg_prompt_embeds(load_path="./prompt_embeds.pt")
    prompt_embeds = prompt_embeds.expand(1, -1, -1)

    print(f'There are {len(image_names)} images (root: {root}).')
    for i in tqdm(range(len(image_names)), desc="Processing images", unit="image"):
        image_name = image_names[i]
        # make sure that the input image is a multiple of 8
        input_image = Image.open(image_name).convert('RGB')
        ori_width, ori_height = input_image.size
        rscale = args.upscale
        if ori_width < args.process_size//rscale or ori_height < args.process_size//rscale:
            scale = (args.process_size//rscale)/min(ori_width, ori_height)
            input_image = input_image.resize((int(scale*ori_width), int(scale*ori_height)))
        input_image = input_image.resize((input_image.size[0]*rscale, input_image.size[1]*rscale))

        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)

        # keep the input's structure below root: <group>/turb/0.png -> output/<group>/turb/0.png
        rel = os.path.relpath(image_name, root)
        # print(rel)
        out_file = os.path.join(args.output_path, os.path.splitext(rel)[0] + ".png")  # 确保输出文件为 PNG 格式
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        # 获取描述
        lq = get_validation_prompt(input_image)

        # 模型推理
        with torch.no_grad():
            lqq = F.normalize(lq, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            timesteps = torch.tensor([0], device=lqq.device)  # 确保与输入在同一设备
            degra_context = ddpm_model(lqq, timesteps)
            degra_context = degra_context.float()    #[8, 1024, 8, 8]
            degra_context = context_processor(degra_context)
            output_image = model(lqq, degra_context, prompt_embeds) 

                # 保存图片
            output_image = denormalize(output_image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            output_image = torch.clamp(output_image, 0, 1)
            output_pil = transforms.ToPILImage()(output_image[0].cpu())

            # always return to the original size (also undoes the multiple-of-8 resize)
            target = (int(args.upscale * ori_width), int(args.upscale * ori_height))
            if output_pil.size != target:
                output_pil = output_pil.resize(target, Image.LANCZOS)

        # 保存为 PNG 格式
        output_pil.save(out_file, format='PNG')