import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from net.model_3D import R2RLocal
from PIL import Image
from tqdm import tqdm

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", required=True, help="glob, directory, or single image (quote globs!)")
    p.add_argument("--output_path", required=True, help="output folder")
    p.add_argument("--ckpt_name", default="./ckpt/train_ckpt_3D_f/")
    p.add_argument("--prompt_dir", default="./ckpt/save_prompts_3D/")
    opt = p.parse_args()

    # Same env var as net/model_3D.py and feature_bank_3D.py, so all three agree.
    # The pixi task sets DEVICE=cpu (MPS adaptive pooling needs divisible sizes).
    device = torch.device(os.environ.get("DEVICE", "mps" if torch.backends.mps.is_available() else "cpu"))
    print("device:", device)

    # Builds the model and loads both last.ckpt and the prompt bank
    net = R2RLocal(ckpt_path=opt.ckpt_name, prompts_path=opt.prompt_dir,
                   prompts_name="last").to(device).eval()

    paths, root = list_inputs(opt.input_path)
    print(f"{len(paths)} images (root: {root})")

    with torch.no_grad():
        for path in tqdm(paths):
            img = Image.open(path).convert("RGB")
            x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
            x = x.permute(2, 0, 1).unsqueeze(0).to(device)          # [1,3,H,W] in [0,1]

            # pad to a multiple of 8 (no-op for 512x512 ATSyn; needed for OTIS)
            h, w = x.shape[2:]
            ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
            if ph or pw:
                x = F.pad(x, (0, pw, 0, ph), mode="reflect")

            out = net(x, interact_label=None)
            if isinstance(out, list):                               # R2R returns a list; last = final output
                out = out[-1]

            out = out[:, :, :h, :w].clamp(0, 1)                     # crop padding back off
            arr = (out[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)

            # keep the input's structure below root: <group>/turb/0.png -> output/<group>/turb/0.png
            rel = os.path.relpath(path, root)
            out_file = os.path.join(opt.output_path, os.path.splitext(rel)[0] + ".png")
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            Image.fromarray(arr).save(out_file)


if __name__ == "__main__":
    main()