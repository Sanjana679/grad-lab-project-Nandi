import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from net.model_3D import DEVICE, R2RLocal
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test_path", required=True, help="input folder")
    p.add_argument("--output_path", required=True, help="output folder")
    p.add_argument("--ckpt_name", default="./ckpt/train_ckpt_3D_f/")
    p.add_argument("--prompt_dir", default="./ckpt/save_prompts_3D/")
    opt = p.parse_args()

    # DEVICE comes from net/model_3D.py (set DEVICE=cpu in the environment to override)
    device = DEVICE
    print("device:", device)

    # Builds the model and loads both last.ckpt and the prompt bank
    net = R2RLocal(ckpt_path=opt.ckpt_name, prompts_path=opt.prompt_dir,
                   prompts_name="last").to(device).eval()

    os.makedirs(opt.output_path, exist_ok=True)
    names = sorted(f for f in os.listdir(opt.test_path)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))

    with torch.no_grad():
        for name in names:
            img = Image.open(os.path.join(opt.test_path, name)).convert("RGB")
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
            Image.fromarray(arr).save(os.path.join(opt.output_path, os.path.splitext(name)[0] + ".png"))
            print("saved", name)


if __name__ == "__main__":
    main()