import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from options import (
    train_options,  # parser; builds the MoCE_IR architecture settings
)
from PIL import Image
from test import (
    PLTestModel,  # wrapper; safe to import (main is behind __name__ guard)
)
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
    # Their parser is strict, so take our own --input_path off the command line first,
    # then let train_options() parse the rest (--model, --checkpoint_id, --ckpt_dir, --output_path, ...)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--input_path", required=True, help="glob, directory, or single image (quote globs!)")
    args, rest = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    opt = train_options()

    # DEVICE=cpu overrides
    dev_name = os.environ.get("DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(dev_name)
    print("device:", device)

    ckpt = os.path.join(opt.ckpt_dir, opt.checkpoint_id, "last.ckpt")
    net = PLTestModel.load_from_checkpoint(ckpt, opt=opt, map_location="cpu").to(device).eval()

    paths, root = list_inputs(args.input_path)
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

            out = net(x)
            if isinstance(out, (list, tuple)) and len(out) == 2:   # model may return [restored, aux]
                out, _ = out

            out = out[:, :, :h, :w].clamp(0, 1)                     # crop padding back off
            arr = (out[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)

            # keep the input's structure below root: <group>/turb/0.png -> output/<group>/turb/0.png
            rel = os.path.relpath(path, root)
            out_file = os.path.join(opt.output_path, os.path.splitext(rel)[0] + ".png")
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            Image.fromarray(arr).save(out_file)


if __name__ == "__main__":
    main()