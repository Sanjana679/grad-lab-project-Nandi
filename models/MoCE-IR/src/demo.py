import os

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


def main():
    # Reuses flags: --data_file_dir = input folder, --output_path = output folder
    opt = train_options()

    # DEVICE=cpu overrides (parser is strict, so no new flags)
    dev_name = os.environ.get("DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(dev_name)
    print("device:", device)

    ckpt = os.path.join(opt.ckpt_dir, opt.checkpoint_id, "last.ckpt")
    net = PLTestModel.load_from_checkpoint(ckpt, opt=opt, map_location="cpu").to(device).eval()

    os.makedirs(opt.output_path, exist_ok=True)
    names = sorted(f for f in os.listdir(opt.data_file_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))

    with torch.no_grad():
        for name in names:
            img = Image.open(os.path.join(opt.data_file_dir, name)).convert("RGB")
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
            Image.fromarray(arr).save(os.path.join(opt.output_path, os.path.splitext(name)[0] + ".png"))
            print("saved", name)


if __name__ == "__main__":
    main()