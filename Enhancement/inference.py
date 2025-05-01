import os
import argparse
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from basicsr.models import create_model
from basicsr.utils.options import parse
from utils import load_img, save_img  # your existing util functions

def enhance_image(img: np.ndarray, model: torch.nn.Module, factor: int = 4) -> np.ndarray:
    """Enhance a single H×W×C image (float32 in [0,1])."""
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()

    # pad so H,W are multiples of factor
    b, c, h, w = x.shape
    H = ((h + factor - 1) // factor) * factor
    W = ((w + factor - 1) // factor) * factor
    pad_h, pad_w = H - h, W - w
    x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

    with torch.no_grad():
        y = model(x)
        y = y[:, :, :h, :w]

    y = torch.clamp(y, 0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy()
    return y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt',       type=str, required=True,
                        help='Path to RetinexFormer .yml config')
    parser.add_argument('--weights',   type=str, required=True,
                        help='Path to .pth weights')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Folder of low-light images')
    parser.add_argument('--output_dir',type=str, required=True,
                        help='Where to save enhanced images')
    parser.add_argument('--exts',      type=str, default='png,jpg,jpeg',
                        help='Comma-separated image extensions to process')
    args = parser.parse_args()

    # load config & build model
    opt = parse(args.opt, is_train=False)
    opt['dist'] = False
    model = create_model(opt).net_g
    ckpt = torch.load(args.weights)
    # handle DataParallel keys if needed
    state = ckpt.get('params', ckpt)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        new_state = {('module.'+k if not k.startswith('module.') else k): v
                     for k, v in state.items()}
        model.load_state_dict(new_state)
    model.cuda().eval()

    # gather input paths
    exts = tuple(e.strip().lower() for e in args.exts.split(','))
    files = []
    for e in exts:
        files += glob(os.path.join(args.input_dir, f'**/*.{e}'),
                      recursive=True)
    files = sorted(files)
    assert files, f'No images found in {args.input_dir}'

    os.makedirs(args.output_dir, exist_ok=True)

    for inp in files:
        # 1) load; 2) normalize to [0,1]
        img = np.float32(load_img(inp)) / 255.0
        out = enhance_image(img, model)

        # recreate sub-folder structure if nested
        rel = os.path.relpath(inp, args.input_dir)
        out_path = os.path.join(args.output_dir, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        save_img(out_path, (out * 255).astype(np.uint8))
        print(f'[✓] {rel}')

if __name__ == '__main__':
    main()
