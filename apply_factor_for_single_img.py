import argparse

import torch
from torchvision import utils
import legacy
# from model import Generator
import dnnlib
import PIL.Image
import numpy as np


def factor_weight(ckpt_p, indexes):
    ckpt = torch.load(ckpt_p)
    modulate = {
        k: v
        for k, v in ckpt["g_ema"].items()
        if "modulation" in k and "to_rgbs" not in k and "to_rgb1" not in k and "conv1" not in k and "weight" in k
    }

    # print(modulate.keys())

    weight_mat = []
    for k, v in modulate.items():
        weight_mat.append(v)
    if indexes != 'all':
        weight_mat = weight_mat[indexes[0]:indexes[1] + 1]

    W = torch.cat(weight_mat, 0)
    eigvec = torch.svd(W).V.to("cpu")
    return eigvec


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser(description="Apply closed form factorization")

    parser.add_argument("--ckpt",
                        default='network-snapshot-000800.pkl',
                        type=str, help="stylegan2 checkpoints")

    parser.add_argument(
        "-n", "--n_sample", type=int, default=1, help="number of samples created"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to run the model"
    )

    parser.add_argument(
        "--factor",
        default='/workspace/codes/stylegan2_pack/stylegan2-pytorch-master-rosinality/stylegan2-pytorch-master/factor.pt',
        type=str,
        help="name of the closed form factorization result factor file",
    )

    args = parser.parse_args()

    weight_p = '/workspace/output/network-snapshot-000800.pt'

    print('Loading networks from "%s"...' % args.ckpt)
    device = torch.device('cuda')
    with dnnlib.util.open_url(args.ckpt) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    z = torch.from_numpy(np.random.RandomState(10011).randn(args.n_sample, G.z_dim)).to(device)
    latent = G.mapping(z, None, truncation_psi=1)

    img_list = [G.synthesis(latent, noise_mode='const')]

    for index in ['all', [0, 0], [1, 1], [2, 2],[3, 3], [4, 4], [5, 5],[0, 1], [2, 5], [6, 13]]:
        eigvec = factor_weight(weight_p, index).to(args.device)
        for j in range(10):
            direction = eigvec[:, j].unsqueeze(0)

            img_list.append(G.synthesis(latent + (-3) * direction, noise_mode='const'))
            img_list.append(G.synthesis(latent + (3) * direction, noise_mode='const'))

    for index, img in enumerate(img_list):
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'/output/index{index}.png')
