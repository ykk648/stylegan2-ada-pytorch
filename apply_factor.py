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

    print(modulate.keys())

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

    parser.add_argument("--ckpt", default='', type=str, help="stylegan2 checkpoints")

    parser.add_argument(
        "-n", "--n_sample", type=int, default=5, help="number of samples created"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to run the model"
    )

    parser.add_argument(
        "--factor",
        default='/stylegan2_pack/stylegan2-pytorch-master-rosinality/stylegan2-pytorch-master/factor.pt',
        type=str,
        help="name of the closed form factorization result factor file",
    )

    args = parser.parse_args()

    weight_p = 'output/network-snapshot-000800.pt'
    # indexes = 'all'
    indexes = [6, 13]

    print(indexes)

    eigvec = factor_weight(weight_p, indexes).to(args.device)

    print('Loading networks from "%s"...' % args.ckpt)
    device = torch.device('cuda')
    with dnnlib.util.open_url(args.ckpt) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    z = torch.from_numpy(np.random.RandomState(100).randn(args.n_sample, G.z_dim)).to(device)
    latent = G.mapping(z, None, truncation_psi=1)

    for j in range(10):
        direction = eigvec[:, j].unsqueeze(0)

        img_list = []

        for i in range(21):
            # print(i)
            img_list.append(G.synthesis(latent + (-3 + i * 0.3) * direction, noise_mode='const'))

        utils.save_image(
            torch.cat(img_list, 0),
            f"/output/{indexes[0]}-{indexes[1]}/{indexes}_index-{j}_degree-.png",
            normalize=True,
            range=(-1, 1),
            nrow=args.n_sample,
        )
