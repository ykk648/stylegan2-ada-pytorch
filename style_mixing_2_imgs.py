# collapse-hide
import os
import re
from typing import List

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
import legacy
from PIL import Image


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def linear_interpolate(code1, code2, alpha):
    return code1 * alpha + code2 * (1 - alpha)


def make_latent_interp_animation_real_faces(code1, code2, img1, img2, num_interps):
    step_size = 1.0 / num_interps

    all_imgs = []

    amounts = np.arange(0, 1, step_size)

    for alpha in tqdm(amounts):
        interpolated_latent_code = linear_interpolate(code1, code2, alpha)
        images = generate_image_from_projected_latents(interpolated_latent_code)
        images = (images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        images = images[0].cpu().numpy()
        interp_latent_image = Image.fromarray(images).resize((400, 400))
        frame = get_concat_h(img2, interp_latent_image)
        frame = get_concat_h(frame, img1)
        all_imgs.append(frame)

    save_name = './projected_latent_space_traversal.gif'
    all_imgs[0].save(save_name, save_all=True, append_images=all_imgs[1:], duration=1000 / 20, loop=0)


if __name__ == '__main__':
    device = torch.device('cuda')


    print('Loading networks from "%s"...' % './ffhq.pkl')
    with dnnlib.util.open_url('./ffhq.pkl') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore


    def generate_image_from_projected_latents(latent_vector):
        images = G.synthesis(latent_vector)
        return images


    w1 = np.load('projected_w1.npz')['w'][0]
    w2 = np.load('projected_w1.npz')['w'][0]
    w1 = torch.tensor(w1, device=device)
    w2 = torch.tensor(w2, device=device)

    recreated_img1 = G.synthesis(w1[np.newaxis])
    recreated_img1 = (recreated_img1.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    recreated_img1 = recreated_img1[0].cpu().numpy()
    recreated_img1 = Image.fromarray(recreated_img1).resize((400, 400))

    recreated_img2 = G.synthesis(w2[np.newaxis])
    recreated_img2 = (recreated_img2.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    recreated_img2 = recreated_img2[0].cpu().numpy()
    recreated_img2 = Image.fromarray(recreated_img2).resize((400, 400))

    make_latent_interp_animation_real_faces(w1[np.newaxis], w2[np.newaxis], recreated_img1, recreated_img2, num_interps=200)

    # # save txt file
    # def text_save(file, data):
    #     for i in range(len(data[0])):
    #         s = str(data[0][i]) + '\n'
    #         file.write(s)
    #
    #
    # w1 = np.load('projected_w1.npz')['w'][0]
    # w2 = np.load('projected_w2.npz')['w'][0]
    # w1 = torch.tensor(w1, device=device)
    # w2 = torch.tensor(w2, device=device)
    #
    # interpolated_latent_code = linear_interpolate(w1[np.newaxis], w2[np.newaxis], 0.5).cpu()
    #
    # out = np.array(interpolated_latent_code[0][0].reshape(1, 512))
    # print(out)
    # print(out.shape)
    #
    # txt_file = './out.txt'
    # with open(txt_file, 'w') as f:
    #     text_save(f, out)
