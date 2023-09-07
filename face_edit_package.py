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
import random


def linear_interpolate(code1, code2, alpha):
    return code1 * alpha + code2 * (1 - alpha)


def save_img(images_, out_p):
    images_ = (images_.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    images_ = images_[0].cpu().numpy()
    interp_latent_image = Image.fromarray(images_)
    interp_latent_image.save(out_p)


def get_w_from_npy(npy_p, device_):
    return torch.tensor(np.load(npy_p)['w'][0], device=device_)[np.newaxis]


def move_latent(latent_vector, direction_list):
    # new_latent_vector = latent_vector
    for index, direction in enumerate(direction_list):
        if index in [0, 1]:
            latent_vector[0][:8] = (latent_vector[0] + random.randint(-10, 10) * direction)[:8]
        else:
            latent_vector[0][:8] = (latent_vector[0] + random.randint(-5, 5) * direction)[:8]
    return latent_vector


class FaceEdit:
    def __init__(self, device_, model_p_):
        print('Loading networks from "%s"...' % model_p_)
        with dnnlib.util.open_url(model_p_) as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].to(device_)  # type: ignore

    def generate_image_from_projected_latents(self, latent_vector):
        return self.G.synthesis(latent_vector)


if __name__ == '__main__':
    device = torch.device('cuda')
    # model_p = './models/ffhq.pkl'
    # model_p = './models/metfaces.pkl'
    model_p = './models/star.pkl'

    face_edit = FaceEdit(device, model_p)

    # w1 = get_w_from_npy('projected_w1.npz', device)
    # w2 = get_w_from_npy('projected_w2.npz', device)
    # # interpolated_latent_code = linear_interpolate(w1, w2, 0.5)
    # images = face_edit.generate_image_from_projected_latents(w1)
    # save_img(images, './output/interp_img_met3.jpg')

    direction_file_list = []
    for direction_file in ['angle_horizontal.npy', 'angle_pitch.npy', 'smile.npy',
                           'beauty.npy', 'emotion_angry.npy',
                           'emotion_disgust.npy', 'emotion_easy.npy', 'emotion_fear.npy', 'emotion_happy.npy',
                           'emotion_sad.npy', 'emotion_surprise.npy', 'eyes_open.npy']:
        direction_file_list.append(np.load('./latent_directions/' + direction_file))

    for i in range(2000):
        interpolated_latent_code_mod = get_w_from_npy('projected_w.npz', device)
        # interpolated_latent_code_mod = interpolated_latent_code
        new_latent_vector = move_latent(interpolated_latent_code_mod.cpu(), direction_file_list)
        images = face_edit.generate_image_from_projected_latents(new_latent_vector.cuda())
        save_img(images, './output/interp_img_move_{}.jpg'.format(str(i)))
