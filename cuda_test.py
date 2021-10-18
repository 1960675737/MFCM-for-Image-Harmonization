import os
import os.path as osp
import torch
import shutil
comp_path = './images/composite_images'
comp_imgs = os.listdir(comp_path)
for comp_img in comp_imgs:
    if comp_img.startswith('a'):
        data_path = "/data/xiechengjuan/Datasets/iHarmony4/HAdobe5k/"
    elif comp_img.startswith('f'):
        data_path = "/data/xiechengjuan/Datasets/iHarmony4/HFlickr/"
    elif comp_img.startswith('d'):
        data_path = "/data/xiechengjuan/Datasets/iHarmony4/Hday2night/"
    mask_path = osp.join(data_path, 'masks', '_'.join(comp_img.split('_')[:-1]) + '.png')
    real_image_path = osp.join(data_path, 'real_images', '_'.join(comp_img.split('_')[:-2]) + '.jpg')
    shutil.copy(mask_path, './images/masks')
    shutil.copy(real_image_path, './images/real_images')