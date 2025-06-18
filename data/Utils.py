import numpy as np
import scipy.io as sio
import os
import glob
import torch
import torch.nn as nn
import skimage.measure as measure
import torch.nn.functional as F
import cv2
import Pypher
import random
import re
import math
import h5py
from skimage.metrics import structural_similarity as compare_ssim


def _as_floats(im1, im2):
    float_type = np.result_type(im1.dtype, im2.dtype, np.float32)
    im1 = np.asarray(im1, dtype=float_type)
    im2 = np.asarray(im2, dtype=float_type)
    return im1, im2


def compare_mse(im1, im2):
    im1, im2 = _as_floats(im1, im2)
    return np.mean(np.square(im1 - im2), dtype=np.float64)

def cal_psnr(im1, im2): # [512,512,31]
    num_spectral = im1.shape[-1]
    im1 = np.reshape(im1, (-1, num_spectral))
    im2 = np.reshape(im2, (-1, num_spectral))
    diff = im1 - im2
    mse = np.mean(np.square(diff), axis=0)
    return np.mean(10 * np.log10(1 / (mse+1e-8)))

def compute_sam(im1, im2):
    num_spectral = im1.shape[-1]
    im1 = np.reshape(im1, (-1, num_spectral))
    im2 = np.reshape(im2, (-1, num_spectral))
    mole = np.sum(np.multiply(im1, im2), axis=1)
    im1_norm = np.sqrt(np.sum(np.square(im1), axis=1))
    im2_norm = np.sqrt(np.sum(np.square(im2), axis=1))
    deno = np.multiply(im1_norm, im2_norm)

    sam = np.rad2deg(np.arccos((mole) / (deno + 1e-7)))
    sam = np.mean(sam)
    return sam

def compute_ergas(out, gt, c):
    num_spectral = out.shape[-1]
    out = np.reshape(out, (-1, num_spectral))
    gt = np.reshape(gt, (-1, num_spectral))
    diff = gt - out
    mse = np.mean(np.square(diff), axis=0)
    gt_mean = np.mean(gt, axis=0)
    mse = np.reshape(mse, (num_spectral, 1))
    gt_mean = np.reshape(gt_mean, (num_spectral, 1))
    ergas = 100 / c * np.sqrt(np.mean(mse / (gt_mean ** 2 + 1e-6)))
    return ergas

def compute_ssim(im1, im2):
    n = im1.shape[2]
    ms_ssim = 0.0
    for i in range(n):
        single_ssim = compare_ssim(im1[:, :, i], im2[:, :, i], data_range=1.0)
        ms_ssim += single_ssim
    return ms_ssim / n

# def compare_psnr(im_true, im_test, data_range=None):
#     im_true, im_test = _as_floats(im_true, im_test)

#     err = compare_mse(im_true, im_test)
#     return 10 * np.log10((data_range ** 2) / err)


# def psnr(img1, img2):
#     mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
#     if mse < 1.0e-10:
#         return 100
#     PIXEL_MAX = 1
#     return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


# def PSNR_GPU(im_true, im_fake):
#     im_true *= 255
#     im_fake *= 255
#     im_true = im_true.round()
#     im_fake = im_fake.round()
#     data_range = 255
#     esp = 1e-12
#     C = im_true.size()[0]
#     H = im_true.size()[1]
#     W = im_true.size()[2]
#     Itrue = im_true.clone()
#     Ifake = im_fake.clone()
#     mse = nn.MSELoss(reduce=False)
#     err = mse(Itrue, Ifake).sum() / (C * H * W)
#     psnr = 10. * np.log((data_range ** 2) / (err.data + esp)) / np.log(10.)
#     return psnr


# def PSNR_Nssr(im_true, im_fake):
#     mse = ((im_true - im_fake) ** 2).mean()
#     psnr = 10. * np.log10(1 / mse)
#     return psnr


# def c_psnr(im1, im2):
#     '''
#     Compute PSNR
#     :param im1: input image 1 ndarray ranging [0,1]
#     :param im2: input image 2 ndarray ranging [0,1]
#     :return: psnr=-10*log(mse(im1,im2))
#     '''
#     # mse = np.power(im1 - im2, 2).mean()
#     # psnr = 20 * np.log10(255.0) - 10 * np.log10(mse)
#     im1 = np.maximum(np.minimum(im1, 1.0), 0.0)
#     im2 = np.maximum(np.minimum(im2, 1.0), 0.0)
#     return measure.compare_psnr(im1, im2)


# def c_ssim(im1, im2):
#     '''
#     Compute PSNR
#     :param im1: input image 1 ndarray ranging [0,1]
#     :param im2: input image 2 ndarray ranging [0,1]
#     :return: psnr=-10*log(mse(im1,im2))
#     '''
#     # mse = np.power(im1 - im2, 2).mean()
#     # psnr = 20 * np.log10(255.0) - 10 * np.log10(mse)

#     im1 = np.maximum(np.minimum(im1, 1.0), 0.0)
#     im2 = np.maximum(np.minimum(im2, 1.0), 0.0)
#     return measure.compare_ssim(im1, im2, win_size=11, data_range=1, gaussian_weights=True)


# def batch_PSNR(im_true, im_fake, data_range):
#     N = im_true.size()[0]
#     C = im_true.size()[1]
#     H = im_true.size()[2]
#     W = im_true.size()[3]
#     Itrue = im_true.clone().resize_(N, C * H * W)
#     Ifake = im_fake.clone().resize_(N, C * H * W)
#     mse = nn.MSELoss(reduce=False)
#     err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)
#     psnr = 10. * torch.log((data_range ** 2) / err) / np.log(10.)
#     return torch.mean(psnr)


# def PSNR_GPU(im_true, im_fake):
#     im_true *= 255
#     im_fake *= 255
#     im_true = im_true.round()
#     im_fake = im_fake.round()
#     data_range = 255
#     esp = 1e-12
#     C = im_true.size()[0]
#     H = im_true.size()[1]
#     W = im_true.size()[2]
#     Itrue = im_true.clone()
#     Ifake = im_fake.clone()
#     mse = nn.MSELoss(reduce=False)
#     err = mse(Itrue, Ifake).sum() / (C * H * W)
#     psnr = 10. * np.log((data_range ** 2) / (err.data + esp)) / np.log(10.)
#     return psnr


# def batch_SAM_GPU(im_true, im_fake):
#     N = im_true.size()[0]
#     C = im_true.size()[1]
#     H = im_true.size()[2]
#     W = im_true.size()[3]
#     Itrue = im_true.clone().resize_(N, C, H * W)
#     Ifake = im_fake.clone().resize_(N, C, H * W)
#     nom = torch.mul(Itrue, Ifake).sum(dim=1).resize_(N, H * W)
#     denom1 = torch.pow(Itrue, 2).sum(dim=1).sqrt_().resize_(N, H * W)
#     denom2 = torch.pow(Ifake, 2).sum(dim=1).sqrt_().resize_(N, H * W)
#     sam = torch.div(nom, torch.mul(denom1, denom2)).acos_().resize_(N, H * W)
#     sam = sam / np.pi * 180
#     sam = torch.sum(sam) / (N * H * W)
#     return sam


# def SAM_GPU(im_true, im_fake):
#     C = im_true.size()[0]
#     H = im_true.size()[1]
#     W = im_true.size()[2]
#     esp = 1e-12
#     Itrue = im_true.clone()  # .resize_(C, H*W)
#     Ifake = im_fake.clone()  # .resize_(C, H*W)
#     nom = torch.mul(Itrue, Ifake).sum(dim=0)  # .resize_(H*W)
#     denominator = Itrue.norm(p=2, dim=0, keepdim=True).clamp(min=esp) * \
#                   Ifake.norm(p=2, dim=0, keepdim=True).clamp(min=esp)
#     denominator = denominator.squeeze()
#     sam = torch.div(nom, denominator).acos()
#     sam[sam != sam] = 0
#     sam_sum = torch.sum(sam) / (H * W) / np.pi * 180
#     return sam_sum


# def batch_SAM_CPU(im_true, im_fake):
#     I_true = im_true.data.cpu().numpy()
#     I_fake = im_fake.data.cpu().numpy()
#     N = I_true.shape[0]
#     C = I_true.shape[1]
#     H = I_true.shape[2]
#     W = I_true.shape[3]
#     batch_sam = 0
#     for i in range(N):
#         true = I_true[i, :, :, :].reshape(C, H * W)
#         fake = I_fake[i, :, :, :].reshape(C, H * W)
#         nom = np.sum(np.multiply(true, fake), 0).reshape(H * W, 1)
#         denom1 = np.sqrt(np.sum(np.square(true), 0)).reshape(H * W, 1)
#         denom2 = np.sqrt(np.sum(np.square(fake), 0)).reshape(H * W, 1)
#         sam = np.arccos(np.divide(nom, np.multiply(denom1, denom2))).reshape(H * W, 1)
#         sam = sam / np.pi * 180
#         # ignore pixels that have zero norm
#         idx = (np.isfinite(sam))
#         batch_sam += np.sum(sam[idx]) / np.sum(idx)
#         if np.sum(~idx) != 0:
#             print("waring: some values were ignored when computing SAM")
#     return batch_sam / N


# def SAM_CPU(im_true, im_fake):
#     I_true = im_true.data.cpu().numpy()
#     I_fake = im_fake.data.cpu().numpy()
#     N = I_true.shape[0]
#     C = I_true.shape[1]
#     H = I_true.shape[2]
#     W = I_true.shape[3]
#     batch_sam = 0
#     for i in range(N):
#         true = I_true[i, :, :, :].reshape(C, H * W)
#         fake = I_fake[i, :, :, :].reshape(C, H * W)
#         nom = np.sum(np.multiply(true, fake), 0).reshape(H * W, 1)
#         denom1 = np.sqrt(np.sum(np.square(true), 0)).reshape(H * W, 1)
#         denom2 = np.sqrt(np.sum(np.square(fake), 0)).reshape(H * W, 1)
#         sam = np.arccos(np.divide(nom, np.multiply(denom1, denom2))).reshape(H * W, 1)
#         sam = sam / np.pi * 180
#         # ignore pixels that have zero norm
#         idx = (np.isfinite(sam))
#         batch_sam += np.sum(sam[idx]) / np.sum(idx)
#         if np.sum(~idx) != 0:
#             print("waring: some values were ignored when computing SAM")
#     return batch_sam / N


# Convert an image into patches
def Im2Patch(img, img2, win, stride=1, istrain=False):  # Based on code written by Shuhang Gu (cssgu@comp.polyu.edu.hk)
    k = 0
    endw = img.shape[0]
    endh = img.shape[1]
    if endw < win or endh < win:
        return None, None
    patch = img[0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    TotalPatNum = patch.shape[0] * patch.shape[1]
    # TotalPatNum = (img.shape[0]-win+1)*(img.shape[1]-win+1)
    Y = np.zeros([win * win, TotalPatNum])
    Y2 = np.zeros([win * win, TotalPatNum])
    for i in range(win):
        for j in range(win):
            patch = img[i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[k, :] = np.array(patch[:]).reshape(TotalPatNum)
            patch2 = img2[i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y2[k, :] = np.array(patch2[:]).reshape(TotalPatNum)
            k = k + 1
    if istrain:
        return Y.reshape([win, win, TotalPatNum]), Y2.reshape([win, win, TotalPatNum])
    else:
        return Y.transpose().reshape([TotalPatNum, win, win, 1]), Y2.transpose().reshape([TotalPatNum, win, win, 1])


def para_setting(kernel_type, sf, sz, sigma):
    if kernel_type == 'uniform_blur':
        psf = np.ones([sf, sf]) / (sf * sf)
    elif kernel_type == 'gaussian_blur':
        psf = np.multiply(cv2.getGaussianKernel(sf, sigma), (cv2.getGaussianKernel(sf, sigma)).T)

    fft_B = Pypher.psf2otf(psf, sz)
    fft_BT = np.conj(fft_B)
    return fft_B, fft_BT


def H_z(z, factor, fft_B):
    #     z  [31 , 96 , 96]
    #     ch, h, w = z.shape
    f = torch.rfft(z, 2, onesided=False)
    # -------------------complex myltiply-----------------#
    if len(z.shape) == 3:
        ch, h, w = z.shape
        fft_B = fft_B.unsqueeze(0).repeat(ch, 1, 1, 1)
        M = torch.cat(((f[:, :, :, 0] * fft_B[:, :, :, 0] - f[:, :, :, 1] * fft_B[:, :, :, 1]).unsqueeze(3),
                       (f[:, :, :, 0] * fft_B[:, :, :, 1] + f[:, :, :, 1] * fft_B[:, :, :, 0]).unsqueeze(3)), 3)
        Hz = torch.irfft(M, 2, onesided=False)
        x = Hz[:, int(factor // 2)::factor, int(factor // 2)::factor]
    elif len(z.shape) == 4:
        bs, ch, h, w = z.shape
        fft_B = fft_B.unsqueeze(0).unsqueeze(0).repeat(bs, ch, 1, 1, 1)
        M = torch.cat(((f[:, :, :, :, 0] * fft_B[:, :, :, :, 0] - f[:, :, :, :, 1] * fft_B[:, :, :, :, 1]).unsqueeze(4),
                       (f[:, :, :, :, 0] * fft_B[:, :, :, :, 1] + f[:, :, :, :, 1] * fft_B[:, :, :, :, 0]).unsqueeze(
                           4)), 4)
        Hz = torch.irfft(M, 2, onesided=False)
        x = Hz[:, :, int(factor // 2)::factor, int(factor // 2)::factor]
    return x


def HT_y(y, sf, fft_BT):
    if len(y.shape) == 3:
        ch, w, h = y.shape
        # z = torch.zeros([ch, w*sf ,h*sf])
        # z[:,::sf, ::sf] = y
        z = F.pad(y, [0, 0, 0, 0, 0, sf * sf - 1], "constant", value=0)
        z = F.pixel_shuffle(z, upscale_factor=sf).view(1, ch, w * sf, h * sf)

        f = torch.rfft(z, 2, onesided=False)
        fft_BT = fft_BT.unsqueeze(0).repeat(ch, 1, 1, 1)
        M = torch.cat(((f[:, :, :, 0] * fft_BT[:, :, :, 0] - f[:, :, :, 1] * fft_BT[:, :, :, 1]).unsqueeze(3),
                       (f[:, :, :, 0] * fft_BT[:, :, :, 1] + f[:, :, :, 1] * fft_BT[:, :, :, 0]).unsqueeze(3)), 3)
        Hz = torch.irfft(M, 2, onesided=False)
    elif len(y.shape) == 4:
        bs, ch, w, h = y.shape
        # z = torch.zeros([bs ,ch ,sf*w ,sf*w])
        # z[:,:,::sf,::sf] = y
        z = y.view(-1, 1, w, h)
        z = F.pad(z, [0, 0, 0, 0, 0, sf * sf - 1, 0, 0], "constant", value=0)
        z = F.pixel_shuffle(z, upscale_factor=sf).view(bs, ch, w * sf, h * sf)

        f = torch.rfft(z, 2, onesided=False)
        fft_BT = fft_BT.unsqueeze(0).unsqueeze(0).repeat(bs, ch, 1, 1, 1)
        M = torch.cat(
            ((f[:, :, :, :, 0] * fft_BT[:, :, :, :, 0] - f[:, :, :, :, 1] * fft_BT[:, :, :, :, 1]).unsqueeze(4),
             (f[:, :, :, :, 0] * fft_BT[:, :, :, :, 1] + f[:, :, :, :, 1] * fft_BT[:, :, :, :, 0]).unsqueeze(4)), 4)
        Hz = torch.irfft(M, 2, onesided=False)
    return Hz


def dataparallel(model, ngpus, gpu0=0):
    """
    根据 GPU 数量将模型放置到单个或多个 GPU 上。
    
    :param model: 要部署的 PyTorch 模型
    :param ngpus: 使用的 GPU 数量
    :param gpu0: 起始的 GPU 编号（默认从 0 开始）
    :return: 已部署到指定 GPU 的模型
    """

    # 如果不使用 GPU，则直接报错（函数仅支持 GPU 模式）
    if ngpus == 0:
        assert False, "only support gpu mode"

    # 构建要使用的 GPU 编号列表，例如 [0, 1, 2]
    gpu_list = list(range(gpu0, gpu0 + ngpus))

    # 检查本机 GPU 数量是否足够
    assert torch.cuda.device_count() >= gpu0 + ngpus, "Not enough available GPUs."

    # 多 GPU 模式
    if ngpus > 1:
        # 如果模型尚未被包裹为 DataParallel，则包装起来
        if not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model, device_ids=gpu_list).cuda()
        else:
            # 如果已经是 DataParallel 模型，直接将其转移到 GPU
            model = model.cuda()

    # 单 GPU 模式
    elif ngpus == 1:
        model = model.cuda()

    return model


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def prepare_data(path, file_list, file_num):
    HR_HSI = np.zeros((((512, 512, 31, file_num))))
    HR_MSI = np.zeros((((512, 512, 3, file_num))))
    for idx in range(file_num):
        ####  read HR-HSI
        HR_code = file_list[idx]
        path1 = os.path.join(path, 'HSI/') + HR_code + '.mat'
        data = sio.loadmat(path1)
        HR_HSI[:, :, :, idx] = data['hsi']

        ####  get HR-MSI
        path2 = os.path.join(path, 'RGB/') + HR_code + '.mat'
        data = sio.loadmat(path2)
        HR_MSI[:, :, :, idx] = data['rgb']
    return HR_HSI, HR_MSI

def prepare_data_Chikusei(path, file_list, file_num):
    HR_HSI = np.zeros((((512, 512, 128, file_num))))
    HR_MSI = np.zeros((((512, 512, 6, file_num))))
    for idx in range(file_num):
        ####  read HR-HSI
        HR_code = file_list[idx]
        path1 = os.path.join(path, 'HSI/HSI_') + HR_code + '.mat'
        data = sio.loadmat(path1)
        HR_HSI[:, :, :, idx] = data['HSI']

        ####  get HR-MSI
        path2 = os.path.join(path, 'MSI/MSI_') + HR_code + '.mat'
        data = sio.loadmat(path2)
        HR_MSI[:, :, :, idx] = data['MSI']
    return HR_HSI, HR_MSI


def prepare_data_harvard(path, file_num):
    HR_HSI = np.zeros((((1040, 1392, 31, file_num))))
    HR_MSI = np.zeros((((1040, 1392, 3, file_num))))

    for idx in range(file_num):
        ####  read HR-HSI
        path1 = os.path.join(path, '') + str(idx+1) + '.mat'
        data = sio.loadmat(path1)

        HR_HSI[:, :, :, idx] = data['HS']
        HR_MSI[:, :, :, idx] = data['HRMS']
    return HR_HSI, HR_MSI

def prepare_data_cave(path, file_num):
    HR_HSI = np.zeros((((512, 512, 31, file_num))))
    HR_MSI = np.zeros((((512, 512, 3, file_num))))
    for idx in range(file_num):
        ####  read HR-HSI
        path1 = os.path.join(path, '') + str(idx+1) + '.mat'
        data = sio.loadmat(path1)

        HR_HSI[:, :, :, idx] = data['HS']
        HR_MSI[:, :, :, idx] = data['HRMS']
    return HR_HSI, HR_MSI


def loadpath(pathlistfile, shuffle=True):
    fp = open(pathlistfile)
    pathlist = fp.read().splitlines()
    fp.close()
    if shuffle == True:
        random.shuffle(pathlist)
    return pathlist


def data_augmentation(imagein, mode):
    image = imagein
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        image = torch.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        image = torch.rot90(image, 1, [2, 3])
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = torch.rot90(image, 1, [2, 3])
        image = torch.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        image = torch.rot90(image, 2, [2, 3])
    elif mode == 5:
        # rotate 180 degree and flip
        image = torch.rot90(image, 2, [2, 3])
        image = torch.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        image = torch.rot90(image, 3, [2, 3])
    elif mode == 7:
        # rotate 270 degree and flip
        image = torch.rot90(image, 3, [2, 3])
        image = torch.flipud(image)
    imageout = image
    return imageout


def calc_ergas(img_tgt, img_fus):
    scale = 4
    img_tgt = img_tgt.squeeze(0).data.cpu().numpy()
    img_fus = img_fus.squeeze(0).data.cpu().numpy()
    img_tgt = np.squeeze(img_tgt)
    img_fus = np.squeeze(img_fus)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)

    rmse = np.mean((img_tgt - img_fus) ** 2, axis=1)
    rmse = rmse ** 0.5
    mean = np.mean(img_tgt, axis=1)

    ergas = np.mean((rmse / mean) ** 2)
    ergas = 100 / scale * ergas ** 0.5

    return ergas


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    ranged in [-1, 1]
    e.g.
        shape = [2] get (-0.5, 0.5)
        shape = [3] get (-0.67, 0, 0.67)
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)  # [H, W, 2]
    if flatten:
        ret = ret.view(-1, ret.shape[-1])  # [H*W, 2]
    return ret


def extract_high_freq(x):
    # x shape: (B, C, H, W)
    B, C, H, W = x.shape

    # 傅里叶变换
    fft_x = torch.fft.fft2(x, dim=(-2, -1))
    fft_x = torch.fft.fftshift(fft_x)

    # 带通滤波器，仅保留高频信息
    freq_thresh = 0.5  # 可以根据需要调整
    center_h, center_w = H // 2, W // 2
    mask = torch.zeros_like(x)
    for i in range(H):
        for j in range(W):
            dist = ((i - center_h) / H) ** 2 + ((j - center_w) / W) ** 2
            if dist > freq_thresh:
                mask[:, :, i, j] = 1

    high_freq = fft_x * mask
    high_freq = torch.fft.ifftshift(high_freq)
    high_freq = torch.fft.ifft2(high_freq, dim=(-2, -1))
    high_freq = high_freq.real

    return high_freq
