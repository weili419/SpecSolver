from CAVE_Dataset import cave_dataset
import torch.utils.data as tud
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import time
import datetime
import argparse
from torch.autograd import Variable
from Utils import *
import logging
from SSIM import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from Model.Transolver import Modelcave
from thop import profile

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

model_name = 'SpectralSolver'

parser = argparse.ArgumentParser(description="PyTorch Code for HSI Fusion")
parser.add_argument('--data_path', default='/root/data1/SSF/Dataset/Cave/Test/', type=str,
                    help='path of the testing data')
parser.add_argument("--sizeI", default=512, type=int, help='the size of trainset')
parser.add_argument("--testset_num", default=12, type=int, help='total number of testset')
parser.add_argument("--batch_size", default=1, type=int, help='Batch size')
parser.add_argument("--scale", default=4, type=int, help='Scaling factor')
parser.add_argument("--sample_q", default=96, type=int, help='Scaling factor')
parser.add_argument("--sf", default=2, type=int, help='Scaling factor')
parser.add_argument("--val", default=0, type=int, help='Scaling factor')
parser.add_argument("--seed", default=1, type=int, help='Random seed')
parser.add_argument("--kernel_type", default='gaussian_blur', type=str, help='Kernel type')
opt = parser.parse_args()
print(opt)

key = 'Test.txt'
file_path = opt.data_path + key
file_list = loadpath(file_path, shuffle=False)
HR_HSI, HR_MSI = prepare_data(opt.data_path, file_list, 12)

dataset = cave_dataset(opt, HR_HSI, HR_MSI, istrain=False)
loader_test = tud.DataLoader(dataset, batch_size=opt.batch_size)

import sys
sys.path.append("/root/data1/CAVE")

if model_name == 'AFNO':
    model = torch.load("/mnt/16t/zhujunwei/code/AFNO/Checkpoint/f8/AFNO_CAVE_8_1/model_1761.pth")
elif model_name == "DHIF":
    model = torch.load("/mnt/16t/zhujunwei/code/AFNO/Checkpoint/f8/DHIF_CAVE_8/best.pth")
elif model_name == "PSRT":
    model = torch.load("/mnt/16t/zhujunwei/code/AFNO/Checkpoint/f8/PSRT_CAVE_8/best.pth")
elif model_name == "MIMO":
    model = torch.load("/mnt/16t/zhujunwei/code/AFNO/Checkpoint/f8/MIMO_CAVE_8/best.pth")
elif model_name == "DSP":
    model = torch.load("/mnt/16t/zhujunwei/code/AFNO/Checkpoint/f8/DSP_CAVE_8/best1.pth")
elif model_name == "KNLNET":
    model = torch.load("/mnt/16t/zhujunwei/code/AFNO/Checkpoint/f4/KNLNET_CAVE_8/model_0081.pth")
    model = torch.load('/mnt/16t/zhujunwei/code/AFNO/Checkpoint/f4/KNLNET_CAVE_4/model_0191.pth')
# elif model_name == "Bicubic":
#     model = torch.load('/root/data1/CAVE/experiment/Debug/4/checkpoint/BestModel.pth')
elif model_name == "SpectralSolver" or model_name == "Bicubic":
    if opt.sf == 4:
        bestmodel = torch.load('/root/data1/CAVE/experiment/SpectralSolver2/4/checkpoint/BestModel.pth')
    elif opt.sf == 8:
        bestmodel = torch.load('/root/data1/CAVE/experiment/SpectralSolver2/8/checkpoint/BestModel.pth')
    elif opt.sf == 2:
        bestmodel = torch.load('/root/data1/CAVE/experiment/SpectralSolver4/2/checkpoint/BestModel.pth')
    model = Modelcave(space_dim=2,
                    n_layers=1,
                    n_hidden=64,
                    dropout=0.0,
                    n_head=8,
                    Time_Input=False,
                    mlp_ratio=1,
                    fun_dim=128,
                    out_dim=31,
                    slice_num=64,
                    ref=8,
                    unified_pos=1,
                    H=512, W=512).cuda()
    model.load_state_dict(bestmodel)
    print(model)
num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
print(f'[INFO] #parameters: {num_params / 1e6:.2f} M')

model = model.eval()
model = model.cuda()

psnr_total = []
sam_total = []
ergas_total = []
ssim_total = []
k = 0
import time
# 统计推理时间
inference_times = []

for j, (LR, RGB, HR, COORD) in enumerate(loader_test):
    with torch.no_grad():
        LR, RGB, HR, COORD = Variable(LR), Variable(RGB), Variable(HR), Variable(COORD)
        LR, RGB, HR, COORD = LR.cuda(), RGB.cuda(), HR.cuda(), COORD.cuda()
        # 开始记录推理时间
        start_time = time.time()

        if model_name == 'AFNO':
            out = model(LR, RGB, COORD, opt.sf)
        elif model_name == 'DSP':
            out = model(LR, RGB, opt.sf)
        elif model_name == 'MIMO':
            outputs3, outputs2, out = model(RGB, LR)
        elif model_name == 'PSRT':
            UP = F.interpolate(LR, scale_factor=opt.sf, mode='bilinear')
            out = model(RGB, UP)
        elif model_name == 'DHIF':
            out = model(RGB.cuda(),LR.cuda()) 
        elif model_name == 'KNLNET':
            out = model(LR,RGB)
        elif model_name == 'Bicubic':
            out = F.interpolate(LR,scale_factor=opt.sf,mode='bicubic')
        elif model_name == 'SpectralSolver':
            up_LR = F.interpolate(LR, scale_factor=opt.sf, mode='bicubic', align_corners=False)
            out, ffted_mag, fxHR_spa_mag = model(COORD.cuda(), up_LR.cuda(), RGB.cuda(), HR.cuda())
        result = out
        # 记录推理结束时间
        end_time = time.time()
        inference_time = end_time - start_time
        inference_times.append(inference_time)

        result = result.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
        HR = HR.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1,2,0)

        # sio.savemat('/root/data1/CAVE/experiment/SpectralSolver2/4/mat/' + file_list[j] + '.mat', {'result': result, 'GT': HR}) # cave

    psnr = cal_psnr(result, HR)
    psnr_total.append(psnr)
    sam = compute_sam(result, HR)
    sam_total.append(sam)
    ergas = compute_ergas(result, HR, opt.sf)
    ergas_total.append(ergas)
    ssim_v = compute_ssim(result, HR)
    ssim_total.append(ssim_v)
    k = k + 1
    avg_psnr = np.mean(psnr_total)
    avg_sam = np.mean(sam_total)
    avg_ergas = np.mean(ergas_total)
    avg_ssim = np.mean(ssim_total)

    # 计算标准差
    std_psnr = np.std(psnr_total)
    std_sam = np.std(sam_total)
    std_ergas = np.std(ergas_total)
    std_ssim = np.std(ssim_total)

# 计算并输出平均推理时间
average_inference_time = sum(inference_times) / len(inference_times)
print("----------------")
print(k)
print("Avg PSNR = %.2f, Std PSNR = %.2f" % (avg_psnr, std_psnr))
print("Avg SAM = %.2f, Std SAM = %.2f" % (avg_sam, std_sam))
print("Avg ERGAS = %.2f, Std ERGAS = %.2f" % (avg_ergas, std_ergas))
print("Avg SSIM = %.3f, Std SSIM = %.4f" % (avg_ssim, std_ssim))
