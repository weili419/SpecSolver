from data.Harvard_Dataset import harvard_dataset
import torch.utils.data as tud
import time
import argparse
from torch.autograd import Variable
from utils.utils import *
from utils.SSIM import *
from Model.SpecSolver import Modelharvard

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

model_name = 'SpectralSolver'

parser = argparse.ArgumentParser(description="PyTorch Code for HSI Fusion")
parser.add_argument('--data_path', default='/root/data1/dataset/Harvard/Test/', type=str,
                    help='path of the testing data')
parser.add_argument("--sizeI", default=1024, type=int, help='the size of trainset')
parser.add_argument("--testset_num", default=10, type=int, help='total number of testset')
parser.add_argument("--batch_size", default=1, type=int, help='Batch size')
parser.add_argument("--scale", default=4, type=int, help='Scaling factor')
parser.add_argument("--sample_q", default=96, type=int, help='Scaling factor')
parser.add_argument("--sf", default=8, type=int, help='Scaling factor')
parser.add_argument("--val", default=0, type=int, help='Scaling factor')
parser.add_argument("--seed", default=1, type=int, help='Random seed')
parser.add_argument("--kernel_type", default='gaussian_blur', type=str, help='Kernel type')
opt = parser.parse_args()
print(opt)

test_HR_HSI, test_HR_MSI = prepare_data_harvard(opt.data_path, 10)
test_dataset = harvard_dataset(opt, test_HR_HSI, test_HR_MSI, istrain=False)
loader_test = tud.DataLoader(test_dataset, batch_size=1, num_workers=8)

if model_name == "SpectralSolver":
    if opt.sf == 4:
        bestmodel = torch.load('./Checkpoint/Harvard/SpectralSolver_4x.pth')

    model = Modelharvard(n_layers=1,
                n_hidden=128,
                dropout=0.0,
                n_head=8,
                mlp_ratio=1,
                fun_dim=96,
                out_dim=31,
                slice_num=64,
                ref=8,
                H=1024, W=1024)
    model.load_state_dict(bestmodel)

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

inference_times = []

for j, (LR, RGB, HR, COORD) in enumerate(loader_test):
    with torch.no_grad():
        LR, RGB, HR, COORD = Variable(LR), Variable(RGB), Variable(HR), Variable(COORD)
        LR, RGB, HR, COORD = LR.cuda(), RGB.cuda(), HR.cuda(), COORD.cuda()

        start_time = time.time()

        up_LR = F.interpolate(LR, scale_factor=opt.sf, mode='bicubic', align_corners=False)
        out = model(COORD.cuda(), up_LR.cuda(), RGB.cuda(), HR.cuda())

        result = out

        end_time = time.time()
        inference_time = end_time - start_time
        inference_times.append(inference_time)

        result = result.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
        HR = HR.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1,2,0)

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

    std_psnr = np.std(psnr_total)
    std_sam = np.std(sam_total)
    std_ergas = np.std(ergas_total)
    std_ssim = np.std(ssim_total)

average_inference_time = sum(inference_times) / len(inference_times)
print("----------------")
print(k)
print("Avg PSNR = %.2f, Std PSNR = %.2f" % (avg_psnr, std_psnr))
print("Avg SAM = %.2f, Std SAM = %.2f" % (avg_sam, std_sam))
print("Avg ERGAS = %.2f, Std ERGAS = %.2f" % (avg_ergas, std_ergas))
print("Avg SSIM = %.3f, Std SSIM = %.3f" % (avg_ssim, std_ssim))