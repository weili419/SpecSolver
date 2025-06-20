import torch.utils.data as tud
from Utils import *

try:
    from torch import irfft
    from torch import rfft
except ImportError:
    from torch.fft import irfft2
    from torch.fft import rfft2
    
    def rfft(x, d):
        t = rfft2(x, dim = (-d))
        return torch.stack((t.real, t.imag), -1)
    def irfft(x, d, signal_sizes):
        return irfft2(torch.complex(x[:,:,0], x[:,:,1]), s = signal_sizes, dim = (-d))

def add_impulse_noise(img, amount=0.02, salt_vs_pepper=0.5):
    """
    输入 img: torch.Tensor, shape [C, H, W]
    输出: torch.Tensor, shape [C, H, W]
    """
    # assert isinstance(img, torch.Tensor)
    img_np = img.cpu().numpy()
    c, h, w = img.shape
    img_np = np.transpose(img_np, (1, 2, 0))  # HWC

    num_pixels = h * w
    n_salt = int(amount * num_pixels * salt_vs_pepper)
    n_pepper = int(amount * num_pixels * (1. - salt_vs_pepper))

    # 随机选像素
    idx = np.random.choice(num_pixels, n_salt + n_pepper, replace=False)
    rows, cols = np.unravel_index(idx, (h, w))
    max_v = img_np.max()
    min_v = img_np.min()
    # salt
    img_np[rows[:n_salt], cols[:n_salt], :] = max_v
    # pepper
    img_np[rows[n_salt:], cols[n_salt:], :] = min_v

    img_np = np.transpose(img_np, (2, 0, 1))  # 回到CHW
    return torch.from_numpy(img_np).type_as(img)

class cave_dataset(tud.Dataset):
    def __init__(self, opt, HR_HSI, HR_MSI, istrain = True):
        super(cave_dataset, self).__init__()
        self.path = opt.data_path
        self.istrain  =  istrain
        self.factor = opt.sf
        if istrain:
            self.num = opt.trainset_num
            self.file_num = 20
            self.sizeI = opt.sizeI
        else:
            self.num = opt.testset_num
            self.file_num = 12
            self.sizeI = 512
        self.HR_HSI, self.HR_MSI = HR_HSI, HR_MSI

    def H_z(self, z, factor, fft_B):
        f = torch.fft.fft2(z, dim=(-2, -1))
        f = torch.stack((f.real,f.imag),-1)
        # -------------------complex myltiply-----------------#
        if len(z.shape) == 3:
            ch, h, w = z.shape
            fft_B = fft_B.unsqueeze(0).repeat(ch, 1, 1, 1)
            M = torch.cat(((f[:, :, :, 0] * fft_B[:, :, :, 0] - f[:, :, :, 1] * fft_B[:, :, :, 1]).unsqueeze(3),
                           (f[:, :, :, 0] * fft_B[:, :, :, 1] + f[:, :, :, 1] * fft_B[:, :, :, 0]).unsqueeze(3)), 3)
            Hz = torch.irfft(M, 2, onesided=False)
            x = Hz[:, int(factor // 2)-1::factor, int(factor // 2)-1::factor]
        elif len(z.shape) == 4:
            bs, ch, h, w = z.shape
            fft_B = fft_B.unsqueeze(0).unsqueeze(0).repeat(bs, ch, 1, 1, 1)
            M = torch.cat(
                ((f[:, :, :, :, 0] * fft_B[:, :, :, :, 0] - f[:, :, :, :, 1] * fft_B[:, :, :, :, 1]).unsqueeze(4),
                 (f[:, :, :, :, 0] * fft_B[:, :, :, :, 1] + f[:, :, :, :, 1] * fft_B[:, :, :, :, 0]).unsqueeze(4)), 4)
            #Hz = torch.irfft(M, 2, onesided=False)
            Hz = torch.fft.ifft2(torch.complex(M[..., 0],M[..., 1]), dim=(-2, -1))
            x = Hz[:, :, int(factor // 2)-1::factor, int(factor // 2)-1::factor]
        return x.real

    def __getitem__(self, index):
        if self.istrain == True:
            index1   = random.randint(0, self.file_num-1)
        else:
            index1 = index

        sigma = 2.0
        HR_HSI = self.HR_HSI[:,:,:,index1]
        HR_MSI = self.HR_MSI[:,:,:,index1]

        sz = [self.sizeI, self.sizeI]
        fft_B, fft_BT = para_setting('gaussian_blur', self.factor, sz, sigma)
        fft_B = torch.cat((torch.Tensor(np.real(fft_B)).unsqueeze(2), torch.Tensor(np.imag(fft_B)).unsqueeze(2)),2)
        fft_BT = torch.cat((torch.Tensor(np.real(fft_BT)).unsqueeze(2), torch.Tensor(np.imag(fft_BT)).unsqueeze(2)), 2)

        px      = random.randint(0, 512-self.sizeI)
        py      = random.randint(0, 512-self.sizeI)
        hr_hsi  = HR_HSI[px:px + self.sizeI:1, py:py + self.sizeI:1, :]
        hr_msi  = HR_MSI[px:px + self.sizeI:1, py:py + self.sizeI:1, :]

        if self.istrain == True:
            rotTimes = random.randint(0, 3)
            vFlip    = random.randint(0, 1)
            hFlip    = random.randint(0, 1)

            # Random rotation
            for j in range(rotTimes):
                hr_hsi  =  np.rot90(hr_hsi)
                hr_msi  =  np.rot90(hr_msi)

            # Random vertical Flip
            for j in range(vFlip):
                hr_hsi = hr_hsi[:, ::-1, :].copy()
                hr_msi = hr_msi[:, ::-1, :].copy()

            # Random horizontal Flip
            for j in range(hFlip):
                hr_hsi = hr_hsi[::-1, :, :].copy()
                hr_msi = hr_msi[::-1, :, :].copy()

        hr_hsi = torch.FloatTensor(hr_hsi.copy()).permute(2,0,1).unsqueeze(0)
        hr_msi = torch.FloatTensor(hr_msi.copy()).permute(2,0,1).unsqueeze(0)
        lr_hsi = self.H_z(hr_hsi, self.factor, fft_B)
        lr_hsi = torch.FloatTensor(lr_hsi)

        hr_hsi = hr_hsi.squeeze(0)
        hr_msi = hr_msi.squeeze(0)
        lr_hsi = lr_hsi.squeeze(0)

        lr_hsi = add_impulse_noise(lr_hsi)   # impulse_noise

        coord  = make_coord((self.sizeI, self.sizeI), flatten=False)

        return lr_hsi, hr_msi, hr_hsi, coord

    def __len__(self):
        return self.num
    
