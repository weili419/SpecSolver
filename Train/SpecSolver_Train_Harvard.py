from data.Harvard_Dataset import harvard_dataset
import torch.utils.data as tud
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import time
import datetime
import argparse
from torch.autograd import Variable
from utils.SSIM import *
from utils.utils import *
import logging
from tqdm import tqdm
from Model.SpecSolver import Modelharvard
from torch.utils.tensorboard import SummaryWriter

def check_and_create_path_and_file(save_path, filename=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        print(f"Path {save_path} Does not exist, has been created.")
    else:
        print(f"Path {save_path} Already exists.")

    if filename is not None:
        file_path = os.path.join(save_path, filename)
        if not os.path.exists(file_path):
            with open(file_path, 'w') as file:
                file.write('')  
            print(f"File {filename} Does not exist, has been created.")
        else:
            print(f"File {filename} Already exists.")


def evaluate(save_path):
    parser_evaluate = argparse.ArgumentParser(description="PyTorch Code for HSI Fusion")
    parser_evaluate.add_argument('--data_path', default='/root/data1/dataset/Harvard/Test/', type=str,
                                 help='path of the testing data')
    parser_evaluate.add_argument("--sizeI", default=1024, type=int, help='the size of trainset')
    parser_evaluate.add_argument("--testset_num", default=10, type=int, help='total number of testset')
    parser_evaluate.add_argument("--batch_size", default=1, type=int, help='Batch size')
    parser_evaluate.add_argument("--sf", default=8, type=int, help='Scaling factor')
    parser_evaluate.add_argument("--scale", default=1, type=int, help='Scaling factor')
    parser_evaluate.add_argument("--seed", default=1, type=int, help='Random seed')
    parser_evaluate.add_argument("--kernel_type", default='gaussian_blur', type=str, help='Kernel type')
    opt_evaluate = parser_evaluate.parse_args()

    test_HR_HSI, test_HR_MSI = prepare_data_harvard(opt_evaluate.data_path, 10)
    test_dataset = harvard_dataset(opt_evaluate, test_HR_HSI, test_HR_MSI, istrain=False)
    loader_test = tud.DataLoader(test_dataset, batch_size=1, num_workers=8)
    Testmodel = Modelharvard(n_layers=1,
                    n_hidden=128,
                    dropout=0.0,
                    n_head=8,
                    mlp_ratio=1,
                    fun_dim=96,
                    out_dim=31,
                    slice_num=64,
                    ref=8,
                    H=1024, W=1024).cuda()
    Testmodel.load_state_dict(torch.load(save_path))
    Testmodel.eval()

    psnr_total = 0.
    k = 0
    for j, (LR, RGB, HR, COORD) in enumerate(loader_test):
        with torch.no_grad():
            up_LR = F.interpolate(LR, scale_factor=opt_evaluate.sf, mode='bicubic', align_corners=False)
            out= Testmodel(COORD.cuda(), up_LR.cuda(), RGB.cuda())
            result = out
            result = result.clamp(min=0., max=1.)
            HR = HR.clamp(min=0., max=1.)
        psnr = cal_psnr(result.cpu().data.squeeze().numpy().transpose(1, 2, 0),
                        HR.cpu().data.squeeze().numpy().transpose(1, 2, 0))
        psnr_total = psnr_total + psnr
        k = k + 1
    average_psnr = psnr_total / k
    return average_psnr

if __name__ == "__main__":

    ## Model Config
    parser = argparse.ArgumentParser(description="PyTorch Code for HSI Fusion")
    parser.add_argument('--data_path', default='/root/data1/dataset/Harvard/Train/', type=str,
                        help='Path of the training data')
    parser.add_argument("--sizeI", default=64, type=int, help='The image size of the training patches')
    parser.add_argument("--batch_size", default=50, type=int, help='Batch size')
    parser.add_argument("--trainset_num", default=20000, type=int, help='The number of training samples of each epoch')
    parser.add_argument("--sf", default=8, type=int, help='Scaling factor')
    parser.add_argument("--scale", default=1, type=int, help='Scaling factor')
    parser.add_argument("--seed", default=24, type=int, help='Random seed')
    parser.add_argument("--kernel_type", default='gaussian_blur', type=str, help='Kernel type')
    parser.add_argument('--model', type=str, default='SpecSolverHarverd', help='model type')
    opt = parser.parse_args()

    name = opt.model #
    log_dir = "./experiment"+'/'+ name+'/'+str(opt.sf)
    check_and_create_path_and_file(log_dir)
    log_dir_tb = log_dir +'/tensorboard' 
    check_and_create_path_and_file(log_dir_tb)
    log_check = log_dir +'/checkpoint'
    check_and_create_path_and_file(log_check)
    log_txt = log_dir + '/logg'
    check_and_create_path_and_file(log_txt, '%s.txt' % name)
    logger = logging.getLogger("LOG")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/logg/%s.txt' % (log_dir+'/', name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('=======Option used=======')

    writer = SummaryWriter(log_dir=log_dir_tb)

    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    print('==> Start saving configure......')
    print(opt)

    ## New model
    print("===> New Model")

    model = Modelharvard(n_layers=1,
                    n_hidden=128,
                    dropout=0.0,
                    n_head=8,
                    mlp_ratio=1,
                    fun_dim=96,
                    out_dim=31,
                    slice_num=64,
                    ref=8,
                    H=64, W=64).cuda()

    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f'[INFO] #parameters: {num_params / 1e6:.2f} M')

    ## Initialize weight
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
        if isinstance(layer, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(layer.weight)

    ## Load training data
    HR_HSI, HR_MSI = prepare_data_harvard(opt.data_path, 67)
    dataset = harvard_dataset(opt, HR_HSI, HR_MSI)
    loader_train = tud.DataLoader(dataset, num_workers=8, batch_size=opt.batch_size, shuffle=True)

    initial_epoch = 0

    ## Loss function
    criterion = nn.L1Loss()

    ## optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-8)
    scheduler = MultiStepLR(optimizer, milestones=list(range(1,500,75)), gamma=0.95)

    bestpsnr = 0.
    ep = 0
    ## pipline of training
    total_epoch = 1000
    for epoch in range(initial_epoch, total_epoch):
        model.train()
        epoch_loss = 0

        start_time = time.time()
        for i, (LR, RGB, HR, COORD) in enumerate(tqdm(loader_train, desc=f"Epoch {epoch}/{total_epoch}", ascii=True)):
            LR, RGB, HR, COORD = Variable(LR), Variable(RGB), Variable(HR), Variable(COORD)

            up_LR = F.interpolate(LR, scale_factor=opt.sf, mode='bicubic', align_corners=False)
            out = model(COORD.cuda(), up_LR.cuda(), RGB.cuda())
            loss = criterion(out, HR.cuda())
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        writer.add_scalar('Loss/train/epoch', epoch_loss / len(dataset), epoch)

        save_path = os.path.join(log_check, 'model_%04d.pth' % (epoch + 1))
        torch.save(model.state_dict(), save_path)  # save model

        current_lr = optimizer.param_groups[0]['lr']
        if epoch % 2 == 0:
            ave = evaluate(save_path)
            # ave_val = evaluate_val(model)
            if ave > bestpsnr:
                bestpsnr = ave
                ep = epoch
                torch.save(model.state_dict(), os.path.join(log_check, 'BestModel.pth'))  # save model
            logger.info('Epoch: {}/{} average psnr: {:.7f}  bestpsnr: {:.7f}, bestepoch: {}'.format(epoch, total_epoch, ave, bestpsnr, ep))
            writer.add_scalar('PSNR/test', ave, epoch)
            # writer.add_scalar('PSNR/val', ave_val, epoch)
            print('Epoch: {}/{} average psnr: {:.7f} bestpsnr: {:.7f}, bestepoch: {}'.format(epoch, total_epoch, ave, bestpsnr, ep))

        elapsed_time = time.time() - start_time
        print('Epcoh = %4d , loss = %.10f , time = %4.2f s' % (epoch + 1, epoch_loss / len(dataset), elapsed_time))

        if epoch > 0:  
            previous_model_path = os.path.join(log_check, 'model_%04d.pth' % epoch)
            if os.path.exists(previous_model_path):
                os.remove(previous_model_path)
                print(f'Removed previous model: {previous_model_path}')
            else:
                print(f'Previous model not found: {previous_model_path}')

    print('Epoch: {} BestPSNR: {:.7f}'.format(ep, bestpsnr))
    logger.info('Epoch: {} BestPSNR: {:.7f}'.format(ep, bestpsnr))
    writer.close()
