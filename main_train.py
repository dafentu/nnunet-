
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

import torch
import argparse
from vnet3d import VNet
from unet3d import UNet3D
from torch import optim
from dataset_3d import MyDataset
from torch.utils.data import DataLoader
import numpy as np
import logging
import time
#from u4d import BalancedDataParallel
from monai.networks.nets import UNETR

from torch.utils.data.distributed import DistributedSampler
# 这个参数是torch.distributed.launch传递过来的，我们设置位置参数来接受，local_rank代表当前程序进程使用的GPU标号

torch.distributed.init_process_group(backend="nccl")
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

#device = torch.device("cuda")
respth = './res'
if not os.path.exists(respth):
    os.makedirs(respth)
logger = logging.getLogger()
class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict) # sigmoide获取概率
        tmp1 = pt.cpu().detach().numpy().flatten()

        tmp2 = target.cpu().detach().numpy().flatten()
        n = np.array([0])
        
        for i,j in zip(tmp1,tmp2):
           if j==1:
              n=np.append(n,[i])
        print(n)

        N = target.size(0)
        print("batch",N)
        pred_flat = pt.view(-1)
        gt_flat = target.view(-1)
        smooth = 1e-5
        intersection = (pred_flat * gt_flat).sum()
        unionset = pred_flat.sum() + gt_flat.sum()
        l = 2 * (intersection + smooth) / (unionset + smooth)
      #  l1 = torch.nn.CrossEntropyLoss(pt,target)
      #  l1 = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
   #     l1 = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
    #    return 1-loss
    #    loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
    #    if self.reduction == 'mean':
   #         loss = torch.mean(loss)
   #     elif self.reduction == 'sum':
    #        loss = torch.sum(loss)
        return 1-l

def setup_logger(logpth):
    logfile = 'UNet3d-Pre-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = os.path.join(logpth, logfile)
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())


def train(model):
    model.train()
    if args.load:
        model.load_state_dict(torch.load(args.load,map_location='cuda'))
        logger.info('ok')
    loss_fn = BCEFocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    source_train_dir = r'/home/user/hym2/segmentation_3d-main/src'
    label_train_dir = r'/home/user/hym2/segmentation_3d-main/bi'
    train_dataset = MyDataset(source_train_dir, label_train_dir)
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)
    train_dataloader = DataLoader(train_dataset.training_set,
                                  batch_size=1,
                                  pin_memory=True,
                                  drop_last=True,
                                  sampler=DistributedSampler(train_dataset))
    for epoch in range(587,args.num_epochs):
        logger.info('Epoch {}/{}'.format(epoch + 1, args.num_epochs))
        logger.info('-' * 10)
        dataset_size = len(train_dataloader.dataset)
        epoch_loss = 0
        step = 0
        for i, batch in enumerate(train_dataloader):
            x = batch['source']['data']
            y = batch['label']['data']
            x = x.to(device)
            y = torch.squeeze(y, 1).long()
            y = y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn.forward(output, y)

       
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
            logger.info("%d/%d,train_loss:%0.5f" % (step, dataset_size // train_dataloader.batch_size, loss.item()))
        logger.info("epoch %d sumloss:%0.5f" % (epoch, epoch_loss))
        torch.save(model.module.state_dict(), './model/UNet3d_%d_bi.pth' % epoch)


if __name__ == '__main__':
    setup_logger(respth)
    model = UNETR(
        img_size=(384, 384, 32),
        in_channels=1,
        out_channels=1,).to(device)

  #  model.to(device)
 
 #   if torch.cuda.device_count() > 1:
 #       print("Let's use", torch.cuda.device_count(), "GPUs!")
  #  model = torch.nn.DataParallel(model)
   #  5) 封装
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=2, help='batch_size')
    parser.add_argument('--load', dest='load', type=str,default='./model/UNet3d_586_bi.pth' ,help='the path of the .pth file')
    parser.add_argument('--epoch', dest='num_epochs', type=int, default=800, help='num_epochs')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.001, help='learning_rate')
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    train(model)
