import torch
import argparse
from vnet3d import VNet
from unet3d import UNet3D
from dataset_3d import test_dataset
from torch.utils.data import DataLoader
import numpy as np
from monai.networks.nets import UNETR
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'

import tqdm
import torchio
from torchio.transforms import ZNormalization, CropOrPad, Compose, Resample, Resize
import SimpleITK as sitk


def test(model):
    
    model.load_state_dict(torch.load(args.load, map_location='cuda'))
    model = torch.nn.DataParallel(model)
    model.eval()
    batch_size = args.batch_size
    source_test_dir = r'./src'
    save_path1 = r'./out'
    save_path2 = r'./out2'
    dataset = test_dataset(source_test_dir)
    patch_overlap = 64, 64, 64
    patch_size = 128
    cnt=0
    for i, subj in enumerate(dataset.test_set):
     #   grid_sampler = torchio.inference.GridSampler(subj, patch_size, patch_overlap)  # 从图像中提取patch
      #  patch_loader = torch.utils.data.DataLoader(subj], batch_size)
     #   aggregator = torchio.inference.GridAggregator(grid_sampler, 'average')  # 用于聚合patch推理结果
        with torch.no_grad():
            for patches_batch in tqdm.tqdm(subj):

                cnt+=1
                input_tensor = torch.unsqueeze(subj['source'][torchio.DATA].to(device).float(),0)
                outputs = torch.sigmoid(model(input_tensor))
                outputs[outputs>0.5]=1
                outputs[outputs<0.5]=0
                outputs = torch.squeeze(outputs,0)
            #    outputs = torch.squeeze(outputs,0)
          #      locations = patches_batch[torchio.LOCATION]  # patch的位置信息
         #       aggregator.add_batch(outputs, locations)
      #  output_tensor = aggregator.get_output_tensor()  # 获取聚合后volume
        affine = subj['source']['affine']
        input_tensor = torch.squeeze(input_tensor,0)
        input_image = torchio.ScalarImage(tensor=input_tensor.cpu().numpy(), affine=affine)
        output_image = torchio.ScalarImage(tensor=outputs.cpu().numpy(), affine=affine)
        new_img = sitk.GetImageFromArray(np.transpose(np.squeeze(input_image),(2, 1, 0)))
        new_img.SetSpacing(output_image.spacing)
        new_img.SetDirection(output_image.direction)
        new_img.SetOrigin(output_image.origin)
        
     #   out_transform = Resize(dataset.get_shape(i)[1:])  # 恢复原始大小
     #   output_image = out_transform(output_image)
 
        name = subj['source']['path']
        _, fullflname = os.path.split(name)
        sitk.WriteImage(new_img, os.path.join(save_path1, fullflname))
        new_mask = sitk.GetImageFromArray(np.transpose(np.squeeze(output_image),(2, 1, 0)))
        new_mask.SetSpacing(output_image.spacing)
        new_mask.SetDirection(output_image.direction)
        new_mask.SetOrigin(output_image.origin)
        
        sitk.WriteImage(new_mask, os.path.join(save_path2, fullflname))
	
        # output_image.save(os.path.join(save_path, fullflname))
        # print(os.path.join(save_path, fullflname) + '保存成功')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = VNet(elu=True, in_channels=1, classes=3).to(device)
    model = UNETR(
        img_size=(384, 384, 32),
        in_channels=1,
        out_channels=1,).to(device)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--load', dest='load', type=str, default='./model/UNet3d_650_bi.pth',
                        help='the path of the .pth file')
    args = parser.parse_args()
    test(model)
