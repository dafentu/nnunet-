import os
import shutil
import numpy as np

# 创建文件夹
img_tr_path = r'./DATASET/nnUNet_raw/nnUNet_raw_data/Task608_PA/imagesTr/'
img_ts_path = r'./DATASET/nnUNet_raw/nnUNet_raw_data/Task608_PA/imagesTs/'
label_tr_path = r'./DATASET/nnUNet_raw/nnUNet_raw_data/Task608_PA/labelsTr/'
label_ts_path = r'./DATASET/nnUNet_raw/nnUNet_raw_data/Task608_PA/labelsTs/'
infer_ts_path = r'./DATASET/nnUNet_raw/nnUNet_raw_data/Task608_PA/inferTs/'
if not os.path.isdir(img_tr_path):
    os.mkdir(img_tr_path)
    os.mkdir(img_ts_path)
    os.mkdir(label_tr_path)
    os.mkdir(label_ts_path)
    os.mkdir(infer_ts_path)
#[19, 19, 18, 18, 18, 18, 18, 18, 18, 16, 15, 18, 16, 20, 18, 20, 20, 22, 20, 18, 18, 18, 18, 18, 18, 20, 20, 18, 18, 20, 18, 22, 18, 18, 18, 18, 20, 18, 18, 20, 18, 18, 20, 18, 20, 18, 20, 18, 20, 20]
# [19, 19, 18, 18, 18, 18, 18, 18, 18, 16, 15, 18, 16, 20, 18, 20, 20, 22, 20, 18, 18, 18, 18, 18, 18, 20, 20, 18, 18, 20, 18, 22, 18, 18, 18, 18, 20, 18, 18, 20, 18, 18, 20, 18, 20, 18, 20, 18, 20, 20]
#获取训练、测试集的ID，按需修改
train_id = [i for i in range(1,99)]
train_id.remove(83)
test_id = [99,100,101,102,103]

# 复制数据文件并改成nnunet的命名形式
data_folder1 = r'./src'
data_folder2 = r'./if'		# 个人数据集的文件夹路径
for patient_id in train_id:
    # 预处理文件夹下文件名，我这里有两种数据模态PET/CT，以及一个分割标签mask

    ct_file = os.path.join(data_folder1,  " " +str(patient_id)+'.nii.gz')
    mask_file = os.path.join(data_folder2, " " +str(patient_id)+'.nii.gz')
    # nnunet文件夹文件名，nnUNet通过_0000和_0001这种形式分辨多模态输入
  
    ct_new_file = os.path.join(img_tr_path,str(patient_id) + '_image_0000.nii.gz')
    mask_new_file = os.path.join(label_tr_path, str(patient_id) + '_image.nii.gz')
	# 复制

    shutil.copyfile(ct_file, ct_new_file)
    shutil.copyfile(mask_file, mask_new_file)
#nnUNet_predict -i /home/user/hym/DATASET/nnUNet_raw/nnUNet_raw_data/Task501_PC/imagesTs/ -o /home/user/hym/DATASET/nnUNet_raw/nnUNet_raw_data/Task501_PC/inferTs/ -t 501 -m 3d_fullres
for patient_id in test_id: 
    # 预处理文件夹下文件名
   
    ct_file = os.path.join(data_folder1,  " " +str(patient_id)+'.nii.gz')
    mask_file = os.path.join(data_folder2, " " +str(patient_id)+'.nii.gz')
    # nnunet文件夹文件名
  
    ct_new_file = os.path.join(img_ts_path, str(patient_id) + '_image_0000.nii.gz')
    mask_new_file = os.path.join(label_ts_path, str(patient_id) + '_image.nii.gz')
    # 复制

    shutil.copyfile(ct_file, ct_new_file)
    shutil.copyfile(mask_file, mask_new_file)
