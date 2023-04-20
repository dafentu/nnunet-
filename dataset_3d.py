import torch.utils.data as data
import os
import torch
import numpy as np
from torchio.data import UniformSampler
from torchio.transforms import ZNormalization, CropOrPad, Compose, Resample, Resize,ToCanonical
import torchio
from torch.utils.data import DataLoader


def get_listdir(path):
    tmp_list = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.gz':
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list


class MyDataset(data.Dataset):
    def __init__(self, imgs_path, mask_path):
        self.img_list = get_listdir(imgs_path)
        self.img_list.sort()
        self.mask_list = get_listdir(mask_path)
        self.mask_list.sort()
        self.subjects = []
        for (image_path, label_path) in zip(self.img_list, self.mask_list):
            subject = torchio.Subject(
                source=torchio.ScalarImage(image_path),

                label=torchio.LabelMap(label_path),
            )
            self.subjects.append(subject)
        self.transforms = self.transform()
        self.training_set = torchio.SubjectsDataset(self.subjects, transform=self.transforms)
        queue_length = 5
        samples_per_volume = 5
        patch_size = 128
        self.queue_dataset = torchio.Queue(self.training_set, queue_length, samples_per_volume,
                                           UniformSampler(patch_size))

    def transform(self):
        crop_or_pad_size = (320, 320, 32)
        training_transform = Compose([
            Resample(),
            Resize((384,384 , 32)),
            ZNormalization(),
       #     CropOrPad(crop_or_pad_size, padding_mode='reflect'),
       #     ToCanonical(),
      #      CropOrPad(crop_or_pad_size, padding_mode='reflect'),
       #     Resample(),
           
        ])
        return training_transform
    def __len__(self):
        return len(self.subjects)


class test_dataset(data.Dataset):
    def __init__(self, imgs_path):
        self.img_list = get_listdir(imgs_path)
        self.img_list.sort()
        self.subjects = []
        for image_path in self.img_list:
            subject = torchio.Subject(
                source=torchio.ScalarImage(image_path)
            )

            self.subjects.append(subject)
        self.transforms = self.transform()

        self.test_set = torchio.SubjectsDataset(self.subjects, transform=self.transforms)

    def transform(self):
        test_transform = Compose([
            Resample(),
            Resize((384,384 , 32)),
            
            ZNormalization(),

        ])
        return test_transform

    def get_shape(self, i):
        return self.subjects[i].shape


if __name__ == '__main__':
    source_train_dir = r'/home/user/hym2/segmentation_3d-main/src'
    label_train_dir = r'/home/user/hym2/segmentation_3d-main/bi'
    train_dataset = MyDataset(source_train_dir, label_train_dir)
    train_dataloader = DataLoader(train_dataset.training_set,
                                  batch_size=1,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)

    for i, batch in enumerate(train_dataloader):
        x = batch['source']['data']
        y = batch['label']['data']
        torch.set_printoptions(threshold=np.inf)
        print(y.shape)

