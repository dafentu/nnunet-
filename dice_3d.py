import SimpleITK as sitk
import numpy as np
import os


def get_listdir(path):  # 获取目录下所有gz格式文件的地址，返回地址list
    tmp_list = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.gz':
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list

def fei(i):
    if i==0:
        return 1
    elif i==1:
        return 0
class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,self.num_class))

    def Pixel_Accuracy(self):
        # acc = (TP + TN) / (TP + TN + FP + TN)
        Acc = np.diag(self.confusion_matrix).sum() / \
            self.confusion_matrix.sum()
        return Acc
    """
    confusionMetric,真真假假
    P\L     P    N

    P      TP    FP

    N      FN    TN

    """
    def precision(self):
        return self.confusion_matrix[0][0]/(self.confusion_matrix[0][0]+self.confusion_matrix[0][1])


    def recall(self):
        return self.confusion_matrix[0][0]/(self.confusion_matrix[0][0]+self.confusion_matrix[1][0])


    def F1(self):
        return 2*self.recall()*self.precision()/(self.recall()+self.precision())


    def specificity(self):
        return self.confusion_matrix[1][1]/(self.confusion_matrix[0][1]+self.confusion_matrix[1][1])



    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))
      #  print(self.confusion_matrix)
       # print(np.diag(self.confusion_matrix))
       # print(np.sum(self.confusion_matrix, axis=1))
       # print(np.sum(self.confusion_matrix, axis=0))
       # print(np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0))
       # print(np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0)-np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / \
            np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
       # mask = (gt_image >= 0) & (gt_image < self.num_class)
       # label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        #count = np.bincount(label, minlength=self.num_class**2)

       #self.confusion_matrix = np.zeros((self.num_class,self.num_class))
       #print(gt_image.shape)
      # print(pre_image.shape)
       confusion_matrix = np.zeros((self.num_class,self.num_class))
       pre_label = pre_image.flatten()
       true_label = gt_image.flatten()
       np.set_printoptions(threshold=sys.maxsize)
       for i, j in zip(true_label, pre_label):
           if i in [0,1] and j in [0,1]:
            confusion_matrix[fei(i)][fei(j)] += 1
           else:
            confusion_matrix[0][0] += 1
       return confusion_matrix




    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,self.num_class))


def dice_3d():
    root1 = '/home/user/hym/DATASET/nnUNet_raw/nnUNet_raw_data/Task606_PP/labelsTs'
    root2 = '/home/user/hym/DATASET/nnUNet_raw/nnUNet_raw_data/Task606_PP/inferTs'
    nums = len(os.listdir(root1))
    list = [95,142,143,144,145]
    ans_dice = 0
    ans_pc=0
    ans_rc=0
    ans_sp=0
    ans_miot=0
    ans_f1=0
    for i in range(nums):
        ground_path = os.path.join(root1, "%d_image.nii.gz" % list[i])
        predict_path = os.path.join(root2, "%d_image.nii.gz" % list[i])
        mask_sitk_img = sitk.ReadImage(ground_path)
        mask_img_arr = sitk.GetArrayFromImage(mask_sitk_img)
        pred_sitk_img = sitk.ReadImage(predict_path)
        pred_img_arr = sitk.GetArrayFromImage(pred_sitk_img)

        denominator = np.sum(mask_img_arr) + np.sum(pred_img_arr)
        numerator = 2 * np.sum(mask_img_arr * pred_img_arr)
        dice = numerator / denominator

        confusion_matrix = np.zeros((2, 2))
        pre_label = pred_img_arr.flatten()
        true_label = mask_img_arr.flatten()

        for i, j in zip(true_label, pre_label):
            if i in [0, 1] and j in [0, 1]:
                confusion_matrix[fei(i)][fei(j)] += 1
            else:
                confusion_matrix[0][0] += 1
        pc = confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[0][1])
        recall = confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[1][0]) #灵敏度
        sp = confusion_matrix[1][1]/(confusion_matrix[0][1]+confusion_matrix[1][1])   #特异性
        MIoU = np.diag(confusion_matrix) / (
                np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                np.diag(confusion_matrix))
        MIoU = np.nanmean(MIoU)
        f1 = 2 * recall * pc / (recall + pc)
        ans_dice += dice
        ans_pc += pc
        ans_rc += recall
        ans_sp += sp
        ans_miot += MIoU
        ans_f1 += f1
    return ans_dice / 5,ans_pc/5,ans_rc/5,ans_sp/5,ans_miot/5,ans_f1/5

print("dice,pc,灵敏度,特异性,miou,f1",dice_3d())
