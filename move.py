import SimpleITK as sitk
import os
import numpy as np
import shutil
from itertools import combinations

def get_listdir(path):  # 获取目录下所有gz格式文件的地址，返回地址list
    tmp_list = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.gz':
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list
'''
def jug(a,b):
    c = a.flatten()
    d = b.flatten()
    for i,j in zip(c,d):
        if i!=0 and j!=0:
            return False
imagepath = r'G:/BrainInjury_dicom-QUAN/'

lis = os.listdir(imagepath)
f = ["_BFV","_CV","_FLV","_OLV","_TCV","_TLV","_TV"]
f1 = ["_BFV","_OLV","_FLV","_TLV","_TCV"]
c=0
for a in range(1, 2):
    for k in combinations(f1, a):
        k += ('_CV', '_TV')
        #k = ["_BFV","_BI","_CV","_FLV"]
        #print(type(k))
        flag = True
        for i in lis:
                tmp_list=os.path.join(imagepath+i)
                cnt = 0
                l = get_listdir(tmp_list)
                s = sitk.GetArrayFromImage(sitk.ReadImage(l[0], sitk.sitkFloat32)).shape[0]
                image = np.zeros([512, 512, s])
                for j in l:
                    for z in k:
                        if z in j:

                        # print(k)
                            image2 = sitk.ReadImage(j, sitk.sitkFloat32)
                           # print(j)
                            npImage = sitk.GetArrayFromImage(image2)

                            npImage = npImage.transpose(2, 1, 0)
                            cnt += 1
                            npImage *= cnt
                            if jug(image, npImage) == False:
                                flag = False
                                break
                            image += npImage
                    if flag == False:
                        break
                if flag == False:
                    break
        if flag == True:
            print(k.__str__() + "能够行")

        else:
            print(k.__str__() + "不行")


   # print(s)
    #print(l[0])
   # print("ss")





   # c += 1
   # image = image.transpose(2, 1, 0)
   # sitkImage = sitk.GetImageFromArray(image)
#np.set_printoptions(threshold=np.inf)
    #print(sitkImage)
   # writepath = r'E:/cv_bfv_flv/' + str(c) + '.nii.gz'

   # sitk.WriteImage(sitkImage, writepath)
     # 保存图像路径
'''

cnt = 0
imagepath = "/home/user/hym/103-Important region-quan/103SJ-important region/"
list = os.listdir(r"/home/user/hym/103-Important region-quan/103SJ-important region/")
for i in list:
    tmp_path = imagepath + str(i)
    tmp_list = os.listdir(tmp_path)
    cnt += 1
    for j in tmp_list:
        if "LV" in j and "T2" in j:
            shutil.copyfile(tmp_path + "/" + j, "./lv/" + str(cnt) + ".nii.gz")
        elif "src" in j and "T2" in j:
            shutil.copyfile(tmp_path +"/"+ j, "./src/ "+ str(cnt) +".nii.gz")
        elif "IF" in j and "T2" in j:
            shutil.copyfile(tmp_path +"/"+ j, "./if/ "+ str(cnt) +".nii.gz")
    
               