import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd

from nilearn import plotting
from nilearn import image
from scipy import ndimage as nd
import cv2
import os
import glob
import SimpleITK as sitk

path1='/media/fimcp/DATA/WMH - Original/Utrecht/'
path2='/media/fimcp/DATA/WMH - Original/GE3T'
path3='/media/fimcp/DATA/WMH - Original/Singapore/'
model = '/media/fimcp/DATA/WMH/Models/'

img_row=128
img_col=128


def flair_times_mask(flair_image, t1_mask):
    shape_flair = np.shape(flair_image)
    shape_mask = np.shape(t1_mask)
    for i in range(shape_flair):
        pass


def image_processing(file_path):
    data_path=os.listdir(file_path)
    flair_dataset=[]
    t1_dataset = []
    mask_dataset=[]
    for i in data_path:
        img_path=os.path.join(file_path, i,'pre')
        mask_path=os.path.join(file_path,i)

        for name in glob.glob(img_path+'/FLAIR.nii.gz*'):
            t1_mask = glob.glob(img_path+'/T1-SS_mask.nii.gz*')
            t1_mask_img = image.load_img(t1_mask[0])
            t1_mask_data = t1_mask_img.get_data()
            t1_mask_data = np.transpose(t1_mask_data)
            flair_img=image.load_img(name)
            flair_data=flair_img.get_data()
            # flair_times_mask(flair_data, t1_mask_data[...,0])
            # image.new_img_like(flair_img, flair_data)
            dsfactor = [w/float(f) for w,f in zip(flair_data.shape, t1_mask_data.shape)]
            t1_mask_data = nd.interpolation.zoom(t1_mask_data, zoom=dsfactor)

            flair_data = np.multiply(flair_data, t1_mask_data)
            flair_data=np.transpose(flair_data, (1,0,2))
            flair_resized = cv2.resize(flair_data, dsize=(img_row,img_col), interpolation=cv2.INTER_CUBIC)
            filename_resultImage = os.path.join(model, 'T1_2.nii.gz')
            sitk.WriteImage(sitk.GetImageFromArray(flair_resized), filename_resultImage)

            flair_dataset.append(flair_resized)

        #perform some image augmentation (use same parameters for mask for deterministic augmentation)
        #rotate
            M1 = cv2.getRotationMatrix2D((img_row/2,img_col/2),90,1)
            flair_rotate= cv2.warpAffine(flair_resized,M1,(img_row,img_col))
            flair_dataset.append(flair_rotate)
        #shearing
            pts1 = np.float32([[50,50],[100,100],[50,200]])
            pts2 = np.float32([[10,100],[100,100],[100,210]])
            M2 = cv2.getAffineTransform(pts1,pts2)
            flair_shear= cv2.warpAffine(flair_resized,M2,(img_row,img_col))
            flair_dataset.append(flair_shear)
        #zoom
            pts3 = np.float32([[45,48],[124,30],[50,120],[126,126]])
            pts4= np.float32([[0,0],[128,0],[0,128],[128,128]])
            M3 = cv2.getPerspectiveTransform(pts3,pts4)
            flair_zoom = cv2.warpPerspective(flair_resized,M3,(img_row,img_col))
            flair_dataset.append(flair_zoom)

        for name in glob.glob(img_path + '/T1-SS.nii.gz*'):
            flair_img = image.load_img(name)
            t1_data = flair_img.get_data()
            t1_data = np.transpose(t1_data, (1, 0, 2))
            t1_resized = cv2.resize(t1_data, dsize=(img_row, img_col), interpolation=cv2.INTER_CUBIC)
            t1_dataset.append(t1_resized)
            # perform some image augmentation (use same parameters for mask for deterministic augmentation)
            # rotate
            M1 = cv2.getRotationMatrix2D((img_row / 2, img_col / 2), 90, 1)
            t1_rotate = cv2.warpAffine(t1_resized, M1, (img_row, img_col))
            t1_dataset.append(t1_rotate)
            # shearing
            pts1 = np.float32([[50, 50], [100, 100], [50, 200]])
            pts2 = np.float32([[10, 100], [100, 100], [100, 210]])
            M2 = cv2.getAffineTransform(pts1, pts2)
            t1_shear = cv2.warpAffine(t1_resized, M2, (img_row, img_col))
            t1_dataset.append(t1_shear)
            # zoom
            pts3 = np.float32([[45, 48], [124, 30], [50, 120], [126, 126]])
            pts4 = np.float32([[0, 0], [128, 0], [0, 128], [128, 128]])
            M3 = cv2.getPerspectiveTransform(pts3, pts4)
            t1_zoom = cv2.warpPerspective(t1_resized, M3, (img_row, img_col))
            t1_dataset.append(t1_zoom)

        # perform same transformation on mask files
        for name in glob.glob(mask_path+'/wmh*'):
            mask_img=image.load_img(name)
            mask_data=mask_img.get_data()
            mask_data=np.transpose(mask_data, (1,0,2)) #transpose so orientation matches nilearn plot
            mask_resized=cv2.resize(mask_data, dsize=(img_row,img_col), interpolation=cv2.INTER_CUBIC)
            ret, mask_binary=cv2.threshold(mask_resized,0.6,1,cv2.THRESH_BINARY)
            mask_dataset.append(mask_binary)
            #need to run binary threshold again after augmentation
            mask_rotate= cv2.warpAffine(mask_binary,M1,(img_row,img_col))
            ret, mask_rotate=cv2.threshold(mask_rotate,0.6,1,cv2.THRESH_BINARY)
            mask_dataset.append(mask_rotate)
                
            mask_shear= cv2.warpAffine(mask_binary,M2,(img_row,img_col))
            ret, mask_shear=cv2.threshold(mask_shear,0.6,1,cv2.THRESH_BINARY)
            mask_dataset.append(mask_shear)
                
            mask_zoom= cv2.warpPerspective(mask_binary,M3,(img_row,img_col))
            ret, mask_zoom=cv2.threshold(mask_zoom,0.6,1,cv2.THRESH_BINARY)
            mask_dataset.append(mask_zoom)
                

    flair_array=np.array(flair_dataset)
    t1_array = np.array(t1_dataset)
    mask_array=np.array(mask_dataset)
    return flair_array, mask_array,t1_array


utrecht_flair, utrecht_mask , utrecht_t1 = image_processing(path1)
amsterdam_flair, amsterdam_mask, amsterdam_t1 = image_processing(path2)
singapore_flair, singapore_mask, singapore_t1 = image_processing(path3)

np.save(model + 'utrecht_flair(200)aug.npy', utrecht_flair)
np.save(model + 'utrecht_t1(200)aug.npy', utrecht_t1)
np.save(model + 'utrecht_mask(200)aug.npy', utrecht_mask)

np.save(model + 'amsterdam_flair(200)aug.npy', amsterdam_flair)
np.save(model + 'amsterdam_t1(200)aug.npy', amsterdam_t1)
np.save(model + 'amsterdam_mask(200)aug.npy', amsterdam_mask)

np.save(model + 'singapore_flair(200)aug.npy', singapore_flair)
np.save(model + 'singapore_t1(200)aug.npy', singapore_t1)
np.save(model + 'singapore_mask(200)aug.npy', singapore_mask)
