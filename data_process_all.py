import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd

from nilearn import plotting
from nilearn import image
import cv2
import os
import glob

path1 = '/media/fimcp/DATA/WMH - Original/Utrecht/'
path2 = '/media/fimcp/DATA/WMH - Original/GE3T'
path3 = '/media/fimcp/DATA/WMH - Original/Singapore/'
model = '/media/fimcp/DATA/WMH/Models/'

img_row = 128
img_col = 128


def image_processing_all(paths):
    image_dataset = []
    # t1_dataset = np.empty((80,img_row,img_col,48))
    # flair_dataset = np.empty((80,img_row,img_col,48))
    # mask_dataset = np.empty((80,img_row,img_col,48))
    t1_dataset = []
    flair_dataset = []
    mask_dataset = []

    for path in paths:
        t1_dataset_, flair_dataset_, mask_dataset_ = image_processing(path)
        # t1_dataset = np.append(t1_dataset, t1_dataset_, axis= 0)
        # flair_dataset = np.append(flair_dataset, flair_dataset_, axis= 0)
        # mask_dataset = np.append(mask_dataset, mask_dataset_, axis= 0)
        t1_dataset += t1_dataset_
        flair_dataset += flair_dataset_
        mask_dataset += mask_dataset_

    # image_dataset = [t1_dataset, flair_dataset]
        # image_dataset.append(flair_dataset)


    flair_array = np.ndarray(flair_dataset)
    t1_array = np.ndarray(t1_dataset)
    image_array = np.concatenate((flair_array, t1_array), axis=3)
    mask_array = np.array(mask_dataset)

    return image_array, mask_array


def image_processing(file_path):
    data_path = os.listdir(file_path)
    image_dataset = []
    flair_dataset = []
    t1_dataset = []
    mask_dataset = []
    slices_id_labels = []
    j = 0
    for i in data_path:
        img_path = os.path.join(file_path, i, 'pre')
        mask_path = os.path.join(file_path, i)

        for name in glob.glob(img_path + '/FLAIR.nii.gz*'):
            flair_img = image.load_img(name)
            flair_data = flair_img.get_data()
            flair_data = np.transpose(flair_data, (1, 0, 2))
            flair_resized = cv2.resize(flair_data, dsize=(img_row, img_col), interpolation=cv2.INTER_CUBIC)
            flair_dataset.append(flair_resized)
            # perform some image augmentation (use same parameters for mask for deterministic augmentation)
            # rotate
            M1 = cv2.getRotationMatrix2D((img_row / 2, img_col / 2), 90, 1)
            flair_rotate = cv2.warpAffine(flair_resized, M1, (img_row, img_col))
            flair_dataset.append(flair_rotate)
            # shearing
            pts1 = np.float32([[50, 50], [100, 100], [50, 200]])
            pts2 = np.float32([[10, 100], [100, 100], [100, 210]])
            M2 = cv2.getAffineTransform(pts1, pts2)
            flair_shear = cv2.warpAffine(flair_resized, M2, (img_row, img_col))
            flair_dataset.append(flair_shear)
            # zoom
            pts3 = np.float32([[45, 48], [124, 30], [50, 120], [126, 126]])
            pts4 = np.float32([[0, 0], [128, 0], [0, 128], [128, 128]])
            M3 = cv2.getPerspectiveTransform(pts3, pts4)
            flair_zoom = cv2.warpPerspective(flair_resized, M3, (img_row, img_col))
            flair_dataset.append(flair_zoom)

        for name in glob.glob(img_path + '/T1.nii.gz*'):
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
        for name in glob.glob(mask_path + '/wmh*'):
            mask_img = image.load_img(name)
            mask_data = mask_img.get_data()
            mask_data = np.transpose(mask_data, (1, 0, 2))  # transpose so orientation matches nilearn plot
            mask_resized = cv2.resize(mask_data, dsize=(img_row, img_col), interpolation=cv2.INTER_CUBIC)
            ret, mask_binary = cv2.threshold(mask_resized, 0.6, 1, cv2.THRESH_BINARY)
            mask_dataset.append(mask_binary)
            # need to run binary threshold again after augmentation
            mask_rotate = cv2.warpAffine(mask_binary, M1, (img_row, img_col))
            ret, mask_rotate = cv2.threshold(mask_rotate, 0.6, 1, cv2.THRESH_BINARY)
            mask_dataset.append(mask_rotate)

            mask_shear = cv2.warpAffine(mask_binary, M2, (img_row, img_col))
            ret, mask_shear = cv2.threshold(mask_shear, 0.6, 1, cv2.THRESH_BINARY)
            mask_dataset.append(mask_shear)

            mask_zoom = cv2.warpPerspective(mask_binary, M3, (img_row, img_col))
            ret, mask_zoom = cv2.threshold(mask_zoom, 0.6, 1, cv2.THRESH_BINARY)
            mask_dataset.append(mask_zoom)

        j += 1

    flair_array = np.array(flair_dataset)
    t1_array = np.array(t1_dataset)
    mask_array = np.array(mask_dataset)
    # return flair_array, t1_array,  mask_array
    return flair_dataset, t1_dataset,  mask_dataset


# utrecht_flair, utrecht_mask = image_processing(path1)
# amsterdam_flair, amsterdam_mask = image_processing(path2)
# singapore_flair, singapore_mask = image_processing(path3)

image_data , mask_data = image_processing_all([path1, path2, path3])

np.save(model + 'imgs_three_datasets_two_channels.npy', image_data)
np.save(model + 'imgs_mask_three_datasets_two_channels.npy', mask_data)

# np.save(model + 'amsterdam_flair(128)aug.npy', amsterdam_flair)
# np.save(model + 'amsterdam_mask(128)aug.npy', amsterdam_mask)
#
# np.save(model + 'singapore_flair(128)aug.npy', singapore_flair)
# np.save(model + 'singapore_mask(128)aug.npy', singapore_mask)
