import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from itertools import izip
import os

import keras
keras.__version__

import SimpleITK as sitk
from keras.models import Model,load_model
from keras.layers import Input, concatenate,Activation, Conv3D, MaxPooling3D,UpSampling3D, BatchNormalization, Dropout, ZeroPadding3D
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.callbacks import EarlyStopping, History

K.set_image_dim_ordering('tf')
K.image_dim_ordering()

model_file = '/media/fimcp/DATA/WMH/Models/'
result_file = '/media/fimcp/DATA/WMH/Results/'

ROWS=128
COLS=128

imgs_utrecht = np.load(model_file + 'utrecht_flair(200)aug.npy')
imgs_t1_utrecht = np.load(model_file + 'utrecht_t1(200)aug.npy')
mask_utrecht = np.load(model_file + 'utrecht_mask(200)aug.npy')

imgs_amsterdam = np.load(model_file + 'amsterdam_flair(200)aug.npy')
imgs_t1_amsterdam = np.load(model_file + 'amsterdam_t1(200)aug.npy')
mask_amsterdam= np.load(model_file + 'amsterdam_mask(200)aug.npy')

imgs_singapore = np.load(model_file + 'singapore_flair(200)aug.npy')
imgs_t1_singapore = np.load(model_file + 'singapore_t1(200)aug.npy')
mask_singapore= np.load(model_file + 'singapore_mask(200)aug.npy')


print(imgs_utrecht.shape)
print(mask_utrecht.shape)
print(imgs_amsterdam.shape)
print(mask_amsterdam.shape)
print(imgs_singapore.shape)
print(mask_singapore.shape)


############## optional: take a look at the image ################
plt.subplot(1,2,1)
plt.imshow(imgs_utrecht[12][:,:,24],cmap='gray')
plt.subplot(1,2,2)
plt.imshow(mask_utrecht[12][:,:,24],cmap='gray')

## Slice the data array so z-dimension is the same across samples
#### get 16 slices for model training to reduce computation time

# # train_img_ut=imgs_utrecht[:,:,:,9:48] + imgs_t1_utrecht[:,:,:,9:48]
# train_img_ut=imgs_utrecht[:,:,:,9:48]
# train_mask_ut=mask_utrecht[:,:,:,9:48]
#
# # train_img_am=imgs_amsterdam[:,:,:,28:67] + imgs_t1_amsterdam[:,:,:,28:67]
# train_img_am=imgs_amsterdam[:,:,:,28:67]
# train_mask_am=mask_amsterdam[:,:,:,28:67]
#
# # train_img_si=imgs_singapore[:,:,:,9:48] + imgs_t1_singapore[:,:,:,9:48]
# train_img_si=imgs_singapore[:,:,:,9:48]
# train_mask_si=mask_singapore[:,:,:,9:48]

train_img_ut=imgs_utrecht[:,:,:,20:36]
train_img_t1_ut=imgs_t1_utrecht[:,:,:,20:36]
train_mask_ut=mask_utrecht[:,:,:,20:36]

train_img_am=imgs_amsterdam[:,:,:,45:61]
train_img_t1_am=imgs_t1_amsterdam[:,:,:,45:61]
train_mask_am=mask_amsterdam[:,:,:,45:61]

train_img_si=imgs_singapore[:,:,:,20:36]
train_img_t1_si=imgs_t1_singapore[:,:,:,20:36]
train_mask_si=mask_singapore[:,:,:,20:36]

#concate the arrays, and add one more dimension for Keras model
# train_img_set=np.expand_dims(np.concatenate((train_img_ut, train_img_am, train_img_si), axis=0), 4)
# train_img_set=np.expand_dims((train_img_ut), 4)
# train_img_set=np.expand_dims((train_img_am), 4)
train_img_set=np.expand_dims((train_img_si), 4)

train_img_t1_set=np.expand_dims(np.concatenate((train_img_t1_ut, train_img_t1_am, train_img_t1_si), axis=0), 4)

# train_mask_set=np.expand_dims(np.concatenate((train_mask_ut, train_mask_am, train_mask_si), axis=0), 4)
# train_mask_set=np.expand_dims((train_mask_ut), 4)
# train_mask_set=np.expand_dims((train_mask_am), 4)
train_mask_set=np.expand_dims((train_mask_si), 4)

# train_img_set=np.concatenate((train_img_ut, train_img_am, train_img_si), axis=0)
# train_mask_set=np.concatenate((train_mask_ut, train_mask_am, train_mask_si), axis=0)

print(train_img_set.shape)
print(train_mask_set.shape)

##### Always standardized the image arrays for neural networks #####
train_img_set-=np.mean(train_img_set)
train_img_set/=np.std(train_img_set)

train_img_t1_set = train_img_t1_set.astype('float64')
train_img_t1_set-=np.mean(train_img_t1_set)
train_img_t1_set/=np.std(train_img_t1_set)


##### separate original images from augemented to keep track 
train_img_orig=train_img_set[0:len(train_img_set):4]
train_img_rotate=train_img_set[1:len(train_img_set):4]
train_img_shear=train_img_set[2:len(train_img_set):4]
train_img_zoom=train_img_set[3:len(train_img_set):4]

train_img_t1_orig = train_img_t1_set[0:len(train_img_t1_set):4]
train_img_t1_rotate = train_img_t1_set[1:len(train_img_t1_set):4]
train_img_t1_shear = train_img_t1_set[2:len(train_img_t1_set):4]
train_img_t1_zoom = train_img_t1_set[3:len(train_img_t1_set):4]

train_mask_orig=train_mask_set[0:len(train_mask_set):4]
train_mask_rotate=train_mask_set[1:len(train_mask_set):4]
train_mask_shear=train_mask_set[2:len(train_mask_set):4]
train_mask_zoom=train_mask_set[3:len(train_mask_set):4]

# np.save(model_file + 'imgs_three_datasets_two_channels.npy',train_img_orig)
# np.save(model_file + 'imgs_mask_three_datasets_two_channels.npy',train_mask_orig)

###### lump augmented images together
img_aug=np.concatenate((train_img_rotate, train_img_shear, train_img_zoom), axis=0)
img_t1_aug=np.concatenate((train_img_t1_rotate, train_img_t1_shear, train_img_t1_zoom), axis=0)
mask_aug=np.concatenate((train_mask_rotate,train_mask_shear,train_mask_zoom ), axis=0)


#define loss function using Dice Coefficient
def dice_loss(y_true, y_pred):
    smooth=1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return -((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
#since the loss function will be minimized during model fitting, a negative sign is added.
#but the dice coefficient (positive) will be used later to evaluate test data

#calculate F1 score used in previous Keras version(https://github.com/fchollet/keras/commit/a56b1a55182acf061b1eb2e2c86b48193a0e88f7)
#F1 is weighted mean of the proportion of correct class assignments (precision) vs. the proportion of incorrect class assignments (recall)

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

    
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

################################### Build 3D model architecture (memory intensive!) ##############################
def wmh_unet():
    inputs = Input((ROWS, COLS,16,1))
    conv1 = Conv3D(32, (3, 3, 1), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(64, (3, 3, 1), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 1))(conv1)

    conv2 = Conv3D(64, (3, 3, 1), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(128, (3, 3, 1), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2)

    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv4)

    # expansive/synthesis path
    up5 = concatenate(
        [Conv3D(512, (3, 3, 3), activation='relu', padding='same')(UpSampling3D((2, 2, 2))(conv4)), conv3], axis=4)
    conv5 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up5)
    conv5 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate(
        [Conv3D(256, (3, 3, 1), activation='relu', padding='same')(UpSampling3D((2, 2, 1))(conv5)), conv2], axis=4)
    conv6 = Conv3D(128, (3, 3, 1), activation='relu', padding='same')(up6)
    conv6 = Conv3D(128, (3, 3, 1), activation='relu', padding='same')(conv6)

    up7 = concatenate(
        [Conv3D(128, (3, 3, 1), activation='relu', padding='same')(UpSampling3D((2, 2, 1))(conv6)), conv1], axis=4)
    conv7 = Conv3D(64, (3, 3, 1), activation='relu', padding='same')(up7)
    conv7 = Conv3D(64, (3, 3, 1), activation='relu', padding='same')(conv7)

    conv8 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv7)

    model = Model(inputs=[inputs], outputs=[conv8])

    return model

def wmh_unet_2():
    inputs = Input((ROWS, COLS,16,1))
    conv1 = Conv3D(32, (3, 3, 1), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(64, (3, 3, 1), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 1))(conv1)

    conv2 = Conv3D(64, (3, 3, 1), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(128, (3, 3, 1), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2)

    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv4)

    # expansive/synthesis path
    up5 = concatenate(
        [Conv3D(512, (3, 3, 3), activation='relu', padding='same')(UpSampling3D((2, 2, 2))(conv4)), conv3], axis=4)
    conv5 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up5)
    conv5 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate(
        [Conv3D(256, (3, 3, 1), activation='relu', padding='same')(UpSampling3D((2, 2, 1))(conv5)), conv2], axis=4)
    conv6 = Conv3D(128, (3, 3, 1), activation='relu', padding='same')(up6)
    conv6 = Conv3D(128, (3, 3, 1), activation='relu', padding='same')(conv6)

    up7 = concatenate(
        [Conv3D(128, (3, 3, 1), activation='relu', padding='same')(UpSampling3D((2, 2, 1))(conv6)), conv1], axis=4)
    conv7 = Conv3D(64, (3, 3, 1), activation='relu', padding='same')(up7)
    conv7 = Conv3D(64, (3, 3, 1), activation='relu', padding='same')(conv7)

    conv8 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv7)

    model = Model(inputs=[inputs], outputs=[conv8])

    return model

model=wmh_unet()
model.compile(optimizer=Adam(lr=2e-5), loss=dice_loss, metrics=[f1])

model.summary()


################ optional - load saved weights from previous training  ####################

model.load_weights(model_file + 'training_weights.h5') #change this to name of weight files

###########################################################################################

early_stopping =EarlyStopping(monitor='val_loss', patience=4)
model_checkpoint = ModelCheckpoint(model_file + 'training_weights.h5', monitor='val_loss', save_best_only=True,save_weights_only=True)


from sklearn.cross_validation import train_test_split
#majority of original images are used for validation. 
train_img1, val_img1, train_mask1, val_mask1 = train_test_split(
    train_img_orig, train_mask_orig, test_size=0.99, random_state=42)

train_img2, val_img2, train_mask2, val_mask2 = train_test_split(
    img_aug, mask_aug, test_size=0.15, random_state=42)

# train_img3, val_img3, train_mask3, val_mask3 = train_test_split(
#     train_img_t1_orig, train_mask_orig, test_size=0.9, random_state=42)
#
# train_img4, val_img4, train_mask4, val_mask4 = train_test_split(
#     img_t1_aug, mask_aug, test_size=0.15, random_state=42)


train_img_combined=np.concatenate((train_img1, train_img2), axis=0)
train_mask_combined=np.concatenate((train_mask1, train_mask2), axis=0)
val_img_combined=np.concatenate((val_img1, val_img2), axis=0)
val_mask_combined=np.concatenate((val_mask1, val_mask2), axis=0)

# train_img_t1_combined=np.concatenate((train_img3, train_img4), axis=0)
# train_mask_t1_combined=np.concatenate((train_mask3, train_mask4), axis=0)
# val_img_t1_combined=np.concatenate((val_img3, val_img4), axis=0)
# val_mask_t1_combined=np.concatenate((val_mask3, val_mask4), axis=0)

print(train_img_combined.shape)
print(val_img_combined.shape)


################### shuffle the dataset before feeding them to model
from sklearn.utils import shuffle
train_shuffled, train_mask_shuffled = shuffle(train_img_combined, train_mask_combined, random_state=12)
val_shuffled, val_mask_shuffled = shuffle(val_img_combined, val_mask_combined, random_state=12)

# train_shuffled, train_mask_shuffled = shuffle(train_img_t1_combined, train_mask_t1_combined, random_state=12)
# val_shuffled, val_mask_shuffled = shuffle(val_img_t1_combined, val_mask_t1_combined, random_state=12)


# hist=model.fit(train_shuffled, train_mask_shuffled , batch_size=1, epochs=50, verbose=1, shuffle=True,
#               validation_data=(val_shuffled, val_mask_shuffled),callbacks=[model_checkpoint,early_stopping])

################### Predict on validation set ##################
test_pred = model.predict(val_img1, batch_size=1, verbose=1)

test_masks=np.resize(test_pred, (test_pred.shape[0], test_pred.shape[1], test_pred.shape[2],test_pred.shape[3]))
true_masks=np.resize(val_mask1, (val_mask1.shape[0],val_mask1.shape[1],val_mask1.shape[2],val_mask1.shape[3]))
val_image=np.resize(val_img1, (val_img1.shape[0], val_img1.shape[1], val_img1.shape[2], val_img1.shape[3]))

# test_masks=np.resize(test_pred, (test_pred.shape[3], test_pred.shape[2], test_pred.shape[1],test_pred.shape[0]))
# true_masks=np.resize(val_mask1, (val_mask1.shape[3],val_mask1.shape[2],val_mask1.shape[1],val_mask1.shape[0]))
# val_image=np.resize(val_img1, (val_img1.shape[3], val_img1.shape[2], val_img1.shape[1], val_img1.shape[0]))


for i in range(np.shape(test_masks)[0]):
    if not os.path.exists(os.path.join(result_file,'Singapore', str(i))):
        os.mkdir(os.path.join(result_file,'Singapore', str(i)))

    filename_resultImage = os.path.join(result_file,'Singapore', str(i), 'result.nii.gz')
    filename_Images = os.path.join(result_file,'Singapore', str(i) , 'images.nii.gz')
    filename_Images_orig = os.path.join(result_file,'Singapore', str(i) , 'wmh.nii.gz')
    sitk.WriteImage(sitk.GetImageFromArray(test_masks[i, ...]), filename_resultImage)
    sitk.WriteImage(sitk.GetImageFromArray(true_masks[i, ...]), filename_Images_orig)
    sitk.WriteImage(sitk.GetImageFromArray(val_image[i, ...]), filename_Images)

########## calculate total Dice Coefficient as a measure of similarity between predicted mask and true mask
true_mask_f = true_masks.flatten()
test_masks_f = np.around(test_masks.flatten()) 
smooth=1
intersection = np.sum((true_mask_f) *(test_masks_f))
dice=((2. * intersection + smooth) / (np.sum((true_mask_f)) + np.sum((test_masks_f)) + smooth))
print(dice)


#####################  Extra Stuff: visualize activation maps ####################

def extract_map(model, layer_indexes, sample):
    activation_maps=[]
    for i in layer_indexes:
        get_feature = K.function([model.layers[0].input], [model.layers[i].output])
        features = get_feature([sample, 0])[0][:,:,:,:,0] #slice it to only look at one feature map along the slices (z-dimension)
        #print(features.shape)
        for maps in features:
                j=np.resize(maps, (maps.shape[0],maps.shape[1], maps.shape[2]))
                j_ave=np.average(j, axis=2) #takes average of the different activation features for a given layer
                resize_map=cv2.resize(j_ave, dsize=(128, 128))
                activation_maps.append(resize_map)
    return np.array(activation_maps)


#for the down path
sample_map = extract_map(model, [1,2,4,5,7,8,10], val_img1[1:2])
