import numpy as np
import cv2
import os

import SimpleITK as sitk
from keras.models import Model
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, UpSampling3D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model

K.set_image_dim_ordering('tf')
K.image_dim_ordering()

'''
-----------------------------------------------------------------------------------------------------
Establezco las carpetas donde se van a guardar los modelos y luego los resultados. Tambien establezco
los paremetros de las imagenes
-----------------------------------------------------------------------------------------------------
'''
model_file = '/media/fimcp/DATA/WMH/Models/'
result_file = '/media/fimcp/DATA/WMH/Results/'

ROWS = 128
COLS = 128
'''
-----------------------------------------------------------------------------------------------------
Se define la funcion de perdida usando el coeficiente de Dice, dado que es una funcion de perdida esta
se va a minimizar durante el ajuste del modelo, para esto un signo negativo se agrego al final. Pero
para la evaluacion se va a usar el coeficiente de dice positivo.
-----------------------------------------------------------------------------------------------------
'''


def dice_loss(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return -((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))


'''
-----------------------------------------------------------------------------------------------------
El valor F1 es la media con pesos en la proporcion de que las asignaciones de las clases sean correctas
contra la proporcion de las asignaciones de clases incorrectas 
-----------------------------------------------------------------------------------------------------
'''


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

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


'''
-----------------------------------------------------------------------------------------------------
Se hace la arquitectura 3D del modelo U-Net luego se lo compila usando los siguientes hiperparametros
perdida 2e-5, funcion de perdida -> coeficiente de dice, metricas -> coeficiente f1.
Luego se hace un resumen de toda la red, y un grafico con toda las especificaciones de la red
-----------------------------------------------------------------------------------------------------
'''


def unet_3D():
    inputs = Input((ROWS, COLS, 16, 1))
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


'''
-----------------------------------------------------------------------------------------------------
Para poder visualizar los cambios realizados a la imagen por la CNN se usa la funcion extract_map, de
tal manera que dada las capas y una imagen de prueba se puede guardar las imagenes generadas por cada
parte del proceso. 
-----------------------------------------------------------------------------------------------------
'''


def extract_map(model, layer_indexes, sample):
    activation_maps=[]
    for i in layer_indexes:
        get_feature = K.function([model.layers[0].input], [model.layers[i].output])

        # Se hace el slice solo para ver a un feature map a lo largo de los ejes del slicing (dimension z)
        features = get_feature([sample, 0])[0][:,:,:,:,0]
        for maps in features:
                j=np.resize(maps, (maps.shape[0],maps.shape[1], maps.shape[2]))
                # Se toma el promedio de diferentes caracteristicas de activacion para una capa dada
                j_ave=np.average(j, axis=2)
                resize_map=cv2.resize(j_ave, dsize=(128, 128))
                activation_maps.append(resize_map)
    return np.array(activation_maps)


'''
-----------------------------------------------------------------------------------------------------
En este metodo se tiene un parametro para imprimir los valores generados por las formas del aprendiza
je a realizar, y tambien si es que se van a guardar los archivos. Si se desea solo evaluar el parametro
train sea True, ahi se hace el fit del modelo, sino se omite y se hace solo el predict
-----------------------------------------------------------------------------------------------------
'''


def main_segmentation(verbose=True, save_files=True, train=False, activation_maps=False):
    '''
    -----------------------------------------------------------------------------------------------------
    Cargo los archivos con las imagenes preprocesadas y con las imagenes aumentadas, esto se hace para
    cada dataset y por cada tipo de imagen
    -----------------------------------------------------------------------------------------------------
    '''

    imgs_utrecht = np.load(model_file + 'utrecht_flair(200)aug.npy')
    imgs_t1_utrecht = np.load(model_file + 'utrecht_t1(200)aug.npy')
    mask_utrecht = np.load(model_file + 'utrecht_mask(200)aug.npy')

    imgs_amsterdam = np.load(model_file + 'amsterdam_flair(200)aug.npy')
    imgs_t1_amsterdam = np.load(model_file + 'amsterdam_t1(200)aug.npy')
    mask_amsterdam = np.load(model_file + 'amsterdam_mask(200)aug.npy')

    imgs_singapore = np.load(model_file + 'singapore_flair(200)aug.npy')
    imgs_t1_singapore = np.load(model_file + 'singapore_t1(200)aug.npy')
    mask_singapore = np.load(model_file + 'singapore_mask(200)aug.npy')

    if verbose:
        print(imgs_utrecht.shape)
        print(mask_utrecht.shape)
        print(imgs_amsterdam.shape)
        print(mask_amsterdam.shape)
        print(imgs_singapore.shape)
        print(mask_singapore.shape)

    '''
    -----------------------------------------------------------------------------------------------------
    Se realiza el slicing del arreglo en el eje z de tal manera que sea el mismo en todas las muestras, 
    sin embargo se escoje solo 16 slices del modelo para reducir el tiempo de aprendizaje
    =====================================================================================================
    Para cada uno de los dataset se tiene train_img_, train_img_t1, train_mask, lo cual depende de donde 
    se hayan adquirido los datos
    -----------------------------------------------------------------------------------------------------
    '''

    train_img_ut = imgs_utrecht[:, :, :, 20:36]
    train_img_t1_ut = imgs_t1_utrecht[:, :, :, 20:36]
    train_mask_ut = mask_utrecht[:, :, :, 20:36]

    train_img_am = imgs_amsterdam[:, :, :, 45:61]
    train_img_t1_am = imgs_t1_amsterdam[:, :, :, 45:61]
    train_mask_am = mask_amsterdam[:, :, :, 45:61]

    train_img_si = imgs_singapore[:, :, :, 20:36]
    train_img_t1_si = imgs_t1_singapore[:, :, :, 20:36]
    train_mask_si = mask_singapore[:, :, :, 20:36]

    '''
    -----------------------------------------------------------------------------------------------------
    Se concatenan los arreglos, y se les agrega una dimension mas para el modelo del Keras, todos los arre
    glos se concatenan por el eje 0, si se desean concatenar todos los conjuntos de datos se puede usar la
    funcion generalizada; como en el caso de la t1. Sino se puede usar como se muestra por centro de estu
    dio.
    -----------------------------------------------------------------------------------------------------
    '''

    # train_img_set=np.expand_dims(np.concatenate((train_img_ut, train_img_am, train_img_si), axis=0), 4)
    # train_img_set=np.expand_dims((train_img_ut), 4)
    # train_img_set=np.expand_dims((train_img_am), 4)
    train_img_set = np.expand_dims((train_img_si), 4)

    train_img_t1_set = np.expand_dims(np.concatenate((train_img_t1_ut, train_img_t1_am, train_img_t1_si), axis=0), 4)

    # train_mask_set=np.expand_dims(np.concatenate((train_mask_ut, train_mask_am, train_mask_si), axis=0), 4)
    # train_mask_set=np.expand_dims((train_mask_ut), 4)
    # train_mask_set=np.expand_dims((train_mask_am), 4)
    train_mask_set = np.expand_dims((train_mask_si), 4)

    # train_img_set=np.concatenate((train_img_ut, train_img_am, train_img_si), axis=0)
    # train_mask_set=np.concatenate((train_mask_ut, train_mask_am, train_mask_si), axis=0)

    if verbose:
        print(train_img_set.shape)
        print(train_mask_set.shape)

    '''
    -----------------------------------------------------------------------------------------------------
    Siempre que se usan arrays de imagenes se requiere estandarizar la muestra, es decir restar a cada va
    lor del arreglo la media de las imagenes y dividir la desviacion estandar para cada uno de los valores
    en el caso de la imagen t1, este contiene valores enteros. Por ende se debe hacer float para poder 
    estandarizar.  
    -----------------------------------------------------------------------------------------------------
    '''
    train_img_set -= np.mean(train_img_set)
    train_img_set /= np.std(train_img_set)

    train_img_t1_set = train_img_t1_set.astype('float64')
    train_img_t1_set -= np.mean(train_img_t1_set)
    train_img_t1_set /= np.std(train_img_t1_set)

    '''
    -----------------------------------------------------------------------------------------------------
    Se separa cada una de las transformaciones realizadas y tambien la imagen original, dado que esto se
    va a aleatorizar, esto permite tener un track adecuado de que mascara generada pertenece a que imagen.
    Este proceso se hace para la imagen FLAIR , T1 y la mascara.
    -----------------------------------------------------------------------------------------------------
    '''
    train_img_orig = train_img_set[0:len(train_img_set):4]
    train_img_rotate = train_img_set[1:len(train_img_set):4]
    train_img_shear = train_img_set[2:len(train_img_set):4]
    train_img_zoom = train_img_set[3:len(train_img_set):4]

    train_img_t1_orig = train_img_t1_set[0:len(train_img_t1_set):4]
    train_img_t1_rotate = train_img_t1_set[1:len(train_img_t1_set):4]
    train_img_t1_shear = train_img_t1_set[2:len(train_img_t1_set):4]
    train_img_t1_zoom = train_img_t1_set[3:len(train_img_t1_set):4]

    train_mask_orig = train_mask_set[0:len(train_mask_set):4]
    train_mask_rotate = train_mask_set[1:len(train_mask_set):4]
    train_mask_shear = train_mask_set[2:len(train_mask_set):4]
    train_mask_zoom = train_mask_set[3:len(train_mask_set):4]

    '''
    -----------------------------------------------------------------------------------------------------
    Se agrupan las imagenes aumentadas juntas de tal manera que luego pueden ser alimentadas al modelo de
    manera diferente.
    -----------------------------------------------------------------------------------------------------
    '''
    img_aug = np.concatenate((train_img_rotate, train_img_shear, train_img_zoom), axis=0)
    img_t1_aug = np.concatenate((train_img_t1_rotate, train_img_t1_shear, train_img_t1_zoom), axis=0)
    mask_aug = np.concatenate((train_mask_rotate, train_mask_shear, train_mask_zoom), axis=0)

    '''
    -----------------------------------------------------------------------------------------------------
    Se compila el modelo de U-Net 3D con hiperparametros con funcion de perdida dice_loss y metricas con
    coeficiente f1
    -----------------------------------------------------------------------------------------------------
    '''

    model = unet_3D()
    model.compile(optimizer=Adam(lr=2e-5), loss=dice_loss, metrics=[f1])

    model.summary()
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=False)

    '''
    -----------------------------------------------------------------------------------------------------
    Se cargan los pesos de la red previamente entrenada, por cada nueva sesion de aprendizaje se carga el
    aprendizaje previo. Asi mismo el modelo va a detenerse si no encuentra nuevos pesos para los arcos,
    con un grado de tolerancia, cada vez que se hace el checkpoint del modelo se guardan solo los mejores
    valores y solo los pesos de los arcos
    -----------------------------------------------------------------------------------------------------
    '''

    model.load_weights(model_file + 'training_weights.h5')  # change this to name of weight files

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model_checkpoint = ModelCheckpoint(model_file + 'training_weights.h5',
                                       monitor='val_loss',
                                       save_best_only=True,
                                       save_weights_only=True)

    '''
    -----------------------------------------------------------------------------------------------------
    Se divide el modelo en imagenes para entrenamiento e imagenes para prueba, se usa casi todo el dataset
    y se deja solo 1 sujeto afuera 
    -----------------------------------------------------------------------------------------------------
    '''

    from sklearn.cross_validation import train_test_split

    train_img1, val_img1, train_mask1, val_mask1 = train_test_split(
        train_img_orig, train_mask_orig, test_size=0.99, random_state=42)

    train_img2, val_img2, train_mask2, val_mask2 = train_test_split(
        img_aug, mask_aug, test_size=0.15, random_state=42)

    # train_img3, val_img3, train_mask3, val_mask3 = train_test_split(
    #     train_img_t1_orig, train_mask_orig, test_size=0.9, random_state=42)
    #
    # train_img4, val_img4, train_mask4, val_mask4 = train_test_split(
    #     img_t1_aug, mask_aug, test_size=0.15, random_state=42)

    train_img_combined = np.concatenate((train_img1, train_img2), axis=0)
    train_mask_combined = np.concatenate((train_mask1, train_mask2), axis=0)
    val_img_combined = np.concatenate((val_img1, val_img2), axis=0)
    val_mask_combined = np.concatenate((val_mask1, val_mask2), axis=0)

    # train_img_t1_combined=np.concatenate((train_img3, train_img4), axis=0)
    # train_mask_t1_combined=np.concatenate((train_mask3, train_mask4), axis=0)
    # val_img_t1_combined=np.concatenate((val_img3, val_img4), axis=0)
    # val_mask_t1_combined=np.concatenate((val_mask3, val_mask4), axis=0)

    if verbose:
        print(train_img_combined.shape)
        print(val_img_combined.shape)

    '''
    -----------------------------------------------------------------------------------------------------
    Se debe barajar o aleatorizar el dataset antes de alimentar la red neuronal, esto se hace para cada 
    uno de los datasets incluida la mascar
    -----------------------------------------------------------------------------------------------------
    '''

    from sklearn.utils import shuffle
    train_shuffled, train_mask_shuffled = shuffle(train_img_combined, train_mask_combined, random_state=12)
    val_shuffled, val_mask_shuffled = shuffle(val_img_combined, val_mask_combined, random_state=12)

    # train_shuffled, train_mask_shuffled = shuffle(train_img_t1_combined, train_mask_t1_combined, random_state=12)
    # val_shuffled, val_mask_shuffled = shuffle(val_img_t1_combined, val_mask_t1_combined, random_state=12)

    '''
    -----------------------------------------------------------------------------------------------------
    Luego se hace el fitting del modelo, se uso un batch size de 1, 50 epochs por iteracion, las imagenes
    de validacion junto con sus mascaras y los parametros para hacer model_checkpoint y early_stopping
    -----------------------------------------------------------------------------------------------------
    '''
    if train:
        hist = model.fit(train_shuffled,
                         train_mask_shuffled,
                         batch_size=1,
                         epochs=50,
                         verbose=1,
                         shuffle=True,
                         validation_data=(val_shuffled, val_mask_shuffled),
                         callbacks=[model_checkpoint,early_stopping])

    '''
    -----------------------------------------------------------------------------------------------------
    Para poder hacer la prediccion se usa la imagen o imagenes a validar y el modelo ya entrenado
    -----------------------------------------------------------------------------------------------------
    '''

    test_pred = model.predict(val_img1, batch_size=1, verbose=1)

    '''
    -----------------------------------------------------------------------------------------------------
    Con la mascara predicha se debe hacer un resize para que tenga las mismas dimensiones del conjunto de
    entrada, tambien para reestructurar la imagen al ser mostrada
    -----------------------------------------------------------------------------------------------------
    '''
    test_masks = np.resize(test_pred, (test_pred.shape[0], test_pred.shape[1], test_pred.shape[2], test_pred.shape[3]))
    true_masks = np.resize(val_mask1, (val_mask1.shape[0], val_mask1.shape[1], val_mask1.shape[2], val_mask1.shape[3]))
    val_image = np.resize(val_img1, (val_img1.shape[0], val_img1.shape[1], val_img1.shape[2], val_img1.shape[3]))

    # test_masks=np.resize(test_pred, (test_pred.shape[3], test_pred.shape[2], test_pred.shape[1],test_pred.shape[0]))
    # true_masks=np.resize(val_mask1, (val_mask1.shape[3],val_mask1.shape[2],val_mask1.shape[1],val_mask1.shape[0]))
    # val_image=np.resize(val_img1, (val_img1.shape[3], val_img1.shape[2], val_img1.shape[1], val_img1.shape[0]))

    '''
    -----------------------------------------------------------------------------------------------------
    Luego se guarda en una carpeta, la cual puede estar por centro de muestra o en una carpeta general,
    para esto se hace un transpose con el fin de que la imagen generada tega el mismo sistema de referencia
    que las imagenes originales
    -----------------------------------------------------------------------------------------------------
    '''
    if save_files:
        for i in range(np.shape(test_masks)[0]):
            if not os.path.exists(os.path.join(result_file, 'Singapore', str(i))):
                os.mkdir(os.path.join(result_file, 'Singapore', str(i)))

            filename_resultImage = os.path.join(result_file, 'Singapore', str(i), 'result.nii.gz')
            filename_Images = os.path.join(result_file, 'Singapore', str(i), 'images.nii.gz')
            filename_Images_orig = os.path.join(result_file, 'Singapore', str(i), 'wmh.nii.gz')
            sitk.WriteImage(sitk.GetImageFromArray(np.transpose(test_masks[i, ...])), filename_resultImage)
            sitk.WriteImage(sitk.GetImageFromArray(np.transpose(true_masks[i, ...])), filename_Images_orig)
            sitk.WriteImage(sitk.GetImageFromArray(np.transpose(val_image[i, ...])), filename_Images)

    '''
    -----------------------------------------------------------------------------------------------------
    Las imagenes de ground truth usadas para el entrenamiento se les hace un flat de tal manera que se 
    pueda encontrar el coeficiente DICE con las imagenes generadas por el modelo. 
    -----------------------------------------------------------------------------------------------------
    '''
    true_mask_f = true_masks.flatten()
    test_masks_f = np.around(test_masks.flatten())
    smooth = 1
    intersection = np.sum((true_mask_f) * (test_masks_f))
    dice = ((2. * intersection + smooth) / (np.sum((true_mask_f)) + np.sum((test_masks_f)) + smooth))

    print("Coeficiente Dice: " + str(dice))

    '''
    -----------------------------------------------------------------------------------------------------
    Para la visualizacion de las capas de aprendizaje hacia abajo se usan las capas 1,2,4,5,7,8,10 estas
    se guardan en un modelo para poder ser luego visualizadas
    -----------------------------------------------------------------------------------------------------
    '''
    if activation_maps:
        activation_map_down = extract_map(model, [1, 2, 4, 5, 7, 8, 10], val_img1[1:2])
        np.save(model_file + "activation_maps_down.npy", activation_map_down)
        activation_map_up = extract_map(model, [12, 13, 15, 17, 18, 20, 25, 27], val_img1[1:2])
        np.save(model_file + "activation_maps_up.npy", activation_map_up)


if "__main__" == __name__:
    main_segmentation(activation_maps=True)
