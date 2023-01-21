import numpy as np 
import cv2
import os
import shutil
import itertools
import imutils
import matplotlib.pyplot as plt

from PIL import Image as im
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping

from utils import load_data

RANDOM_SEED = 123

def train():
    TRAIN_DIR = 'Data/TRAIN/'
    TEST_DIR = 'Data/TEST/'
    VAL_DIR = 'Data/VAL/'
    IMG_SIZE = (224,224)

    X_train, y_train, labels = load_data(TRAIN_DIR, IMG_SIZE)
    X_test, y_test, _ = load_data(TEST_DIR, IMG_SIZE)
    X_val, y_val, _ = load_data(VAL_DIR, IMG_SIZE)

    X_train_crop = crop_imgs(set_name=X_train)
    X_test_crop =crop_imgs(set_name=X_val)
    X_val_crop = crop_imgs(set_name=X_test)

    save_croped = False

    if save_croped:
        save_new_images(X_train_crop, y_train, folder_name='Data/TRAIN_CROP/')
        save_new_images(X_test_crop, y_val, folder_name='Data/VAL_CROP/')
        save_new_images(X_val_crop, y_test, folder_name='Data/TEST_CROP/')

    X_train_prep = preprocess_imgs(set_name=X_train_crop, img_size=IMG_SIZE)
    X_test_prep = preprocess_imgs(set_name=X_test_crop, img_size=IMG_SIZE)
    X_val_prep = preprocess_imgs(set_name=X_val_crop, img_size=IMG_SIZE)

    train_generator, val_generator = build_data_pipelines(IMG_SIZE)

    # load base model
    vgg16_weight_path = 'Weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    base_model = VGG16(
        weights=vgg16_weight_path,
        include_top=False, 
        input_shape=IMG_SIZE + (3,)
    )

    NUM_CLASSES = 1

    model = Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(NUM_CLASSES, activation='sigmoid'))

    model.layers[0].trainable = False

    model.compile(
        loss='binary_crossentropy',
        optimizer=RMSprop(lr=1e-4),
        metrics=['accuracy']
    )

    model.summary()

    EPOCHS = 30
    es = EarlyStopping(
        monitor='val_acc', 
        mode='max',
        patience=6
    )

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=50,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=25,
        callbacks=[es]
    )

    predictions = model.predict(X_val_prep)
    predictions = [1 if x>0.5 else 0 for x in predictions]

    accuracy = accuracy_score(y_val, predictions)
    print('Val Accuracy = %.2f' % accuracy)

    confusion_mtx = confusion_matrix(y_val, predictions)

def crop_imgs(set_name, add_pixels_value=0):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    set_new = []
    NoneType = type(None)

    for img in set_name:
        if type(img) != NoneType:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # threshold the image, then perform a series of erosions +
            # dilations to remove any small regions of noise
            thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.erode(thresh, None, iterations=2)
            thresh = cv2.dilate(thresh, None, iterations=2)

            # find contours in thresholded image, then grab the largest one
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)

            # find the extreme points
            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])

            ADD_PIXELS = add_pixels_value
            new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
            set_new.append(new_img)

    return np.array(set_new)

def save_new_images(x_set, y_set, folder_name):
    i = 0
    for (img, imclass) in zip(x_set, y_set):
        if imclass == 0:
            cv2.imwrite(folder_name+'NO/'+str(i)+'.jpg', img)
        else:
            cv2.imwrite(folder_name+'YES/'+str(i)+'.jpg', img)
        i += 1

def preprocess_imgs(set_name, img_size):
    """
    Resize and apply VGG-15 preprocessing
    """
    set_new = []
    for img in set_name:
        img = cv2.resize(
            img,
            dsize=img_size,
            interpolation=cv2.INTER_CUBIC
        )
        set_new.append(preprocess_input(img))
    return np.array(set_new)

def build_data_pipelines(IMG_SIZE):

    TRAIN_DIR = 'Data/TRAIN_CROP/'
    VAL_DIR = 'Data/VAL_CROP/'

    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        brightness_range=[0.5, 1.5],
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=preprocess_input
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        color_mode='rgb',
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode='binary',
        seed=RANDOM_SEED
    )

    val_generator = test_datagen.flow_from_directory(
        VAL_DIR,
        color_mode='rgb',
        target_size=IMG_SIZE,
        batch_size=16,
        class_mode='binary',
        seed=RANDOM_SEED
    )

    return train_generator, val_generator

if __name__ == "__main__":

    train()