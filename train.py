import numpy as np 
import matplotlib.pyplot as plt

from PIL import Image as im
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from keras.applications.vgg16 import VGG16
from keras import layers
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping

from utils import load_data, crop_imgs, save_new_images, preprocess_imgs, build_data_pipelines

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
        monitor='val_accuracy', 
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

    accuracy = accuracy_score(y_val[:len(predictions)], predictions)
    print('Val Accuracy = %.2f' % accuracy)

    confusion_mtx = confusion_matrix(y_val[:len(predictions)], predictions)

    model.save('../Models/AvaSure_VGG_model.h5')

if __name__ == "__main__":

    train()