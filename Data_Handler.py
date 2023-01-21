import os
import shutil

def split_data_into_class_folders(IMG_PATH):

    for CLASS in os.listdir(IMG_PATH):
        if not CLASS.startswith('.'):
            IMG_NUM = len(os.listdir(IMG_PATH + CLASS))
            for (n, FILE_NAME) in enumerate(os.listdir(IMG_PATH + CLASS)):
                img = IMG_PATH + CLASS + '/' + FILE_NAME
                if n < 5:
                    shutil.copy(img, 'Data/TEST/' + CLASS.upper() + '/' + FILE_NAME)
                elif n < 0.8*IMG_NUM:
                    shutil.copy(img, 'Data/TRAIN/'+ CLASS.upper() + '/' + FILE_NAME)
                else:
                    shutil.copy(img, 'Data/VAL/'+ CLASS.upper() + '/' + FILE_NAME)

if __name__ == "__main__":
    split_data_switch = False

    path_to_train_data = '../Tumor_Detection/Raw_Data/'

    if split_data_switch :
        split_data_into_class_folders(path_to_train_data)
