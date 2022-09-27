
import os
import pandas as pd
import tensorflow as tf

os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\bin')

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications import Xception

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator

from Dog_Cat_Classification_5_CNN_Models import configuration_models_folder

def Testing_CNN_Models_Biclass_From_Folder(Models, Folder, Technique):

    # * Parameters
    Labels_biclass = ['Abnormal', 'Normal']
    #Labels_triclass = ['Normal', 'Benign', 'Malignant']
    X_size = 224
    Y_size = 224
    Epochs = 4

    #Name_dir = os.path.dirname(Folder)
    #Name_base = os.path.basename(Folder)

    batch_size = 32

    Shape = (X_size, Y_size)

    Name_folder_training = Folder + '/' + 'train'
    Name_folder_val = Folder + '/' + 'val'
    Name_folder_test = Folder + '/' + 'test'

    train_datagen = ImageDataGenerator()
    val_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        directory = Name_folder_training,
        target_size = Shape,
        color_mode = "rgb",
        batch_size = batch_size,
        class_mode = "binary",
        shuffle = True,
        seed = 42
    )

    valid_generator = val_datagen.flow_from_directory(
        directory = Name_folder_val,
        target_size = Shape,
        color_mode = "rgb",
        batch_size = batch_size,
        class_mode = "binary",
        shuffle = False,
        seed = 42        
    )

    test_generator = test_datagen.flow_from_directory(
        directory = Name_folder_test,
        target_size = Shape,
        color_mode = "rgb",
        batch_size = batch_size,
        class_mode = "binary",
        shuffle = False,
        seed = 42
    )
    
    #print(Dataframe_save_mias_folder)
    #Training_data, Validation_data, Test_data, Dataframe_save, Folder_path, DL_model, Enhancement_technique, Class_labels, Column_names, X_size, Y_size, Epochs, Folder_CSV, Folder_models, Folder_models_esp
    Info_dataframe = configuration_models_folder(trainingdata = train_generator, validationdata = valid_generator, testdata = test_generator, foldermodels = 'D:\Test',
                                                    foldermodelesp = 'D:\Test', foldercsv = 'D:\Test', models = Models, technique = Technique, labels = Labels_biclass,
                                                        X = X_size, Y = Y_size, epochs = Epochs)

    return Info_dataframe
