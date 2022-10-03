
import os
import numpy as np
import pandas as pd

from Dog_Cat_Classification_2_Data_Augmentation import DataAugmentation

def preprocessing_DataAugmentation_Biclass_ML(Folder_mass, Folder_calcification, Folder_destination):

    # * List to add images and labels.
    Images = []
    Labels = []

    # * General parameters
    #Iter_Mass = 20 
    #Iter_tumor = 40 

    #Iter_mass = 5 
    #Iter_calcification = 4 

    Iter_mass = 10 
    Iter_calcification = 9  

    Label_mass = 'Mass' 
    Label_calcification = 'Calcification'  

    Mass_images_class = 0 
    Calcification_images_class = 1 

    # * With this class we use the technique called data augmentation to create new images with their transformations
    Data_augmentation_mass = DataAugmentation(Folder = Folder_mass, NewFolder = Folder_destination, Severity = Label_mass, Sampling = Iter_mass, Label = Mass_images_class, Saveimages = False)
    Data_augmentation_calcification = DataAugmentation(Folder = Folder_calcification, NewFolder = Folder_destination, Severity = Label_calcification, Sampling = Iter_calcification, Label = Calcification_images_class, Saveimages = False)

    Images_mass, Labels_mass = Data_augmentation_mass.no_data_augmentation()
    Images_calcification, Labels_calcification = Data_augmentation_calcification.no_data_augmentation()

    # * Add the value in the lists already created

    Images.append(Images_mass)
    Images.append(Images_calcification)

    Labels.append(Labels_mass)
    Labels.append(Labels_calcification)

    print(len(Images_mass))
    print(len(Images_calcification))

    return Images, Labels

def preprocessing_DataAugmentation_Biclass_CNN(Folder_mass, Folder_calcification, Folder_destination):

    # * List to add images and labels.
    #Images = []
    #Labels = []

    # * General parameters
    #Iter_normal = 20 
    #Iter_Clasification = 40 

    #Iter_normal = 18 
    #Iter_Clasification = 34

    Iter_mass = 10 
    Iter_calcification = 10  

    Label_mass = 'Mass' 
    Label_calcification = 'Calcification'  

    Mass_images_class = 0 
    Clasification_images_class = 1 

    # * With this class we use the technique called data augmentation to create new images with their transformations
    Data_augmentation_mass = DataAugmentation(Folder = Folder_mass, NewFolder = Folder_destination, Severity = Label_mass, Sampling = Iter_mass, Label = Mass_images_class, Saveimages = True)
    Data_augmentation_calcification = DataAugmentation(Folder = Folder_calcification, NewFolder = Folder_destination, Severity = Label_calcification, Sampling = Iter_calcification, Label = Clasification_images_class, Saveimages = True)

    Images_mass, Labels_mass = Data_augmentation_mass.data_augmentation_test_images()
    Images_calcification, Labels_calcification = Data_augmentation_calcification.data_augmentation_test_images()

    # * Add the value in the lists already created

    Images_total = Images_mass + Images_calcification
    Labels_total = np.concatenate((Labels_mass, Labels_calcification), axis = None)

    print(Images_mass)
    print(Images_calcification)

    #print(len(Images_mass))
    #print(len(Images_calcification))

    return Images_total, Labels_total

# ? Data augmentation from folder, Splitting data required.

def preprocessing_DataAugmentation_Biclass_Folder(Folder_path, First_label, Second_label, First_number_iter, Second_number_iter):

    # * List to add images and labels.
    #Images = []
    #Labels = []

    # * General parameters
    #First_number_iter = 20 
    #Second_number_iter = 40 

    #First_number_iter = 18 #*
    #Second_number_iter = 34  #* 

    Total_files:int = 0
    Total_dir:int = 0

    #First_number_iter = 1 
    #Second_number_iter = 1  

    First_images_class:int = 0 
    Second_images_class:int = 1 

    Dir_total_training = []
    Dir_total_val = []
    Dir_total_test = []

    Folder_path_train_classes = []
    Folder_path_val_classes = []
    Folder_path_test_classes = []

    Folder_path_train ='{}/train/'.format(Folder_path)
    Folder_path_val ='{}/val/'.format(Folder_path)
    Folder_path_test ='{}/test/'.format(Folder_path)

    for Base, Dirs, Files in os.walk(Folder_path_train):
        print('Searching in : ', Base)
        for Dir in Dirs:
            Dir_total_training.append(Dir)
            Total_dir += 1
        for Index, File in enumerate(Files):
            Total_files += 1
    """

    for base, dirs, files in os.walk(Folder_path_test):
        print('Searching in : ', base)
        for dir in dirs:
            Dir_total_test.append(dir)
            Total_dir += 1
        for file in files:
            Total_files += 1

    
    for base, dirs, files in os.walk(Folder_path_val):
        print('Searching in : ', base)
        for dir in dirs:
            Dir_total_val.append(dir)
            Total_dir += 1
        for file in files:
            Total_files += 1
    """

    #print(Dir_total[0])
    #print(Dir_total[1])
    #print(len(Dir_total))
    #print(Total_dir)
    #print(Total_files)

    for Index, dir in enumerate(Dir_total_training):
        print(Index)
        Folder_path_train_classes.append('{}{}'.format(Folder_path_train, dir))
        print(Folder_path_train_classes[Index])
    """
    for Index, dir in enumerate(Dir_total_test):
        print(Index)
        Folder_path_test_classes.append('{}{}'.format(Folder_path_test, dir))
        print(Folder_path_test_classes[Index])

    
    for Index, dir in enumerate(Dir_total_val):
        print(Index)
        Folder_path_val_classes.append('{}{}'.format(Folder_path_val, dir))
        print(Folder_path_val_classes[Index])
    """
    # * With this class we use the technique called data augmentation to create new images with their transformations

    Data_augmentation_normal = DataAugmentation(Folder = Folder_path_train_classes[0], NewFolder = Folder_path_train_classes[0], Severity = First_label, Sampling = First_number_iter, Label = First_images_class, Saveimages = True)
    Data_augmentation_tumor = DataAugmentation(Folder = Folder_path_train_classes[1], NewFolder = Folder_path_train_classes[1], Severity = Second_label, Sampling = Second_number_iter, Label = Second_images_class, Saveimages = True)

    Images_Normal, Labels_Normal = Data_augmentation_normal.data_augmentation_same_folder()
    Images_Tumor, Labels_Tumor = Data_augmentation_tumor.data_augmentation_same_folder()

    # * Add the value in the lists already created

    Images_total = Images_Normal + Images_Tumor
    Labels_total = np.concatenate((Labels_Normal, Labels_Tumor), axis = None)

    print(len(Images_Normal))
    print(len(Images_Tumor))

    #print(len(Images_total))
    #print(len(Labels_total))

    """
    Data_augmentation_normal = DataAugmentation(Folder = Folder_path_test_classes[0], NewFolder = Folder_path_test_classes[0], Severity = Label_normal, Sampling = Iter_normal, Label = Normal_images_class, Saveimages = True)
    Data_augmentation_tumor = DataAugmentation(Folder = Folder_path_test_classes[1], NewFolder = Folder_path_test_classes[1], Severity = Label_tumor, Sampling = Iter_tumor, Label = Tumor_images_class, Saveimages = True)

    Images_Normal, Labels_Normal = Data_augmentation_normal.data_augmentation_same_folder()
    Images_Tumor, Labels_Tumor = Data_augmentation_tumor.data_augmentation_same_folder()

    # * Add the value in the lists already created

    Images_total = Images_Normal + Images_Tumor
    Labels_total = np.concatenate((Labels_Normal, Labels_Tumor), axis = None)

    print(len(Images_Normal))
    print(len(Images_Tumor))


    Data_augmentation_normal = DataAugmentation(Folder = Folder_path_val_classes[0], NewFolder = Folder_path_val_classes[0], Severity = First_label, Sampling = First_number_iter, Label = First_images_class, Saveimages = True)
    Data_augmentation_tumor = DataAugmentation(Folder = Folder_path_val_classes[1], NewFolder = Folder_path_val_classes[1], Severity = Second_label, Sampling = Second_number_iter, Label = Second_images_class, Saveimages = True)

    Images_Normal, Labels_Normal = Data_augmentation_normal.data_augmentation_test_images()
    Images_Tumor, Labels_Tumor = Data_augmentation_tumor.data_augmentation_test_images()

    # * Add the value in the lists already created

    Images_total = Images_Normal + Images_Tumor
    Labels_total = np.concatenate((Labels_Normal, Labels_Tumor), axis = None)

    print(len(Images_total))
    print(len(Labels_total))
    """

# ? Data augmentation from folder, Splitting data required.

def preprocessing_DataAugmentation_Biclass_Folder(Dataframe: pd.DataFrame) ->:

    # * List to add images and labels.
    #Images = []
    #Labels = []

    # * General parameters
    #First_number_iter = 20 
    #Second_number_iter = 40 

    #First_number_iter = 18 #*
    #Second_number_iter = 34  #* 

    Total_files:int = 0
    Total_dir:int = 0

    #First_number_iter = 1 
    #Second_number_iter = 1  

    First_images_class:int = 0 
    Second_images_class:int = 1 

    Dir_total_training = []
    Dir_total_val = []
    Dir_total_test = []

    Folder_path_train_classes = []
    Folder_path_val_classes = []
    Folder_path_test_classes = []

    Folder_path_train ='{}/train/'.format(Folder_path)
    Folder_path_val ='{}/val/'.format(Folder_path)
    Folder_path_test ='{}/test/'.format(Folder_path)

    for Base, Dirs, Files in os.walk(Folder_path_train):
        print('Searching in : ', Base)
        for Dir in Dirs:
            Dir_total_training.append(Dir)
            Total_dir += 1
        for Index, File in enumerate(Files):
            Total_files += 1
    """

    for base, dirs, files in os.walk(Folder_path_test):
        print('Searching in : ', base)
        for dir in dirs:
            Dir_total_test.append(dir)
            Total_dir += 1
        for file in files:
            Total_files += 1

    
    for base, dirs, files in os.walk(Folder_path_val):
        print('Searching in : ', base)
        for dir in dirs:
            Dir_total_val.append(dir)
            Total_dir += 1
        for file in files:
            Total_files += 1
    """

    #print(Dir_total[0])
    #print(Dir_total[1])
    #print(len(Dir_total))
    #print(Total_dir)
    #print(Total_files)2

    for Index, dir in enumerate(Dir_total_training):
        print(Index)
        Folder_path_train_classes.append('{}{}'.format(Folder_path_train, dir))
        print(Folder_path_train_classes[Index])
    """
    for Index, dir in enumerate(Dir_total_test):
        print(Index)
        Folder_path_test_classes.append('{}{}'.format(Folder_path_test, dir))
        print(Folder_path_test_classes[Index])

    
    for Index, dir in enumerate(Dir_total_val):
        print(Index)
        Folder_path_val_classes.append('{}{}'.format(Folder_path_val, dir))
        print(Folder_path_val_classes[Index])
    """
    # * With this class we use the technique called data augmentation to create new images with their transformations

    Data_augmentation_normal = DataAugmentation(Folder = Folder_path_train_classes[0], NewFolder = Folder_path_train_classes[0], Severity = First_label, Sampling = First_number_iter, Label = First_images_class, Saveimages = True)
    Data_augmentation_tumor = DataAugmentation(Folder = Folder_path_train_classes[1], NewFolder = Folder_path_train_classes[1], Severity = Second_label, Sampling = Second_number_iter, Label = Second_images_class, Saveimages = True)

    Images_Normal, Labels_Normal = Data_augmentation_normal.data_augmentation_same_folder()
    Images_Tumor, Labels_Tumor = Data_augmentation_tumor.data_augmentation_same_folder()

    # * Add the value in the lists already created

    Images_total = Images_Normal + Images_Tumor
    Labels_total = np.concatenate((Labels_Normal, Labels_Tumor), axis = None)

    print(len(Images_Normal))
    print(len(Images_Tumor))

    #print(len(Images_total))
    #print(len(Labels_total))

    """
    Data_augmentation_normal = DataAugmentation(Folder = Folder_path_test_classes[0], NewFolder = Folder_path_test_classes[0], Severity = Label_normal, Sampling = Iter_normal, Label = Normal_images_class, Saveimages = True)
    Data_augmentation_tumor = DataAugmentation(Folder = Folder_path_test_classes[1], NewFolder = Folder_path_test_classes[1], Severity = Label_tumor, Sampling = Iter_tumor, Label = Tumor_images_class, Saveimages = True)

    Images_Normal, Labels_Normal = Data_augmentation_normal.data_augmentation_same_folder()
    Images_Tumor, Labels_Tumor = Data_augmentation_tumor.data_augmentation_same_folder()

    # * Add the value in the lists already created

    Images_total = Images_Normal + Images_Tumor
    Labels_total = np.concatenate((Labels_Normal, Labels_Tumor), axis = None)

    print(len(Images_Normal))
    print(len(Images_Tumor))


    Data_augmentation_normal = DataAugmentation(Folder = Folder_path_val_classes[0], NewFolder = Folder_path_val_classes[0], Severity = First_label, Sampling = First_number_iter, Label = First_images_class, Saveimages = True)
    Data_augmentation_tumor = DataAugmentation(Folder = Folder_path_val_classes[1], NewFolder = Folder_path_val_classes[1], Severity = Second_label, Sampling = Second_number_iter, Label = Second_images_class, Saveimages = True)

    Images_Normal, Labels_Normal = Data_augmentation_normal.data_augmentation_test_images()
    Images_Tumor, Labels_Tumor = Data_augmentation_tumor.data_augmentation_test_images()

    # * Add the value in the lists already created

    Images_total = Images_Normal + Images_Tumor
    Labels_total = np.concatenate((Labels_Normal, Labels_Tumor), axis = None)

    print(len(Images_total))
    print(len(Labels_total))
    """