
########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from configparser import Interpolation
import pandas as pd

from Dog_Cat_Classification_1_General_Functions import concat_dataframe
from Dog_Cat_Classification_4_Image_Processing import ImageProcessing

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

def preprocessing_technique_Biclass(CSV_path: str) -> pd.DataFrame:

    with open(CSV_path) as CSV:

        Data = pd.read_csv(CSV)

        # *
        Folder = Data['Folder'].tolist()
        New_folder = Data['New Folder'].tolist()
        Animal = Data['Animal'].tolist()
        Label = Data['Label'].tolist()

        # *
        Interpolation = Data['Interpolation'].tolist()
        X_size = Data['X'].tolist()
        Y_size = Data['Y'].tolist()

        # *
        Division = Data['Division'].tolist()

        # *
        Clip_limit = Data['Clip Limit'].tolist()
        Title_grid_size_X = Data['TitleGridSize X'].tolist()
        Title_grid_size_Y = Data['TitleGridSize Y'].tolist()
        
        Title_grid_size = (Title_grid_size_X, Title_grid_size_Y)

        # *
        Radius = Data['Radius'].tolist()
        Amount = Data['Amount'].tolist()

        # *
        Technique = Data['Technique'].tolist()
        
        # *
        CSV_path_folder = Data['Folder CSV'].tolist()

        # *
        Rows = len(Folder)
        
    # * 
    Object_IP = []

    # * 
    Dataframes = [None] * len(Folder)

    # * Class problem definition
    Class_problem = len(Folder)

    if Class_problem == 2:
        Class_problem_prefix = 'Biclass'
    elif Class_problem > 2:
        Class_problem_prefix = 'Multiclass'

    # * Image processing class

    for i in range(Rows):
        Object_IP(ImageProcessing(Folder = Folder[i], Newfolder = New_folder[i], animal = Animal[i], label = Label, I = Interpolation, X = X_size, Y = Y_size,
                                    cliplimit = Clip_limit, tileGridSize = Title_grid_size, division = Division, radius = Radius, amount = Amount))


    # * Choose the technique utilized for the test
    if Technique == 'NO':

        for i in range(len(Object_IP)):
            Dataframes[i] = Object_IP[i].normalize_technique()

    elif Technique == 'CLAHE':

        for i in range(len(Object_IP)):
            Dataframes[i] = Object_IP[i].normalize_technique()

    elif Technique == 'HE':

        for i in range(len(Object_IP)):
            Dataframes[i] = Object_IP[i].normalize_technique()

    elif Technique == 'UM':

        for i in range(len(Object_IP)):
            Dataframes[i] = Object_IP[i].normalize_technique()

    elif Technique == 'CS':

        for i in range(len(Object_IP)):
            Dataframes[i] = Object_IP[i].normalize_technique()
    
    elif Technique == 'MF':
        
        for i in range(len(Object_IP)):
            Dataframes[i] = Object_IP[i].normalize_technique()

    else:
        raise ValueError("Choose a new technique")      #! Alert

    # * Concatenate dataframes with this function
    concat_dataframe(DataFrame_Normal, DataFrame_Tumor, folder = CSV_path_folder, classp = Class_problem_prefix, technique = Technique, savefile = True)

def preprocessing_technique_Multiclass(New_technique, Folder_normal, Folder_benign, Folder_malignant, New_folder_normal, New_folder_benign, New_folder_malignant):

    # * Parameters for normalization

    # * Labels
    Label_Normal = 'Normal'   # Normal label 
    Label_Benign = 'Benign'   # Benign label
    Label_Malignant = 'Malignant' # Malignant label

    Cliplimit = 0.01
    Division = 3
    Radius = 2
    Amount = 1

    # * Classes
    Normal_images_class = 0 # Normal class
    Benign_images_class = 1 # Tumor class
    Malignant_images_class = 2 # Tumor class

    # * Problem class
    Multiclass = 'Multiclass' # Multiclass label

    Normalization_Normal = ImageProcessing(Folder = Folder_normal, Newfolder = New_folder_normal, Severity = Label_Normal, Label = Normal_images_class,
                                            cliplimit = Cliplimit, division = Division, radius = Radius, amount = Amount)
    Normalization_Benign = ImageProcessing(Folder = Folder_benign, Newfolder = New_folder_benign, Severity = Label_Benign, Label = Benign_images_class,
                                            cliplimit = Cliplimit, division = Division, radius = Radius, amount = Amount)
    Normalization_Malignant = ImageProcessing(Folder = Folder_malignant, Newfolder = New_folder_malignant, Severity = Label_Malignant, Label = Malignant_images_class,
                                            cliplimit = Cliplimit, division = Division, radius = Radius, amount = Amount)

    if New_technique == 'NO':
        DataFrame_Normal = Normalization_Normal.normalize_technique()
        DataFrame_Benign = Normalization_Benign.normalize_technique()
        DataFrame_Malignant = Normalization_Malignant.normalize_technique()

    elif New_technique == 'CLAHE':
        DataFrame_Normal = Normalization_Normal.CLAHE_technique()
        DataFrame_Benign = Normalization_Benign.CLAHE_technique()
        DataFrame_Malignant = Normalization_Malignant.CLAHE_technique()

    elif New_technique == 'HE':
        DataFrame_Normal = Normalization_Normal.histogram_equalization_technique()
        DataFrame_Benign = Normalization_Benign.histogram_equalization_technique()
        DataFrame_Malignant = Normalization_Malignant.histogram_equalization_technique()

    elif New_technique == 'UM':
        DataFrame_Normal = Normalization_Normal.unsharp_masking_technique()
        DataFrame_Benign = Normalization_Benign.unsharp_masking_technique()
        DataFrame_Malignant = Normalization_Malignant.unsharp_masking_technique()

    elif New_technique == 'CS':
        DataFrame_Normal = Normalization_Normal.contrast_stretching_technique()
        DataFrame_Benign = Normalization_Benign.contrast_stretching_technique()
        DataFrame_Malignant = Normalization_Malignant.contrast_stretching_technique()

    elif New_technique == 'MF':
        DataFrame_Normal = Normalization_Normal.median_filter_technique()
        DataFrame_Benign = Normalization_Benign.median_filter_technique()
        DataFrame_Malignant = Normalization_Malignant.median_filter_technique()

    else:
        raise ValueError("Choose a new technique")    #! Alert

    # * Concatenate dataframes with this function
    concat_dataframe(DataFrame_Normal, DataFrame_Benign, DataFrame_Malignant, Folder = Multiclass_Data_CSV, Class = Multiclass, Technique = New_technique)
