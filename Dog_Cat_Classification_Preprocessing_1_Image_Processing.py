
########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from ntpath import join
import cv2
import pandas as pd

from Dog_Cat_Classification_1_General_Functions import concat_dataframe_list
from Dog_Cat_Classification_4_Image_Processing import ImageProcessing

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

def image_preprocessing(CSV_path: str) -> None:

    with open(CSV_path) as CSV:
        
        Data = pd.read_csv(CSV)

        # *
        Folder = Data['Folder'].tolist()
        New_folder = Data['New Folder'].tolist()
        Animal = Data['Animal'].tolist()
        Label = Data['Label'].tolist()

        #Label = list(map(int, Label))

        # *
        Interpolation = Data['Interpolation'].tolist()
        X_size = Data['X'].tolist()
        Y_size = Data['Y'].tolist()

        for i, Interpo in enumerate(Interpolation):
            if(Interpo == 'INTER_CUBIC'):
                Interpolation[i] = cv2.INTER_CUBIC

        #X_size = list(map(int, X_size))
        #Y_size = list(map(int, Y_size))

        # *
        Division = Data['Division'].tolist()

        #Division = list(map(int, Division))

        # *
        Clip_limit = Data['Clip Limit'].tolist()
        Title_grid_size_X = Data['TitleGridSize X'].tolist()
        Title_grid_size_Y = Data['TitleGridSize Y'].tolist()
        
        Clip_limit = list(map(float, Clip_limit))
        Title_grid_size_X = list(map(int, Title_grid_size_X))
        Title_grid_size_Y = list(map(int, Title_grid_size_Y))

        Title_grid_size = (Title_grid_size_X[0], Title_grid_size_Y[0])

        # *
        Radius = Data['Radius'].tolist()
        Amount = Data['Amount'].tolist()

        Radius = [int(item) for item in Radius]
        Amount = [int(item) for item in Amount]

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
            Object_IP.append(ImageProcessing(folder = Folder[i], newfolder = New_folder[i], animal = Animal[i], label = Label[i], I = Interpolation[i], X = X_size[i], Y = Y_size[i],
                                    cliplimit = Clip_limit[i], tileGridSize = Title_grid_size, division = Division[i], radius = Radius[i], amount = Amount[i]))

        for i in range(len(Object_IP)):

            # * Choose the technique utilized for the test

            if Technique[i] == 'RE':

                Dataframes[i] = Object_IP[i].resize_technique()
                
            elif Technique[i] == 'NO':

                Dataframes[i] = Object_IP[i].normalize_technique()

            elif Technique[i] == 'CLAHE':

                Dataframes[i] = Object_IP[i].CLAHE_technique()

            elif Technique[i] == 'HE':

                Dataframes[i] = Object_IP[i].histogram_equalization_technique()

            elif Technique[i] == 'UM':

                Dataframes[i] = Object_IP[i].unsharp_masking_technique()

            elif Technique[i] == 'CS':

                Dataframes[i] = Object_IP[i].contrast_stretching_technique()
            
            elif Technique[i] == 'MF':

                Dataframes[i] = Object_IP[i].median_filter_technique()
            
            elif Technique[i] == 'CLAHERGB':

                Dataframes[i] = Object_IP[i].CLAHE_RGB_technique()

            else:
                raise ValueError("Choose a new technique")      #! Alert

        if(Technique[i] != 'RE'):
            # * Concatenate dataframes with this function
            concat_dataframe_list(Dataframes, folder = CSV_path_folder[i], classp = Class_problem_prefix, technique = Technique[i], savefile = True)

