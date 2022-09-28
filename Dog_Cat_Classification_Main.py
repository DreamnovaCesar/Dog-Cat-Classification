import numpy as np

from Dog_Cat_Classification_1_General_Functions import Number_images
from Dog_Cat_Classification_1_General_Functions import create_dataframe
from Dog_Cat_Classification_Preprocessing_1_Image_Processing import image_preprocessing

from Dog_Cat_Classification_4_Image_Processing import ImageProcessing

Model_CNN = (13, 14)

def Creation_Dataframe():

    Df_save = 'D:\Classification_Dog_Cat'
    
    create_dataframe([  'Folder', 'New Folder', 
                        'Animal', 'Label', 'Interpolation', 
                        'X', 'Y', 'Division', 'Clip Limit', 
                        'TitleGridSize X', 'TitleGridSize Y', 
                        'Radius', 'Amount', 'Technique', 'Folder CSV' ], Df_save, 'Dog_Cat')
    
def CLAHE_RGB():
    image_preprocessing("D:\Classification_Dog_Cat\Dataframe_Dog_Cat_CLAHE.csv")

def Resize():
    image_preprocessing("D:\Classification_Dog_Cat\Dataframe_Dog_Cat.csv")

def Test():
    Number_images([r"D:\Classification_Dog_Cat\training_set\training_set\cats", r"D:\Classification_Dog_Cat\training_set\training_set\dogs"], r'D:\Classification_Dog_Cat')

def main():
    CLAHE_RGB()

    #Testing_CNN_Models_Biclass_From_Folder(Model_CNN, 'D:\Mini-MIAS\Mini_MIAS_NO_Cropped_Images_Biclass' + '_Split', 'TEST')

if __name__ == "__main__":
    main()