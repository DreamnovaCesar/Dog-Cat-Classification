import numpy as np

from Dog_Cat_Classification_1_General_Functions import Number_images
from Dog_Cat_Classification_Preprocessing_2_CNN_Models import Testing_CNN_Models_Biclass_From_Folder

from Dog_Cat_Classification_4_Image_Processing import ImageProcessing

Model_CNN = (13, 14)

def CLAHE_RGB():
    Normalization_Normal = ImageProcessing(Folder = Folder_normal, Newfolder = New_folder_normal, Severity = Label_Calcification, Label = Calcification_images_class,
                                            cliplimit = Cliplimit, division = Division, radius = Radius, amount = Amount)

def Resize():
    
    Resize_Dog = ImageProcessing(Folder = Folder_normal, Newfolder = New_folder_normal, Severity = Label_Calcification, Label = Calcification_images_class,
                                            cliplimit = Cliplimit, division = Division, radius = Radius, amount = Amount)
    Resize_Dog = ImageProcessing(Folder = Folder_normal, Newfolder = New_folder_normal, Severity = Label_Calcification, Label = Calcification_images_class,
                                            cliplimit = Cliplimit, division = Division, radius = Radius, amount = Amount)
                                            
def test():
    Number_images([r"D:\Classification_Dog_Cat\training_set\training_set\cats", r"D:\Classification_Dog_Cat\training_set\training_set\dogs"], r'D:\Classification_Dog_Cat')

def main():
    test()

    #Testing_CNN_Models_Biclass_From_Folder(Model_CNN, 'D:\Mini-MIAS\Mini_MIAS_NO_Cropped_Images_Biclass' + '_Split', 'TEST')

if __name__ == "__main__":
    main()