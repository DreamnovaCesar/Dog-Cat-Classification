import os
import cv2
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2s

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import normalized_mutual_information as nmi

from skimage import io
from skimage import filters
from skimage import img_as_ubyte
from skimage import img_as_float

from skimage.exposure import equalize_adapthist
from skimage.exposure import equalize_hist
from skimage.exposure import rescale_intensity

from skimage.filters import unsharp_mask

from Dog_Cat_Classification_1_General_Functions import sort_images
from Dog_Cat_Classification_1_General_Functions import remove_all_files

class ImageProcessing:
    """
    _summary_

    _extended_summary_
    """
    def __init__(self, **kwargs) -> None:
        """
        _summary_

        _extended_summary_

        Raises:
            ValueError: _description_
        """
        # * General parameters
        self.Folder = kwargs.get('folder', None)
        self.New_folder = kwargs.get('newfolder', None)
        self.Animal = kwargs.get('animal', None)
        self.Label = kwargs.get('label', None)

        # * Parameters for resizing
        self.Interpolation = kwargs.get('I', cv2.INTER_CUBIC)
        self.X_resize = kwargs.get('X', 224)
        self.Y_resize = kwargs.get('Y', 224)

        # * Parameters for median filter
        self.Division = kwargs.get('division', 3)

        # * Parameters for CLAHE
        self.Clip_limit = kwargs.get('cliplimit', 0.01)
        self.tile_grid_size = kwargs.get('tileGridSize', (8, 8))

        # * Parameters for unsharp masking
        self.Radius = kwargs.get('radius', 1)
        self.Amount = kwargs.get('amount', 1)

        # * Folder attribute (ValueError, TypeError)
        if self.Folder == None:
            raise ValueError("Folder does not exist") #! Alert
        if not isinstance(self.Folder, str):
            raise TypeError("Folder attribute must be a string") #! Alert

        # * New folder attribute (ValueError, TypeError)
        if self.New_folder == None:
            raise ValueError("Folder destination attribute does not exist") #! Alert
        if not isinstance(self.New_folder, str):
            raise TypeError("Folder destination attribute must be a string") #! Alert

        # * Animal (ValueError, TypeError)
        if self.Animal == None:
            raise ValueError("Severity attribute does not exist") #! Alert
        if not isinstance(self.Animal, str):
            raise TypeError("Severity attribute must be a string") #! Alert

        # * Label (ValueError, TypeError)
        if self.Label == None:
            raise ValueError("Label attribute does not exist") #! Alert
        if not isinstance(self.Label, int):
            raise TypeError("Label attribute must be a integer") #! Alert

        # ? Resize
        # * X resize (TypeError)
        if not isinstance(self.X_resize, int):
            raise TypeError("X must be integer") #! Alert
        # * Y resize (TypeError)
        if not isinstance(self.Y_resize, int):
            raise TypeError("Y must be integer") #! Alert

        # ? Median Filter
        # * Division (TypeError)
        if not isinstance(self.Division, int):
            raise TypeError("Division must be integer") #! Alert

        # ? CLAHE
        # * Clip limit (TypeError)
        if not isinstance(self.Clip_limit, float):
            raise TypeError("Clip limit must be float") #! Alert

        # ? Unsharp masking
        # * Radius and amount (TypeError)
        if not isinstance(self.Radius, int):
            raise TypeError("Radius must be integer") #! Alert
        if not isinstance(self.Amount, int):
            raise TypeError("Amount must be integer") #! Alert

    def __repr__(self):

      kwargs_info = "Folder: {}, New_folder: {}, Severity: {}, Label: {}, Label: {}, Interpolation: {}, X_resize: {}, Y_resize: {}, Division: {}, Clip_limit: {}, Radius: {}, Amount: {}".format(   self.Folder, self.New_folder, self.Severity,
                                                                                                                                                                                                    self.Label, self.Interpolation, self.X_resize,
                                                                                                                                                                                                    self.Y_resize, self.Division, self.Clip_limit,
                                                                                                                                                                                                    self.Radius, self.Amount)
      return kwargs_info

    def __str__(self):

        Descripcion_class = ""
        
        return Descripcion_class

    # * Folder attribute
    @property
    def Folder_property(self):
        return self.Folder

    @Folder_property.setter
    def Folder_property(self, New_value):
        if not isinstance(New_value, str):
            raise TypeError("Folder must be a string") #! Alert
        self.Folder = New_value
    
    @Folder_property.deleter
    def Folder_property(self):
        print("Deleting folder...")
        del self.Folder

    # * Folder destination attribute
    @property
    def Folder_dest_property(self):
        return self.Folder_dest

    @Folder_dest_property.setter
    def Folder_dest_property(self, New_value):
        if not isinstance(New_value, str):
            raise TypeError("Folder dest must be a string") #! Alert
        self.Folder_dest = New_value
    
    @Folder_dest_property.deleter
    def Folder_dest_property(self):
        print("Deleting destination folder...")
        del self.Folder_dest

    # * Animal attribute
    @property
    def Animal_property(self):
        return self.Animal

    @Animal_property.setter
    def Animal_property(self, New_value):
        if not isinstance(New_value, str):
            raise TypeError("Animal must be a string") #! Alert
        self.Animal = New_value
    
    @Animal_property.deleter
    def Animal_property(self):
        print("Deleting Animal...")
        del self.Animal
    
    # * Label attribute
    @property
    def Label_property(self):
        return self.Label

    @Label_property.setter
    def Label_property(self, New_value):
        if not isinstance(New_value, int):
            raise TypeError("Must be a integer value") #! Alert
        self.Label = New_value
    
    @Label_property.deleter
    def Label_property(self):
        print("Deleting label...")
        del self.Label

    # * Interpolation attribute
    @property
    def Interpolation_property(self):
        return self.Interpolation

    @Interpolation_property.setter
    def Interpolation_property(self, New_value):
        self.Interpolation = New_value
    
    @Interpolation_property.deleter
    def Interpolation_property(self):
        print("Deleting Interpolation...")
        del self.Interpolation

    # * X resize attribute
    @property
    def X_resize_property(self):
        return self.X_resize

    @X_resize_property.setter
    def X_resize_property(self, New_value):
        if not isinstance(New_value, int):
            raise TypeError("X must be a integer value") #! Alert
        self.X_resize = New_value
    
    @X_resize_property.deleter
    def X_resize_property(self):
        print("Deleting X...")
        del self.X_resize

    # * Y resize attribute
    @property
    def Y_resize_property(self):
        return self.Y_resize

    @Y_resize_property.setter
    def Y_resize_property(self, New_value):
        if not isinstance(New_value, int):
            raise TypeError("Y must be a integer value") #! Alert
        self.Y_resize = New_value
    
    @Y_resize_property.deleter
    def Y_resize_property(self):
        print("Deleting Y...")
        del self.Y_resize

    # * Division attribute
    @property
    def Division_property(self):
        return self.Division

    @Division_property.setter
    def Division_property(self, New_value):
        if not isinstance(New_value, int):
            raise TypeError("Division must be a integer value") #! Alert
        self.Division = New_value
    
    @Division_property.deleter
    def Division_property(self):
        print("Deleting division...")
        del self.Division

    # * Clip limit attribute
    @property
    def Clip_limit_property(self):
        return self.Clip_limit

    @Clip_limit_property.setter
    def Clip_limit_property(self, New_value):
        if not isinstance(New_value, float):
            raise TypeError("Clip limit must be a float value") #! Alert
        self.Clip_limit = New_value
    
    @Clip_limit_property.deleter
    def Clip_limit_property(self):
        print("Deleting clip limit...")
        del self.Clip_limit

    # * Radius attribute
    @property
    def Radius_property(self):
        return self.Radius

    @Radius_property.setter
    def Radius_property(self, New_value):
        if not isinstance(New_value, float):
            raise TypeError("Radius must be a float value") #! Alert
        self.Radius = New_value
    
    @Radius_property.deleter
    def Radius_property(self):
        print("Deleting Radius...")
        del self.Radius

    # * Amount attribute
    @property
    def Amount_property(self):
        return self.Amount

    @Amount_property.setter
    def Amount_property(self, New_value):
        if not isinstance(New_value, float):
            raise TypeError("Amount must be a float value") #! Alert
        self.Amount = New_value
    
    @Amount_property.deleter
    def Amount_property(self):
        print("Deleting Amount...")
        del self.Amount

    @staticmethod
    def print_processing(Count: int, Total_images: int, Animal: str, Filename: str) -> None:
        """
        _summary_

        _extended_summary_

        Args:
            Count (int): _description_
            Total_images (int): _description_
            Animal (_type_): _description_
        """

        print('Working with {} ✅'.format(Filename))
        print('Working with {} of {} {} images ✅'.format(Count, Total_images, Animal))

    # ? Resize technique method

    def resize_technique(self) -> pd.DataFrame:
        """
        _summary_

        _extended_summary_
        """
        # * Save the new images in a list
        #New_images = [] 

        os.chdir(self.Folder)
        print(os.getcwd())
        print("\n")

        # * Using sort function
        Sorted_files, Total_images = sort_images(self.Folder)
        Count:int = 0

        # * Reading the files
        for File in Sorted_files:
        
            # * Extract the file name and format
            Filename, Format  = os.path.splitext(File)
            
            if File.endswith(Format):

                try:
                
                    self.print_processing(Count, Total_images, self.Animal, Filename)
                    
                    # * Resize with the given values 
                    Path_file = os.path.join(self.Folder, File)
                    Imagen = cv2.imread(Path_file)

                    # * Resize with the given values 
                    Shape = (self.X_resize, self.Y_resize)
                    Resized_imagen = cv2.resize(Imagen, Shape, interpolation = self.Interpolation)

                    # * Show old image and new image
                    print(Imagen.shape, ' -------- ', Resized_imagen.shape)

                    # * Name the new file
                    New_name_filename = Filename + Format
                    New_folder = os.path.join(self.New_folder, New_name_filename)

                    # * Save the image in a new folder
                    cv2.imwrite(New_folder, Resized_imagen)
                    #New_images.append(Resized_Imagen)

                    Count += 1

                except OSError:
                    print('Cannot convert {} ❌'.format(str(File))) #! Alert

        print("\n")
        print('{} of {} tranformed ❌'.format(str(Count), str(Total_images))) #! Alert

    # ? Normalization technique method

    def normalize_technique(self):
        
        """
        Get the values from median filter images and save them into a dataframe.

        Parameters:
        argument1 (Folder): Folder chosen.
        argument2 (Folder): Folder destination for new images.
        argument3 (Str): Severity of each image.
        argument4 (Int): The number of the images.
        argument5 (Int): Division for median filter.
        argument6 (Str): Label for each image.

        Returns:
        int:Returning dataframe with all data.
        
        """
        # * Remove all the files in the new folder using this function
        remove_all_files(self.New_folder)

        # * Lists to save the values of the labels and the filename and later use them for a dataframe
        #Images = [] 
        Labels = []
        All_filenames = []
        
        # * Lists to save the values of each statistic
        Mae_ALL = [] # ? Mean Absolute Error.
        Mse_ALL = [] # ? Mean Squared Error.
        Ssim_ALL = [] # ? Structural similarity.
        Psnr_ALL = [] # ? Peak Signal-to-Noise Ratio.
        Nrmse_ALL = [] # ? Normalized Root-Mean-Square Error.
        Nmi_ALL = [] # ? Normalized Mutual Information.
        R2s_ALL = [] # ? Coefficient of determination.

        os.chdir(self.Folder)

        # * Using sort function
        Sorted_files, Total_images = sort_images(self.Folder)
        Count:int = 0

        # * Reading the files
        for File in Sorted_files:

            # * Extract the file name and format
            Filename, Format  = os.path.splitext(File)

            # * Read extension files
            if File.endswith(Format): 
                
                try:

                    self.print_processing(Count, Total_images, self.Animal, Filename)
                    #print(f"Working with {Filename} ✅")

                    # * Resize with the given values
                    Path_file = os.path.join(self.Folder, File)
                    Image = cv2.imread(Path_file)

                    Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)

                    #print("%s has %d and %d" % (File, Imagen.shape[0], Imagen.shape[1]))

                    # * Add a black image for normalization
                    Image_numpy_zeros = np.zeros((Image.shape[0], Image.shape[1]))
                    Normalization_imagen = cv2.normalize(Image, Image_numpy_zeros, 0, 255, cv2.NORM_MINMAX)

                    # * Save each statistic in a variable
                    Mae = mae(Image, Normalization_imagen)
                    Mse = mse(Image, Normalization_imagen)
                    Ssim = ssim(Image, Normalization_imagen)
                    Psnr = psnr(Image, Normalization_imagen)
                    Nrmse = nrmse(Image, Normalization_imagen)
                    Nmi = nmi(Image, Normalization_imagen)
                    R2s = r2s(Image, Normalization_imagen)

                    # * Add the value in the lists already created
                    Mae_ALL.append(Mae)
                    Mse_ALL.append(Mse)
                    Ssim_ALL.append(Ssim)
                    Psnr_ALL.append(Psnr)
                    Nrmse_ALL.append(Nrmse)
                    Nmi_ALL.append(Nmi)
                    R2s_ALL.append(R2s)

                    # * Name the new file
                    Filename_and_technique = '{}_Normalization'.format(str(Filename))
                    #Filename_and_technique = Filename + '_Normalization'
                    New_name_filename = Filename_and_technique + Format
                    New_folder = os.path.join(self.New_folder, New_name_filename)
                    
                    #Normalization_Imagen = Normalization_Imagen.astype('float32')
                    #Normalization_Imagen = Normalization_Imagen / 255.0

                    # * Save the image in a new folder
                    cv2.imwrite(New_folder, Normalization_imagen)
                    
                    # * Save the values of labels and each filenames
                    #Images.append(Normalization_Imagen)
                    All_filenames.append(Filename_and_technique)
                    Labels.append(self.Label)

                    Count += 1

                except OSError:
                    print('Cannot convert {} ❌'.format(str(File))) #! Alert

        print("\n")
        print('{} of {} tranformed ✅'.format(str(Count), str(Total_images))) #! Alert

        # * Return the new dataframe with the new data
        Dataframe = pd.DataFrame({'REFNUMMF_ALL':All_filenames, 'MAE':Mae_ALL, 'MSE':Mse_ALL, 'SSIM':Ssim_ALL, 'PSNR':Psnr_ALL, 'NRMSE':Nrmse_ALL, 'NMI':Nmi_ALL, 'R2s':R2s_ALL, 'Labels':Labels})

        return Dataframe

    # ? Median filter technique method

    def median_filter_technique(self) -> pd.DataFrame:

        """
        Get the values from median filter images and save them into a dataframe.

        Parameters:
        argument1 (Folder): Folder chosen.
        argument2 (Folder): Folder destination for new images.
        argument3 (Str): Severity of each image.
        argument4 (Int): The number of the images.
        argument5 (Int): Division for median filter.
        argument6 (Str): Label for each image.

        Returns:
        int:Returning dataframe with all data.
        
        """
        # * Remove all the files in the new folder using this function
        remove_all_files(self.New_folder)
        
        # * Lists to save the values of the labels and the filename and later use them for a dataframe
        #Images = [] 
        Labels = []
        All_filenames = []

        # * Lists to save the values of each statistic
        Mae_ALL = [] # ? Mean Absolute Error.
        Mse_ALL = [] # ? Mean Squared Error.
        Ssim_ALL = [] # ? Structural similarity.
        Psnr_ALL = [] # ? Peak Signal-to-Noise Ratio.
        Nrmse_ALL = [] # ? Normalized Root-Mean-Square Error.
        Nmi_ALL = [] # ? Normalized Mutual Information.
        R2s_ALL = [] # ? Coefficient of determination.

        os.chdir(self.Folder)

        # * Using sort function
        Sorted_files, Total_images = sort_images(self.Folder)
        Count:int = 0

        # * Reading the files
        for File in Sorted_files:
            
            # * Extract the file name and format
            Filename, Format  = os.path.splitext(File)

            # * Read extension files
            if File.endswith(Format):
            
                try:

                    self.print_processing(Count, Total_images, self.Animal, Filename)

                    # * Resize with the given values
                    Path_file = os.path.join(self.Folder, File)
                    Image = io.imread(Path_file, as_gray = True)

                    #Image_median_filter = cv2.medianBlur(Imagen, Division)
                    Median_filter_image = filters.median(Image, np.ones((self.Division, self.Division)))

                    # * Save each statistic in a variable
                    Mae = mae(Image, Median_filter_image)
                    Mse = mse(Image, Median_filter_image)
                    Ssim = ssim(Image, Median_filter_image)
                    Psnr = psnr(Image, Median_filter_image)
                    Nrmse = nrmse(Image, Median_filter_image)
                    Nmi = nmi(Image, Median_filter_image)
                    R2s = r2s(Image, Median_filter_image)

                    # * Add the value in the lists already created
                    Mae_ALL.append(Mae)
                    Mse_ALL.append(Mse)
                    Ssim_ALL.append(Ssim)
                    Psnr_ALL.append(Psnr)
                    Nrmse_ALL.append(Nrmse)
                    Nmi_ALL.append(Nmi)
                    R2s_ALL.append(R2s)

                    # * Name the new file
                    Filename_and_technique = '{}_Median_Filter'.format(str(Filename))
                    #Filename_and_technique = Filename + '_Median_Filter'
                    New_name_filename = Filename_and_technique + Format
                    New_folder = os.path.join(self.New_folder, New_name_filename)

                    # * Save the image in a new folder
                    io.imsave(New_folder, Median_filter_image)

                    # * Save the values of labels and each filenames
                    #Images.append(Normalization_Imagen)
                    Labels.append(self.Label)
                    All_filenames.append(Filename_and_technique)

                    Count += 1

                except OSError:
                    print('Cannot convert {} ❌'.format(str(File))) #! Alert

        print("\n")
        print('{} of {} tranformed ✅'.format(str(Count), str(Total_images))) #! Alert

        # * Return the new dataframe with the new data
        DataFrame = pd.DataFrame({'REFNUMMF_ALL':All_filenames, 'MAE':Mae_ALL, 'MSE':Mse_ALL, 'SSIM':Ssim_ALL, 'PSNR':Psnr_ALL, 'NRMSE':Nrmse_ALL, 'NMI':Nmi_ALL, 'R2s':R2s_ALL, 'Labels':Labels})

        return DataFrame

    # ? CLAHE technique method

    def CLAHE_technique(self) -> pd.DataFrame:

        """
        Get the values from CLAHE images and save them into a dataframe.

        Parameters:
        argument1 (Folder): Folder chosen.
        argument2 (Folder): Folder destination for new images.
        argument3 (Str): Severity of each image.
        argument4 (Float): clip limit value use to change CLAHE images.
        argument5 (Str): Label for each image.

        Returns:
        int:Returning dataframe with all data.
        
        """
        # * Remove all the files in the new folder using this function
        remove_all_files(self.New_folder)
        
        # * Lists to save the values of the labels and the filename and later use them for a dataframe
        #Images = [] 
        Labels = []
        All_filenames = []

        # * Lists to save the values of each statistic
        Mae_ALL = [] # ? Mean Absolute Error.
        Mse_ALL = [] # ? Mean Squared Error.
        Ssim_ALL = [] # ? Structural similarity.
        Psnr_ALL = [] # ? Peak Signal-to-Noise Ratio.
        Nrmse_ALL = [] # ? Normalized Root-Mean-Square Error.
        Nmi_ALL = [] # ? Normalized Mutual Information.
        R2s_ALL = [] # ? Coefficient of determination.

        os.chdir(self.Folder)

        # * Using sort function
        Sorted_files, Total_images = sort_images(self.Folder)
        Count = 0

        # * Reading the files
        for File in Sorted_files:
            
            # * Extract the file name and format
            Filename, Format  = os.path.splitext(File)

            # * Read extension files
            if File.endswith(Format):
            
                try:

                    self.print_processing(Count, Total_images, self.Animal, Filename)

                    # * Resize with the given values
                    Path_file = os.path.join(self.Folder, File)
                    Image = io.imread(Path_file, as_gray = True)

                    #Imagen = cv2.cvtColor(Imagen, cv2.COLOR_BGR2GRAY)
                    #CLAHE = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
                    #CLAHE_Imagen = CLAHE.apply(Imagen)

                    CLAHE_image = equalize_adapthist(Image, clip_limit = self.Clip_limit)

                    Image = img_as_ubyte(Image)
                    CLAHE_image = img_as_ubyte(CLAHE_image)

                    # * Save each statistic in a variable
                    Mae = mae(Image, CLAHE_image)
                    Mse = mse(Image, CLAHE_image)
                    Ssim = ssim(Image, CLAHE_image)
                    Psnr = psnr(Image, CLAHE_image)
                    Nrmse = nrmse(Image, CLAHE_image)
                    Nmi = nmi(Image, CLAHE_image)
                    R2s = r2s(Image, CLAHE_image)

                    # * Add the value in the lists already created
                    Mae_ALL.append(Mae)
                    Mse_ALL.append(Mse)
                    Ssim_ALL.append(Ssim)
                    Psnr_ALL.append(Psnr)
                    Nrmse_ALL.append(Nrmse)
                    Nmi_ALL.append(Nmi)
                    R2s_ALL.append(R2s)

                    # * Name the new file
                    Filename_and_technique = '{}_CLAHE'.format(str(Filename))
                    #Filename_and_technique = Filename + '_CLAHE'
                    New_name_filename = Filename_and_technique + Format
                    New_folder = os.path.join(self.New_folder, New_name_filename)

                    # * Save the image in a new folder
                    io.imsave(New_folder, CLAHE_image)

                    # * Save the values of labels and each filenames
                    #Images.append(Normalization_Imagen)
                    Labels.append(self.Label)
                    All_filenames.append(Filename_and_technique)

                    Count += 1

                except OSError:
                    print('Cannot convert {} ❌'.format(str(File))) #! Alert

        print("\n")
        print('{} of {} tranformed ✅'.format(str(Count), str(Total_images))) #! Alert

        # * Return the new dataframe with the new data
        DataFrame = pd.DataFrame({'REFNUMMF_ALL':All_filenames, 'MAE':Mae_ALL, 'MSE':Mse_ALL, 'SSIM':Ssim_ALL, 'PSNR':Psnr_ALL, 'NRMSE':Nrmse_ALL, 'NMI':Nmi_ALL, 'R2s':R2s_ALL, 'Labels':Labels})

        return DataFrame

    # ? Histogram equalization technique method

    def histogram_equalization_technique(self) -> pd.DataFrame:

        """
        Get the values from histogram equalization images and save them into a dataframe.

        Parameters:
        argument1 (Folder): Folder chosen.
        argument2 (Folder): Folder destination for new images.
        argument3 (Str): Severity of each image.
        argument4 (Str): Label for each image.

        Returns:
        int:Returning dataframe with all data.
        
        """
        # * Remove all the files in the new folder using this function
        remove_all_files(self.New_folder)
        
        # * Lists to save the values of the labels and the filename and later use them for a dataframe
        #Images = [] 
        Labels = []
        All_filenames = []

        # * Lists to save the values of each statistic
        Mae_ALL = [] # ? Mean Absolute Error.
        Mse_ALL = [] # ? Mean Squared Error.
        Ssim_ALL = [] # ? Structural similarity.
        Psnr_ALL = [] # ? Peak Signal-to-Noise Ratio.
        Nrmse_ALL = [] # ? Normalized Root-Mean-Square Error.
        Nmi_ALL = [] # ? Normalized Mutual Information.
        R2s_ALL = [] # ? Coefficient of determination.

        os.chdir(self.Folder)

        # * Using sort function
        Sorted_files, Total_images = sort_images(self.Folder)
        Count = 0

        # * Reading the files
        for File in Sorted_files:
            
            # * Extract the file name and format
            Filename, Format  = os.path.splitext(File)

            # * Read extension files
            if File.endswith(Format):
            
                try:

                    self.print_processing(Count, Total_images, self.Animal, Filename)

                    # * Resize with the given values
                    Path_file = os.path.join(self.Folder, File)
                    Image = io.imread(Path_file, as_gray = True)

                    HE_image = equalize_hist(Image)

                    Image = img_as_ubyte(Image)
                    HE_image = img_as_ubyte(HE_image)

                    # * Save each statistic in a variable
                    Mae = mae(Image, HE_image)
                    Mse = mse(Image, HE_image)
                    Ssim = ssim(Image, HE_image)
                    Psnr = psnr(Image, HE_image)
                    Nrmse = nrmse(Image, HE_image)
                    Nmi = nmi(Image, HE_image)
                    R2s = r2s(Image, HE_image)

                    # * Add the value in the lists already created
                    Mae_ALL.append(Mae)
                    Mse_ALL.append(Mse)
                    Ssim_ALL.append(Ssim)
                    Psnr_ALL.append(Psnr)
                    Nrmse_ALL.append(Nrmse)
                    Nmi_ALL.append(Nmi)
                    R2s_ALL.append(R2s)

                    # * Name the new file
                    Filename_and_technique = '{}_HE'.format(str(Filename))
                    #Filename_and_technique = Filename + '_HE'
                    New_name_filename = Filename_and_technique + Format
                    New_folder = os.path.join(self.New_folder, New_name_filename)

                    # * Save the image in a new folder
                    io.imsave(New_folder, HE_image)

                    # * Save the values of labels and each filenames
                    #Images.append(Normalization_Imagen)
                    Labels.append(self.Label)
                    All_filenames.append(Filename_and_technique)
                    Count += 1

                except OSError:
                    print('Cannot convert {} ❌'.format(str(File))) #! Alert

        print("\n")
        print('{} of {} tranformed ✅'.format(str(Count), str(Total_images))) #! Alert

        # * Return the new dataframe with the new data
        DataFrame = pd.DataFrame({'REFNUMMF_ALL':All_filenames, 'MAE':Mae_ALL, 'MSE':Mse_ALL, 'SSIM':Ssim_ALL, 'PSNR':Psnr_ALL, 'NRMSE':Nrmse_ALL, 'NMI':Nmi_ALL, 'R2s':R2s_ALL, 'Labels':Labels})

        return DataFrame

    # ? Unsharp masking technique method

    def unsharp_masking_technique(self) -> pd.DataFrame:

        """
        Get the values from unsharp masking images and save them into a dataframe.

        Parameters:
        argument1 (Folder): Folder chosen.
        argument2 (Folder): Folder destination for new images.
        argument3 (str): Severity of each image.
        argument4 (float): Radius value use to change Unsharp mask images.
        argument5 (float): Amount value use to change Unsharp mask images.
        argument6 (str): Label for each image.

        Returns:
        int:Returning dataframe with all data.
        
        """
        # * Remove all the files in the new folder using this function
        remove_all_files(self.New_folder)
        
        # * Lists to save the values of the labels and the filename and later use them for a dataframe
        #Images = [] 
        Labels = []
        All_filenames = []

        # * Lists to save the values of each statistic
        Mae_ALL = [] # ? Mean Absolute Error.
        Mse_ALL = [] # ? Mean Squared Error.
        Ssim_ALL = [] # ? Structural similarity.
        Psnr_ALL = [] # ? Peak Signal-to-Noise Ratio.
        Nrmse_ALL = [] # ? Normalized Root-Mean-Square Error.
        Nmi_ALL = [] # ? Normalized Mutual Information.
        R2s_ALL = [] # ? Coefficient of determination.

        os.chdir(self.Folder)

        # * Using sort function
        Sorted_files, Total_images = sort_images(self.Folder)
        Count = 0

        # * Reading the files
        for File in Sorted_files:
            
            # * Extract the file name and format
            Filename, Format  = os.path.splitext(File)

            # * Read extension files
            if File.endswith(Format):
            
                try:

                    self.print_processing(Count, Total_images, self.Animal, Filename)

                    # * Resize with the given values
                    Path_file = os.path.join(self.Folder, File)
                    Image = io.imread(Path_file, as_gray = True)

                    UM_image = unsharp_mask(Image, radius = self.Radius, amount = self.Amount)

                    Image = img_as_ubyte(Image)
                    UM_image = img_as_ubyte(UM_image)

                    # * Save each statistic in a variable
                    Mae = mae(Image, UM_image)
                    Mse = mse(Image, UM_image)
                    Ssim = ssim(Image, UM_image)
                    Psnr = psnr(Image, UM_image)
                    Nrmse = nrmse(Image, UM_image)
                    Nmi = nmi(Image, UM_image)
                    R2s = r2s(Image, UM_image)

                    # * Add the value in the lists already created
                    Mae_ALL.append(Mae)
                    Mse_ALL.append(Mse)
                    Ssim_ALL.append(Ssim)
                    Psnr_ALL.append(Psnr)
                    Nrmse_ALL.append(Nrmse)
                    Nmi_ALL.append(Nmi)
                    R2s_ALL.append(R2s)

                    # * Name the new file
                    Filename_and_technique = '{}_UM'.format(str(Filename))
                    #Filename_and_technique = Filename + '_UM'
                    New_name_filename = Filename_and_technique + Format
                    New_folder = os.path.join(self.New_folder, New_name_filename)

                    # * Save the image in a new folder
                    io.imsave(New_folder, UM_image)

                    # * Save the values of labels and each filenames
                    #Images.append(Normalization_Imagen)
                    Labels.append(self.Label)
                    All_filenames.append(Filename_and_technique)
                    Count += 1

                except OSError:
                    print('Cannot convert {} ❌'.format(str(File))) #! Alert

        print("\n")
        print('{} of {} tranformed ✅'.format(str(Count), str(Total_images))) #! Alert

        # * Return the new dataframe with the new data
        Dataframe = pd.DataFrame({'REFNUMMF_ALL':All_filenames, 'MAE':Mae_ALL, 'MSE':Mse_ALL, 'SSIM':Ssim_ALL, 'PSNR':Psnr_ALL, 'NRMSE':Nrmse_ALL, 'NMI':Nmi_ALL, 'R2s':R2s_ALL, 'Labels':Labels})

        return Dataframe

    # ? Contrast Stretching technique method

    def contrast_stretching_technique(self) -> pd.DataFrame:

        """
        Get the values from constrast streching images and save them into a dataframe.

        Parameters:
        argument1 (Folder): Folder chosen.
        argument2 (Folder): Folder destination for new images.
        argument3 (str): Severity of each image.
        argument6 (str): Label for each image.

        Returns:
        int:Returning dataframe with all data.
        
        """
        # * Remove all the files in the new folder using this function
        remove_all_files(self.New_folder)
        
        # * Lists to save the values of the labels and the filename and later use them for a dataframe
        #Images = [] 
        Labels = []
        All_filenames = []

        # * Lists to save the values of each statistic
        Mae_ALL = [] # ? Mean Absolute Error.
        Mse_ALL = [] # ? Mean Squared Error.
        Ssim_ALL = [] # ? Structural similarity.
        Psnr_ALL = [] # ? Peak Signal-to-Noise Ratio.
        Nrmse_ALL = [] # ? Normalized Root-Mean-Square Error.
        Nmi_ALL = [] # ? Normalized Mutual Information.
        R2s_ALL = [] # ? Coefficient of determination.

        os.chdir(self.Folder)

        # * Using sort function
        Sorted_files, Total_images = sort_images(self.Folder)
        Count = 0

        # * Reading the files
        for File in Sorted_files:
            
            # * Extract the file name and format
            Filename, Format  = os.path.splitext(File)

            # * Read extension files
            if File.endswith(Format):
            
                try:

                    self.print_processing(Count, Total_images, self.Animal, Filename)

                    # * Resize with the given values
                    Path_file = os.path.join(self.Folder, File)
                    Image = io.imread(Path_file, as_gray = True)

                    p2, p98 = np.percentile(Image, (2, 98))
                    CS_image = rescale_intensity(Image, in_range = (p2, p98))

                    Image = img_as_ubyte(Image)
                    CS_image = img_as_ubyte(CS_image)

                    # * Save each statistic in a variable
                    Mae = mae(Image, CS_image)
                    Mse = mse(Image, CS_image)
                    Ssim = ssim(Image, CS_image)
                    Psnr = psnr(Image, CS_image)
                    Nrmse = nrmse(Image, CS_image)
                    Nmi = nmi(Image, CS_image)
                    R2s = r2s(Image, CS_image)

                    # * Add the value in the lists already created
                    Mae_ALL.append(Mae)
                    Mse_ALL.append(Mse)
                    Ssim_ALL.append(Ssim)
                    Psnr_ALL.append(Psnr)
                    Nrmse_ALL.append(Nrmse)
                    Nmi_ALL.append(Nmi)
                    R2s_ALL.append(R2s)

                    # * Name the new file
                    Filename_and_technique = '{}_CS'.format(str(Filename))
                    #Filename_and_technique = Filename + '_CS'
                    New_name_filename = Filename_and_technique + Format
                    New_folder = os.path.join(self.New_folder, New_name_filename)

                    # * Save the image in a new folder
                    io.imsave(New_folder, CS_image)

                    # * Save the values of labels and each filenames
                    #Images.append(Normalization_Imagen)
                    Labels.append(self.Label)
                    All_filenames.append(Filename_and_technique)
                    Count += 1

                except OSError:
                    print('Cannot convert {} ❌'.format(str(File))) #! Alert

        print("\n")
        print('{} of {} tranformed ✅'.format(str(Count), str(Total_images))) #! Alert

        # * Return the new dataframe with the new data
        Dataframe = pd.DataFrame({'REFNUMMF_ALL':All_filenames, 'MAE':Mae_ALL, 'MSE':Mse_ALL, 'SSIM':Ssim_ALL, 'PSNR':Psnr_ALL, 'NRMSE':Nrmse_ALL, 'NMI':Nmi_ALL, 'R2s':R2s_ALL, 'Labels':Labels})

        return Dataframe

    def CLAHE_RGB_technique(self) -> pd.DataFrame:

        """
        Get the values from CLAHE images and save them into a dataframe.

        Parameters:
        argument1 (Folder): Folder chosen.
        argument2 (Folder): Folder destination for new images.
        argument3 (Str): Severity of each image.
        argument4 (Float): clip limit value use to change CLAHE images.
        argument5 (Str): Label for each image.

        Returns:
        int:Returning dataframe with all data.

        """
        # * Remove all the files in the new folder using this function
        remove_all_files(self.New_folder)

        # * Lists to save the values of the labels and the filename and later use them for a dataframe
        #Images = [] 
        Labels = []
        All_filenames = []

        # * Lists to save the values of each statistic
        Mae_ALL = [] # ? Mean Absolute Error.
        Mse_ALL = [] # ? Mean Squared Error.
        Ssim_ALL = [] # ? Structural similarity.
        Psnr_ALL = [] # ? Peak Signal-to-Noise Ratio.
        Nrmse_ALL = [] # ? Normalized Root-Mean-Square Error.
        Nmi_ALL = [] # ? Normalized Mutual Information.
        R2s_ALL = [] # ? Coefficient of determination.

        # *
        os.chdir(self.Folder)

        # * Using sort function
        Sorted_files, Total_images = sort_images(self.Folder)
        Count = 0

        # * Reading the files
        for File in Sorted_files:

            # * Extract the file name and format
            Filename, Format  = os.path.splitext(File)

            # * Read extension files
            if File.endswith(Format):
                
                try:

                    self.print_processing(Count, Total_images, self.Animal, Filename)

                    # * Resize with the given values
                    Path_file = os.path.join(self.Folder, File)
                    Image = cv2.imread(Path_file)

                    HSV_image = cv2.cvtColor(Image, cv2.COLOR_BGR2HSV)

                    # *
                    h = HSV_image[:, :, 0]
                    s = HSV_image[:, :, 1]
                    v = HSV_image[:, :, 2]

                    #CLAHE = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
                    #CLAHE_Imagen = CLAHE.apply(Imagen)

                    # *
                    CLAHE = cv2.createCLAHE(clipLimit = self.Clip_limit, tileGridSize = self.tile_grid_size)
                    v = CLAHE.apply(v)

                    # *
                    HSV_image = np.dstack((h, s, v))

                    # *
                    RGB_CLAHE_image = cv2.cvtColor(Image, cv2.COLOR_HSV2RGB)

                    # * Save each statistic in a variable
                    #Mae = mae(Image, RGB_CLAHE_image)
                    #Mse = mse(Image, RGB_CLAHE_image)
                    #Ssim = ssim(Image, RGB_CLAHE_image)
                    #Psnr = psnr(Image, RGB_CLAHE_image)
                    #Nrmse = nrmse(Image, RGB_CLAHE_image)
                    #Nmi = nmi(Image, RGB_CLAHE_image)
                    #R2s = r2s(Image, RGB_CLAHE_image)

                    # * Add the value in the lists already created
                    #Mae_ALL.append(Mae)
                    #Mse_ALL.append(Mse)
                    #Ssim_ALL.append(Ssim)
                    #Psnr_ALL.append(Psnr)
                    #Nrmse_ALL.append(Nrmse)
                    #Nmi_ALL.append(Nmi)
                    #R2s_ALL.append(R2s)

                    # * Name the new file
                    Filename_and_technique = '{}_CLAHE_RGB'.format(str(Filename))
                    #Filename_and_technique = Filename + '_CLAHE'
                    New_name_filename = Filename_and_technique + Format
                    New_folder = os.path.join(self.New_folder, New_name_filename)

                    # * Save the image in a new folder
                    cv2.imwrite(New_folder, RGB_CLAHE_image)

                    # * Save the values of labels and each filenames
                    #Images.append(Normalization_Imagen)
                    Labels.append(self.Label)
                    All_filenames.append(Filename_and_technique)

                    Count += 1

                except OSError:
                    print('Cannot convert {} ❌'.format(str(File))) #! Alert

        print("\n")
        print('{} of {} tranformed ✅'.format(str(Count), str(Total_images))) #! Alert

        # * Return the new dataframe with the new data
        #DataFrame = pd.DataFrame({'REFNUMMF_ALL':All_filenames, 'MAE':Mae_ALL, 'MSE':Mse_ALL, 'SSIM':Ssim_ALL, 'PSNR':Psnr_ALL, 'NRMSE':Nrmse_ALL, 'NMI':Nmi_ALL, 'R2s':R2s_ALL, 'Labels':Labels})

        #return DataFrame