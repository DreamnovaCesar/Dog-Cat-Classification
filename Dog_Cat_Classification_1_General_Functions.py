import os
import random
import datetime
import cv2
import string
import shutil
import pydicom
import warnings
import numpy as np
import pandas as pd
import seaborn as sns

import splitfolders
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from random import sample

from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder

from cryptography.fernet import Fernet

from sklearn.metrics import auc

import random
from random import randint

# ? Create dataframes

def create_dataframe(Column_names, Folder_save: str, CSV_name: str) -> None: 

    # * Lists
    #Column_names = ['Folder', 'New Folder', 'Animal', 'Label']
    
    # *
    Dataframe_created = pd.DataFrame(columns = Column_names)

    # *
    Dataframe_name = "Dataframe_{}.csv".format(CSV_name)
    Dataframe_folder = os.path.join(Folder_save, Dataframe_name)

    # *
    Dataframe_created.to_csv(Dataframe_folder)

# ? Create folders

def create_folders(Folder_path: str, Folder_name: str, CSV_name: str) -> None: 

  Path_names = []
  Path_absolute_dir = []

  if(len(Folder_name) >= 2):

    for i, Path_name in enumerate(Folder_name):

      Folder_path_new = r'{}\{}'.format(Folder_path, Folder_name[i])
      print(Folder_path_new)

      Path_names.append(Path_name)
      Path_absolute_dir.append(Folder_path_new)

      Exist_dir = os.path.isdir(Folder_path_new) 

      if Exist_dir == False:
        os.mkdir(Folder_path_new)
      else:
        print('Path: {} exists, use another name for it'.format(Folder_name[i]))

  else:

    Folder_path_new = r'{}\{}'.format(Folder_path, Folder_name)
    print(Folder_path_new)

    Path_names.append(Folder_name)
    Path_absolute_dir.append(Folder_path_new)

    Exist_dir = os.path.isdir(Folder_path_new) 

    if Exist_dir == False:
      os.mkdir(Folder_path_new)
    else:
      print('Path: {} exists, use another name for it'.format(Folder_name))

  Dataframe_name = 'Dataframe_path_names_{}.csv'.format(CSV_name)
  Dataframe_folder = os.path.join(Folder_path, Dataframe_name)

  #Exist_dataframe = os.path.isfile(Dataframe_folder)

  Dataframe = pd.DataFrame({'Names':Path_names, 'Path names':Path_absolute_dir})
  Dataframe.to_csv(Dataframe_folder)


# ? creating_data_students

def creating_data_students(Dataframe: pd.DataFrame, Iter: int, Folder_path: str, Save_dataframe: bool = False) -> pd.DataFrame: 
    
    # * Tuples for random generation.
    Random_Name = ('Tom', 'Nick', 'Chris', 'Jack', 'Thompson')
    Random_Classroom = ('A', 'B', 'C', 'D', 'E')

    for i in range(Iter):

        # *
        New_row = {'Name':random.choice(Random_Name),
                   'Age':randint(16, 21),
                   'Classroom':random.choice(Random_Classroom),
                   'Height':randint(160, 190),
                   'Math':randint(70, 100),
                   'Chemistry':randint(70, 100),
                   'Physics':randint(70, 100),
                   'Literature':randint(70, 100)}

        Dataframe = Dataframe.append(New_row, ignore_index = True)

        # *
        print('Iteration complete: {}'.format(i))

    # *
    if(Save_dataframe == True):
      Dataframe_Key_name = 'Dataframe_filekeys.csv'.format()
      Dataframe_Key_folder = os.path.join(Folder_path, Dataframe_Key_name)

      Dataframe.to_csv(Dataframe_Key_folder)

    return Dataframe

# ? Generate keys

def generate_key(Folder_path: str, Number_keys: int = 2) -> None: 

    Names = []
    Keys = []
    
    # * Folder attribute (ValueError, TypeError)
    if Folder_path == None:
        raise ValueError("Folder does not exist") #! Alert
    if not isinstance(Folder_path, str):
        raise TypeError("Folder must be a string") #! Alert

    # key generation
    for i in range(Number_keys):

        Key = Fernet.generate_key()
        
        print('Key created: {}'.format(Key))

        Key_name = 'filekey_{}'.format(i)
        Key_path_name = '{}/filekey_{}.key'.format(Folder_path, i)

        Keys.append(Key)
        Names.append(Key_name)

        with open(Key_path_name, 'wb') as Filekey:
            Filekey.write(Key)

        Dataframe_keys = pd.DataFrame({'Name':Names, 'Keys':Keys})

        Dataframe_Key_name = 'Dataframe_filekeys.csv'.format()
        Dataframe_Key_folder = os.path.join(Folder_path, Dataframe_Key_name)

        Dataframe_keys.to_csv(Dataframe_Key_folder)

# ? Encrypt files

def Encrypt_files(**kwargs) -> None:

    # * General parameters
    Folder_path = kwargs.get('folderpath', None)
    Key_path = kwargs.get('keypath', None)
    Keys_path = kwargs.get('keyspath', None)
    Key_path_chosen = kwargs.get('newkeypath', None)
    Random_key = kwargs.get('randomkey', False)

    # * Folder attribute (ValueError, TypeError)
    if Folder_path == None:
        raise ValueError("Folder does not exist") #! Alert
    if not isinstance(Folder_path, str):
        raise TypeError("Folder must be a string") #! Alert

    Filenames = []

    if Random_key == True:

        File = random.choice(os.listdir(Keys_path))
        FilenameKey, Format = os.path.splitext(File)

        if File.endswith('.key'):

            try:
                with open(Keys_path + '/' + File, 'rb') as filekey:
                    Key = filekey.read()

                Fernet_ = Fernet(Key)

                #Key_name = 'MainKey.key'.format()
                shutil.copy2(os.path.join(Keys_path, File), Key_path_chosen)

                # * This function sort the files and show them

                for Filename in os.listdir(Folder_path):

                    Filenames.append(Filename)

                    with open(Folder_path + '/' + Filename, 'rb') as File_: # open in readonly mode
                        Original_file = File_.read()
                    
                    Encrypted_File = Fernet_.encrypt(Original_file)

                    with open(Folder_path + '/' + Filename, 'wb') as Encrypted_file:
                        Encrypted_file.write(Encrypted_File) 

                with open(Key_path_chosen + '/' + FilenameKey + '.txt', "w") as text_file:
                    text_file.write('The key {} open the next documents {}'.format(FilenameKey, Filenames))   

            except OSError:
                print('Is not a key {} ❌'.format(str(File))) #! Alert

    elif Random_key == False:

        Name_key = os.path.basename(Key_path)
        Key_dir = os.path.dirname(Key_path)

        if Key_path.endswith('.key'):
            
            try: 
                with open(Key_path, 'rb') as filekey:
                    Key = filekey.read()

                Fernet_ = Fernet(Key)

                #Key_name = 'MainKey.key'.format()
                shutil.copy2(os.path.join(Key_dir, Name_key), Key_path_chosen)

                # * This function sort the files and show them

                for Filename in os.listdir(Folder_path):

                    Filenames.append(Filename)

                    with open(Folder_path + '/' + Filename, 'rb') as File: # open in readonly mode
                        Original_file = File.read()
                    
                    Encrypted_File = Fernet_.encrypt(Original_file)

                    with open(Folder_path + '/' + Filename, 'wb') as Encrypted_file:
                        Encrypted_file.write(Encrypted_File)

                with open(Key_path_chosen + '/' + Name_key + '.txt', "w") as text_file:
                    text_file.write('The key {} open the next documents {}'.format(Name_key, Filenames))  

            except OSError:
                print('Is not a key {} ❌'.format(str(Key_path))) #! Alert

# ? Decrypt files

def Decrypt_files(**kwargs) -> None: 

    # * General parameters
    Folder_path = kwargs.get('folderpath', None)
    Key_path = kwargs.get('keypath', None)

    # * Folder attribute (ValueError, TypeError)
    if Folder_path == None:
        raise ValueError("Folder does not exist") #! Alert
    if not isinstance(Folder_path, str):
        raise TypeError("Folder must be a string") #! Alert

    Key_dir = os.path.dirname(Key_path)
    Key_file = os.path.basename(Key_path)

    Filename_key, Format = os.path.splitext(Key_file)

    Datetime = datetime.datetime.now()

    with open(Key_path, 'rb') as Filekey:
        Key = Filekey.read()

    Fernet_ = Fernet(Key)

    # * This function sort the files and show them

    if Filename_key.endswith('.key'):

        try:
            for Filename in os.listdir(Folder_path):

                print(Filename)

                with open(Folder_path + '/' + Filename, 'rb') as Encrypted_file: # open in readonly mode
                    Encrypted = Encrypted_file.read()
                
                Decrypted = Fernet_.decrypt(Encrypted)

                with open(Folder_path + '/' + Filename, 'wb') as Decrypted_file:
                    Decrypted_file.write(Decrypted)

            with open(Key_dir + '/' + Key_file + '.txt', "w") as text_file:
                    text_file.write('Key used. Datetime: {} '.format(Datetime))  

        except OSError:
                print('Is not a key {} ❌'.format(str(Key_path))) #! Alert
# ? Decorator

def asterisk_row_print(func):
     
    # added arguments inside the inner1,
    # if function takes any arguments,
    # can be added like this.
    def inner1(*args, **kwargs):
 
        # storing time before function execution
        print("*" * 30)
         
        func(*args, **kwargs)
 
        # storing time after function execution
        print("*" * 30)
 
    return inner1

# ? Detect fi GPU exist in your PC for CNN

def detect_GPU() -> None:
    """
    This function shows if a gpu device is available and its name. This function is good if the training is using a GPU  

    Args:
        None

    Returns:
        None
    """
    GPU_name: string = tf.test.gpu_device_name()
    GPU_available: list = tf.config.list_physical_devices()
    print("\n")
    print(GPU_available)
    print("\n")
    #if GPU_available == True:
        #print("GPU device is available")

    if "GPU" not in GPU_name:
        print("GPU device not found")
        print("\n")
    print('Found GPU at: {}'.format(GPU_name))
    print("\n")

# ? Sort Files

def sort_images(Folder_path: str) -> tuple[list[str], int]: 
    """
    Sort the filenames of the obtained folder path.

    Args:
        Folder_path (str): Folder path obtained.

    Returns:
        list[str]: Return all files sorted.
        int: Return the number of images inside the folder.
    """
    
    # * Folder attribute (ValueError, TypeError)
    if Folder_path == None:
        raise ValueError("Folder does not exist") #! Alert
    if not isinstance(Folder_path, str):
        raise TypeError("Folder must be a string") #! Alert

    Asterisks:int = 60
    # * This function sort the files and show them

    Number_images: int = len(os.listdir(Folder_path))
    print("\n")
    print("*" * Asterisks)
    print('Images: {}'.format(Number_images))
    print("*" * Asterisks)
    Files: list[str] = os.listdir(Folder_path)
    print("\n")

    Sorted_files: list[str] = sorted(Files)

    for Index, Sort_file in enumerate(Sorted_files):
        print('Index: {} ---------- {} ✅'.format(Index, Sort_file))

    print("\n")

    return Sorted_files, Number_images

# ? Get number of images from training, validation, and test.

def Number_images(Folder_paths: list[str, str], Folder_model: str, SI: bool = False) -> None: 
    """
    _summary_

    _extended_summary_

    Args:
        Folder_paths (list[str, str]): _description_
        Folder_model (str): _description_
        Save (_type_): _description_
    """
    # *
    colors = ('red', 'blue')

    # * General parameters
    X_figure_size = 8
    Y_figure_size = 8

    # *
    Name_paths = [None] * len(Folder_paths)
    Dir_paths = [None] * len(Folder_paths)
    Number_images = [None] * len(Folder_paths)

    # *
    Final_name = ''

    Asterisks:int = 60
    
    for i, Path in enumerate(Folder_paths):

        # *
        Name_paths[i] = os.path.basename(Path)
        Dir_paths[i] = os.path.dirname(Path)

        # *
        #Final_name = Final_name + str(Name_paths[i])
        Final_name = '{}_{}'.format(Final_name, Name_paths[i])
        
        # * This function sort the files and show them
        Number_images[i] = len(os.listdir(Path))

        print("\n")
        print("*" * Asterisks)
        print('Images: {}'.format(Number_images[i]))
        print('Path: {}'.format(Path))
        print("*" * Asterisks)

        Files: list[str] = os.listdir(Path)
        print("\n")

        Sorted_files: list[str] = sorted(Files)

        for Index, Sort_file in enumerate(Sorted_files):
            print('Index: {} ---------- {} ✅'.format(Index, Sort_file))

    #print(Name_paths)
    #print(Number_images)

    # *
    plt.figure(figsize = (X_figure_size, Y_figure_size))

    for i in range(len(Name_paths)):
        plt.bar(Name_paths[i], Number_images[i], edgecolor = 'black', color = colors[i], width = 0.4)

    for i, value in enumerate(Number_images):
        plt.text(i, value-(Number_images[i]/10), f'{value}', ha = 'center', fontsize = 16, color = 'white')

    plt.xlabel('Labels')
    plt.ylabel('Number of images')
    plt.title('ROC curve')
    #plt.legend(loc = 'lower right')

    if(SI == True):

        # * Save this figure in the folder given
        Plot_name = 'Plot_Images{}.png'.format(Final_name)
        Plot_folder = os.path.join(Folder_model, Plot_name)

        plt.savefig(Plot_folder)

    plt.show()

    print("\n")

# ? Remove all files in folder

def remove_all_files(Folder_path: str) -> None:
    """
    Remove all files inside the folder path obtained.

    Args:
        Folder_path (str): Folder path obtained.

    Returns:
        None
    """
    
    # * Folder attribute (ValueError, TypeError)
    if Folder_path == None:
        raise ValueError("Folder does not exist") #! Alert
    if not isinstance(Folder_path, str):
        raise TypeError("Folder must be a string") #! Alert

    # * This function will remove all the files inside a folder

    for File in os.listdir(Folder_path):
        Filename, Format  = os.path.splitext(File)
        print('Removing: {} . {} ✅'.format(Filename, Format))
        os.remove(os.path.join(Folder_path, File))

# ? Random remove all files in folder

def random_remove_files(Folder_path: str, Value: int) -> None:
    """
    Remove all files inside the folder path obtained.

    Args:
        Folder_path (str): Folder path obtained.

    Returns:
        None
    """
    # * Folder attribute (ValueError, TypeError)
    if Folder_path == None:
        raise ValueError("Folder does not exist") #! Alert
    if not isinstance(Folder_path, str):
        raise TypeError("Folder must be a string") #! Alert

    # * This function will remove all the files inside a folder
    Files = os.listdir(Folder_path)

        #Filename, Format = os.path.splitext(File)

    for File_sample in sample(Files, Value):
        print(File_sample)
        #print('Removing: {}{} ✅'.format(Filename, Format))
        os.remove(os.path.join(Folder_path, File_sample))

# ? Extract the mean of each column

def extract_mean_from_images(Dataframe:pd.DataFrame, Column:int) -> int:
  """
  Extract the mean from the values of the whole dataset using its dataframe.

  Args:
      Dataframe (pd.DataFrame): Dataframe with all the data needed(Mini-MIAS in this case).
      Column (int): The column number where it extracts the values.
  Returns:
      int: Return the mean from the column values.
  """

  # * This function will obtain the main of each column

  List_data_mean:list = []

  for i in range(Dataframe.shape[0]):
      if Dataframe.iloc[i - 1, Column] > 0:
          List_data_mean.append(Dataframe.iloc[i - 1, Column])

  Mean_list:int = int(np.mean(List_data_mean))
  return Mean_list

# ?

class ChangeFormat:
  """
  _summary_

  _extended_summary_

  Raises:
      ValueError: _description_
      TypeError: _description_
      ValueError: _description_
      TypeError: _description_
      ValueError: _description_
      TypeError: _description_
      ValueError: _description_
      TypeError: _description_
  """
  # * Change the format of one image to another 

  def __init__(self, **kwargs):
    """
    _summary_

    _extended_summary_

    Raises:
        ValueError: _description_
        TypeError: _description_
        ValueError: _description_
        TypeError: _description_
        ValueError: _description_
        TypeError: _description_
        ValueError: _description_
        TypeError: _description_
    """
    # * General parameters
    self.Folder = kwargs.get('Folder', None)
    self.New_folder = kwargs.get('Newfolder', None)
    self.Format = kwargs.get('Format', None)
    self.New_format = kwargs.get('Newformat', None)

    # * Values, type errors.
    if self.Folder == None:
      raise ValueError("Folder does not exist") #! Alert
    if not isinstance(self.Folder, str):
      raise TypeError("Folder attribute must be a string") #! Alert

    if self.New_folder == None:
      raise ValueError("Folder destination does not exist") #! Alert
    if not isinstance(self.New_folder, str):
      raise TypeError("Folder destination attribute must be a string") #! Alert

    if self.Format == None:
      raise ValueError("Current format does not exist") #! Alert
    if not isinstance(self.Format, str):
      raise TypeError("Current format must be a string") #! Alert

    if self.New_format == None:
      raise ValueError("New format does not exist") #! Alert
    if not isinstance(self.New_format, str):
      raise TypeError("Current format must be a string") #! Alert

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

  # * New folder attribute
  @property
  def New_folder_property(self):
      return self.New_folder

  @New_folder_property.setter
  def New_folder_property(self, New_value):
      if not isinstance(New_value, str):
        raise TypeError("Folder must be a string") #! Alert
      self.New_folder = New_value
  
  @New_folder_property.deleter
  def New_folder_property(self):
      print("Deleting folder...")
      del self.New_folder

  # * Format attribute
  @property
  def Format_property(self):
      return self.New_folder

  @Format_property.setter
  def Format_property(self, New_value):
      if not isinstance(New_value, str):
        raise TypeError("Folder must be a string") #! Alert
      self.Format = New_value
  
  @Format_property.deleter
  def New_folder_property(self):
      print("Deleting folder...")
      del self.Format

  # * New Format attribute
  @property
  def New_format_property(self):
      return self.New_format

  @New_format_property.setter
  def New_format_property(self, New_value):
      if not isinstance(New_value, str):
        raise TypeError("Folder must be a string") #! Alert
      self.New_format = New_value
  
  @New_format_property.deleter
  def New_format_property(self):
      print("Deleting folder...")
      del self.New_format

  def ChangeExtension(self):
    """
    _summary_

    _extended_summary_
    """
    # * Changes the current working directory to the given path
    os.chdir(self.Folder)
    print(os.getcwd())
    print("\n")

    # * Using the sort function
    Sorted_files, Total_images = sort_images(self.Folder)
    Count:int = 0

    # * Reading the files
    for File in Sorted_files:
      if File.endswith(self.Format):

        try:
            Filename, Format  = os.path.splitext(File)
            print('Working with {} of {} {} images, {} ------- {} ✅'.format(Count, Total_images, self.Format, Filename, self.New_format))
            #print(f"Working with {Count} of {Total_images} {self.Format} images, {Filename} ------- {self.New_format} ✅")
            
            # * Reading each image using cv2
            Path_file = os.path.join(self.Folder, File)
            Image = cv2.imread(Path_file)         
            #Imagen = cv2.cvtColor(Imagen, cv2.COLOR_BGR2GRAY)
            
            # * Changing its format to a new one
            New_name_filename = Filename + self.New_format
            New_folder = os.path.join(self.New_folder, New_name_filename)

            cv2.imwrite(New_folder, Image)
            #FilenamesREFNUM.append(Filename)
            Count += 1

        except OSError:
            print('Cannot convert {} ❌'.format(str(File))) #! Alert
            #print('Cannot convert %s ❌' % File) #! Alert

    print("\n")
    #print(f"COMPLETE {Count} of {Total_images} TRANSFORMED ✅")
    print('{} of {} tranformed ✅'.format(str(Count), str(Total_images))) #! Alert

# ? Concat multiple dataframes

def concat_dataframe(*dfs: pd.DataFrame, **kwargs: str) -> pd.DataFrame:
  """
  Concat multiple dataframes and name it using technique and the class problem

  Args:
      Dataframe (pd.DataFrames): Multiple dataframes can be entered for concatenation

  Raises:
      ValueError: If the folder variable does not found give a error
      TypeError: _description_
      Warning: _description_
      TypeError: _description_
      Warning: _description_
      TypeError: _description_

  Returns:
      pd.DataFrame: Return the concatenated dataframe
  """
  # * this function concatenate the number of dataframes added

  # * General parameters

  Folder_path = kwargs.get('folder', None)
  Technique = kwargs.get('technique', None)
  Class_problem = kwargs.get('classp', None)
  Save_file = kwargs.get('savefile', False)

  # * Values, type errors and warnings
  if Folder_path == None:
    raise ValueError("Folder does not exist") #! Alert
  if not isinstance(Folder_path, str):
    raise TypeError("Folder attribute must be a string") #! Alert
  
  if Technique == None:
    #raise ValueError("Technique does not exist")  #! Alert
    warnings.warn("Technique does not found, the string 'Without_Technique' will be implemented") #! Alert
    Technique = 'Without_Technique'
  if not isinstance(Technique, str):
    raise TypeError("Technique attribute must be a string") #! Alert

  if Class_problem == None:
    #raise ValueError("Class problem does not exist")  #! Alert
    warnings.warn("Class problem does not found, the string 'No_Class' will be implemented") #! Alert
  if not isinstance(Class_problem, str):
    raise TypeError("Class problem must be a string") #! Alert

  # * Concatenate each dataframe
  ALL_dataframes = [df for df in dfs]
  print(len(ALL_dataframes))
  Final_dataframe = pd.concat(ALL_dataframes, ignore_index = True, sort = False)
      
  #pd.set_option('display.max_rows', Final_dataframe.shape[0] + 1)
  #print(DataFrame)

  # * Name the final dataframe and save it into the given path

  if Save_file == True:
    #Name_dataframe =  str(Class_problem) + '_Dataframe_' + str(Technique) + '.csv'
    Dataframe_name = '{}_Dataframe_{}.csv'.format(str(Class_problem), str(Technique))
    Dataframe_folder_save = os.path.join(Folder_path, Dataframe_name)
    Final_dataframe.to_csv(Dataframe_folder_save)

  return Final_dataframe

# ? Concat multiple dataframes

def concat_dataframe_list(Dataframes, **kwargs: str) -> pd.DataFrame:
  """
  Concat multiple dataframes and name it using technique and the class problem

  Args:
      Dataframe (pd.DataFrames): Multiple dataframes can be entered for concatenation

  Raises:
      ValueError: If the folder variable does not found give a error
      TypeError: _description_
      Warning: _description_
      TypeError: _description_
      Warning: _description_
      TypeError: _description_

  Returns:
      pd.DataFrame: Return the concatenated dataframe
  """
  # * this function concatenate the number of dataframes added

  # * General parameters

  Folder_path = kwargs.get('folder', None)
  Technique = kwargs.get('technique', None)
  Class_problem = kwargs.get('classp', None)
  Save_file = kwargs.get('savefile', False)

  # * Values, type errors and warnings
  if Folder_path == None:
    raise ValueError("Folder does not exist") #! Alert
  if not isinstance(Folder_path, str):
    raise TypeError("Folder attribute must be a string") #! Alert
  
  if Technique == None:
    #raise ValueError("Technique does not exist")  #! Alert
    warnings.warn("Technique does not found, the string 'Without_Technique' will be implemented") #! Alert
    Technique = 'Without_Technique'
  if not isinstance(Technique, str):
    raise TypeError("Technique attribute must be a string") #! Alert

  if Class_problem == None:
    #raise ValueError("Class problem does not exist")  #! Alert
    warnings.warn("Class problem does not found, the string 'No_Class' will be implemented") #! Alert
  if not isinstance(Class_problem, str):
    raise TypeError("Class problem must be a string") #! Alert

  # * Concatenate each dataframe
  print(len(Dataframes))
  Final_dataframe = pd.concat(Dataframes, ignore_index = True, sort = False)
      
  #pd.set_option('display.max_rows', Final_dataframe.shape[0] + 1)
  #print(DataFrame)

  # * Name the final dataframe and save it into the given path

  if Save_file == True:
    #Name_dataframe =  str(Class_problem) + '_Dataframe_' + str(Technique) + '.csv'
    Dataframe_name = '{}_Dataframe_{}.csv'.format(str(Class_problem), str(Technique))
    Dataframe_folder_save = os.path.join(Folder_path, Dataframe_name)
    Final_dataframe.to_csv(Dataframe_folder_save)

  return Final_dataframe

# ? Split folders into train/test/validation

def split_folders_train_test_val(Folder_path:str, Only_train_test: bool) -> str:
  """
  Create a new folder with the folders of the class problem and its distribution of training, test and validation.
  The split is 80 and 20. If there is a validation set, it'll be 80, 10, and 10.

  Args:
      Folder_path (str): Folder's dataset for distribution

  Returns:
      None
  """
  # * General parameters

  Asterisks: int = 50
  Train_split: float = 0.8
  Test_split: float = 0.1
  Validation_split: float = 0.1

  #Name_dir = os.path.dirname(Folder)
  #Name_base = os.path.basename(Folder)
  #New_Folder_name = Folder_path + '_Split'

  New_Folder_name = '{}_Split'.format(Folder_path)

  print("*" * Asterisks)
  print('New folder name: {}'.format(New_Folder_name))
  print("*" * Asterisks)

  #1337
  
  try:

    if(Only_train_test == False):

      splitfolders.ratio(Folder_path, output = New_Folder_name, seed = 22, ratio = (Train_split, Test_split, Validation_split)) 
    
    else:

      Test_split: float = 0.2
      splitfolders.ratio(Folder_path, output = New_Folder_name, seed = 22, ratio = (Train_split, Test_split)) 

  except OSError as e:
    print('Cannot split the following folder {}, Type error: {} ❌'.format(str(Folder_path), str(type(e)))) #! Alert

  return New_Folder_name

# ?

class BarChart:
  """
  _summary_

  _extended_summary_
  """
  def __init__(self, **kwargs) -> None:
    """
    _summary_

    _extended_summary_
    """
  
    self.CSV_path = kwargs.get('csv', None)
    self.Folder_path_save = kwargs.get('foldersave', None)
    self.Plot_title = kwargs.get('title', None)
    self.Plot_x_label = kwargs.get('label', None)
    self.Plot_column = kwargs.get('column', None)
    self.Plot_reverse = kwargs.get('reverse', None)
    self.Num_classes = kwargs.get('classes', None)

  def barchart_horizontal(self) -> None:
    """
	  Show CSV's barchar of all models

    Parameters:
    argument1 (folder): CSV that will be used.
    argument2 (str): Title name.
    argument3 (str): Xlabel name.
    argument1 (dataframe): Dataframe that will be used.
    argument2 (bool): if the value is false, higher values mean better, if the value is false higher values mean worse.
    argument3 (folder): Folder to save the images.
    argument3 (int): What kind of problem the function will classify

    Returns:
	  void
   	"""
    # * General parameters
    X_figure_size = 22
    Y_figure_size = 24
    Font_size_title = 40
    Font_size_general = 25
    Font_size_ticks = 15

    # * General lists

    X_fast_list_values = []
    X_slow_list_values = []

    Y_fast_list_values = []
    Y_slow_list_values = []

    X_fastest_list_value = []
    Y_fastest_list_value = []

    X_slowest_list_value = []
    Y_slowest_list_value = []
    
    Colors = ('gray', 'red', 'blue', 'green', 'cyan', 'magenta', 'indigo', 'azure', 'tan', 'purple')

    # * Chosing label
    if self.Num_classes == 2:
      Label_class_name = 'Biclass_'
    elif self.Num_classes > 2:
      Label_class_name = 'Multiclass_'

    # Initialize the lists for X and Y
    #data = pd.read_csv("D:\MIAS\MIAS VS\DataCSV\DataFrame_Binary_MIAS_Data.csv")

    # * Read dataframe csv
    Dataframe = pd.DataFrame(self.CSV_path)

    # * Get X and Y values
    X = list(Dataframe.iloc[:, 0])
    Y = list(Dataframe.iloc[:, self.Plot_column])

    plt.figure(figsize = (X_figure_size, Y_figure_size))

    # * Reverse is a bool variable with the postion of the plot
    if self.Plot_reverse == True:

        for Index, (i, k) in enumerate(zip(X, Y)):
            if k < np.mean(Y):
                X_fast_list_values.append(i)
                Y_fast_list_values.append(k)
            elif k >= np.mean(Y):
                X_slow_list_values.append(i)
                Y_slow_list_values.append(k)

        for Index, (i, k) in enumerate(zip(X_fast_list_values, Y_fast_list_values)):
            if k == np.min(Y_fast_list_values):
                X_fastest_list_value.append(i)
                Y_fastest_list_value.append(k)
                #print(X_fastest_list_value)
                #print(Y_fastest_list_value)

        for Index, (i, k) in enumerate(zip(X_slow_list_values, Y_slow_list_values)):
            if k == np.max(Y_slow_list_values):
                X_slowest_list_value.append(i)
                Y_slowest_list_value.append(k)
    else:

        for Index, (i, k) in enumerate(zip(X, Y)):
            if k < np.mean(Y):
                X_slow_list_values.append(i)
                Y_slow_list_values.append(k)
            elif k >= np.mean(Y):
                X_fast_list_values.append(i)
                Y_fast_list_values.append(k)

        for Index, (i, k) in enumerate(zip(X_fast_list_values, Y_fast_list_values)):
            if k == np.max(Y_fast_list_values):
                X_fastest_list_value.append(i)
                Y_fastest_list_value.append(k)
                #print(XFastest)
                #print(YFastest)

        for Index, (i, k) in enumerate(zip(X_slow_list_values, Y_slow_list_values)):
            if k == np.min(Y_slow_list_values):
                X_slowest_list_value.append(i)
                Y_slowest_list_value.append(k)

                
    # * Plot the data using bar() method
    plt.barh(X_slow_list_values, Y_slow_list_values, label = "Bad", color = 'gray')
    plt.barh(X_slowest_list_value, Y_slowest_list_value, label = "Worse", color = 'black')
    plt.barh(X_fast_list_values, Y_fast_list_values, label = "Better", color = 'lightcoral')
    plt.barh(X_fastest_list_value, Y_fastest_list_value, label = "Best", color = 'red')

    for Index, value in enumerate(Y_slowest_list_value):
        plt.text(0, 68, 'Worse value: {} -------> {}'.format(str(value), str(X_slowest_list_value[0])), fontweight = 'bold', fontsize = Font_size_general)

    for Index, value in enumerate(Y_fastest_list_value):
        plt.text(0, 70, 'Worse value: {} -------> {}'.format(str(value), str(X_fastest_list_value[0])), fontweight = 'bold', fontsize = Font_size_general)

    plt.legend(fontsize = Font_size_general)

    plt.title(self.Plot_title, fontsize = Font_size_title)
    plt.xlabel(self.Plot_x_label, fontsize = Font_size_general)
    plt.xticks(fontsize = Font_size_ticks)
    plt.ylabel("Models", fontsize = Font_size_general)
    plt.yticks(fontsize = Font_size_ticks)
    plt.grid(color = Colors[0], linestyle = '-', linewidth = 0.2)

    # * Name graph and save it
    Graph_name = '{}_Dataframe_{}.png'.format(str(Label_class_name), str(self.Plot_title))
    Graph_name_folder = os.path.join(self.Folder_path_save, Graph_name)

    plt.savefig(Graph_name_folder)


  def barchart_vertical(self) -> None:  
    pass


# ? test_figure_plot

def test_figure_plot(Height: int, Width: int, Annot_kws: int, font: int, CM_dataframe: pd.DataFrame, History_dataframe: pd.DataFrame, ROC_dataframe: pd.DataFrame) -> None: 

  # *
  X_size_figure = 2
  Y_size_figure = 2

  # *
  Confusion_matrix_dataframe = pd.read_csv(CM_dataframe)
  History_data_dataframe = pd.read_csv(History_dataframe)
  
  # *
  Accuracy = History_data_dataframe.accuracy.to_list()
  Loss = History_data_dataframe.loss.to_list()
  Val_accuracy = History_data_dataframe.val_accuracy.to_list()
  Val_loss = History_data_dataframe.val_loss.to_list()

  # *
  print(Loss)
  print(Val_loss)

  #Column_names = ["FPR", "TPR"]
  Roc_curve_dataframe = pd.read_csv(ROC_dataframe)
  
  # *
  FPR = Roc_curve_dataframe.FPR.to_list()
  TPR = Roc_curve_dataframe.TPR.to_list()

  print(FPR)

  # * Figure's size
  plt.figure(figsize = (Width, Height))
  plt.suptitle('MobileNet', fontsize = 20)
  plt.subplot(X_size_figure, Y_size_figure, 4)
  sns.set(font_scale = font)

  # * Confusion matrix heatmap
  ax = sns.heatmap(Confusion_matrix_dataframe, annot = True, fmt = 'd', annot_kws = {"size": Annot_kws})
  #ax.set_title('Seaborn Confusion Matrix with labels\n\n')
  ax.set_xlabel('\nPredicted Values')
  ax.set_ylabel('Actual Values')

  # * Subplot Training accuracy
  plt.subplot(X_size_figure, Y_size_figure, 1)
  plt.plot(Accuracy, label = 'Training Accuracy')
  plt.plot(Val_accuracy, label = 'Validation Accuracy')
  plt.ylim([0, 1])
  plt.legend(loc = 'lower right')
  plt.title('Training and Validation Accuracy')
  plt.xlabel('Epoch')

  plt.subplot(X_size_figure, Y_size_figure, 2)
  plt.plot(Loss, label = 'Training Loss')
  plt.plot(Val_loss, label = 'Validation Loss')
  plt.ylim([0, 2.0])
  plt.legend(loc = 'upper right')
  plt.title('Training and Validation Loss')
  plt.xlabel('Epoch')

  # * FPR and TPR values for the ROC curve
  Auc = auc(FPR, TPR)

  # * Subplot ROC curve
  plt.subplot(X_size_figure, Y_size_figure, 3)
  plt.plot([0, 1], [0, 1], 'k--')
  plt.plot(FPR, TPR, label = 'Test' + '(area = {:.4f})'.format(Auc))
  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.title('ROC curve')
  plt.legend(loc = 'lower right')

  plt.show()

  # * Figure's size
  
  plt.figure(figsize = (Width/2, Height/2))
  plt.title('MobileNet', fontsize = 20)
  sns.set(font_scale = font)

  # * Confusion matrix heatmap
  ax = sns.heatmap(Confusion_matrix_dataframe, annot = True, fmt = 'd', annot_kws = {"size": Annot_kws})
  #ax.set_title('Seaborn Confusion Matrix with labels\n\n')
  ax.set_xlabel('\nPredicted Values')
  ax.set_ylabel('Actual Values')

  plt.show()