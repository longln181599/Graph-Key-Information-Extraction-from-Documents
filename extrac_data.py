# unzip file by python 
import zipfile
import os
import shutil
import glob
 # unzip 
def unzip(file_name):
    zip_ref = zipfile.ZipFile(file_name, 'r')
    zip_ref.extractall(os.path.dirname(file_name))
    zip_ref.close()
    return
unzip("./raw_sroie/interim.zip")