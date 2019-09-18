import os
import arff
from shutil import copyfile

ARFF_FILES_PATH = "../../one_arff_with_label/"

ARFF_FILES_PATH_SORTED = "../../one_arff_with_label_sorted/"

for file in os.listdir(ARFF_FILES_PATH):
    with open(ARFF_FILES_PATH + file) as fh:
        file_to_check = arff.load(fh)
        emotion = file_to_check["data"][0][-1]
        folder_to_save = ARFF_FILES_PATH_SORTED
        if emotion == "higher":
            folder_to_save += "higher/"
        else:
            folder_to_save += "lower/"
        copyfile(ARFF_FILES_PATH + file, folder_to_save + file)
        print("copied to " + folder_to_save + file + ".")
    fh.close()