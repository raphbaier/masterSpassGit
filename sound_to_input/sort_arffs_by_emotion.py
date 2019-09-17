import os
import arff

ARFF_FILES_PATH = "../../one_arff_with_label/"


for file in os.listdir(ARFF_FILES_PATH):
    with open(ARFF_FILES_PATH + file) as fh:
        file_to_check = arff.load(fh)
        print(file_to_check["data"][0])
    fh.close()