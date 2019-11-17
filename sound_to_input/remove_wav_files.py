import os
import shutil

#shutil.rmtree('/media/raphael/masterspass2/masterSpass/wav_words/AMAT-2018-Q4/', ignore_errors=True)

#this script is for removing the .wav-files to save memory. So far, we only kept them for debugging purposes

FOLDER_TO_REMOVE_PATHS = "/media/raphael/masterspass2/masterSpass/wav_words/"

for file in os.listdir(FOLDER_TO_REMOVE_PATHS):
    print(file)
    if os.path.isdir(FOLDER_TO_REMOVE_PATHS + file + "/wav/"):
        shutil.rmtree(FOLDER_TO_REMOVE_PATHS + file + "/wav/")
    #for file_to_remove in os.listdir(FOLDER_TO_REMOVE_PATHS + file + "/wav/"):
    #    os.remove(FOLDER_TO_REMOVE_PATHS + file + "/wav/" + file_to_remove)