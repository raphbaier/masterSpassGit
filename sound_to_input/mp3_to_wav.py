import os
from pydub import AudioSegment
#sound = AudioSegment.from_mp3("/path/to/file.mp3")
#sound.export("/output/path/file.wav", format="wav")

DIRECTORY_NAME = "/media/raphael/MasterSpass/masterSpass/mp3/"
DIRECTORY_OUT_NAME = "/media/raphael/MasterSpass/masterSpass/wav/"

DIRECTORY = os.fsencode(DIRECTORY_NAME)

for file in os.listdir(DIRECTORY):
    filename = DIRECTORY_NAME + os.fsdecode(file)[:-4]
    print(filename)
    sound = AudioSegment.from_mp3(filename + ".mp3")
    sound.export(DIRECTORY_OUT_NAME + os.fsdecode(file)[:-4] + ".wav", format = "wav")