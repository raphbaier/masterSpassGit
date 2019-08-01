import os
import subprocess

directory_name = "../scrape/transcripts/"
extern_directory_name = "/media/raphael/MasterSpass/dwhelper-EXTERN/"
directory = os.fsencode(directory_name)



for file in os.listdir(directory):
     filename = os.fsdecode(file)
     text_file = directory_name + filename
     print(text_file)
     sound_file = extern_directory_name + filename[:-3] + "mp3"
     print(sound_file)
     if not os.path.isfile("alignments/" + filename[:-4] + "_alignment.txt"):
          print(filename[:-4])
          os.system("python3 ../gentle/align.py " + sound_file + " " + text_file + " --output alignments/" + filename[:-4] + "_alignment.txt")



"""
ssh = subprocess.Popen(['ssh', 'rbaier@login.coli.uni-saarland.de', 'Mynamemyname!2', '/proj/courses.shadow/theses/rbaier/dwhelper-EXTERN/files.txt'],
                       stdout=subprocess.PIPE)
for line in ssh.stdout:
    print(line)  # do stuff
"""