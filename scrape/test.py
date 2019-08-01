import os

directory = "transcripts"

for filename in os.listdir(directory):
    file_location = directory + "/" + filename
    with open(file_location, "r+") as f:
        not_to_remove = True
        d = f.readlines()
        f.seek(0)
        for i in d:
            if i == " Like this article" or i == "Like this article" or i == " Like this article\n" or i == "Like this article\n" or i == "Like this article  " or i == "Like this article  \n":
                not_to_remove = False
                print("HIER")

            if "Like this article" in i:
                not_to_remove = False
            if not_to_remove:
                f.write(i)
        f.truncate()
        f.close()