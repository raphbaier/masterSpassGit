import os

directory = "transcripts"

for filename in os.listdir(directory):
    file_location = directory + "/" + filename
    with open(file_location) as transcript_to_check:
        transcript_data = transcript_to_check.read()
        transcript_title = ""
        if os.stat(file_location).st_size != 0 and os.stat(file_location).st_size != 1:
            transcript_title = transcript_data.split("\n")[0]

        keep_string = True
        if ("(" + filename.split("-")[0] + ") ") not in transcript_title:
            keep_string = False
        if (filename.split("-")[2][:-4] + " " + filename.split("-")[1] + " Results - Earnings Call Transcript") not in transcript_title:
            keep_string = False
        if len(transcript_data) < 10000:
            keep_string = False


        if "Accenture" in transcript_title or "EarningsCallTranscript" in transcript_title:
            keep_string = True
        if "F1Q" in transcript_title:
            keep_string = True
        if "F2Q" in transcript_title:
            keep_string = True
        if "F3Q" in transcript_title:
            keep_string = True
        if "F4Q" in transcript_title:
            keep_string = True
        if "AMVMF" in transcript_title:
            keep_string = True

        if keep_string:
            pass
        else:
            #os.remove(file_location)
            print("Removed a file.")
            print(filename)
            print(transcript_title)