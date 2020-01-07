import os

#print out change of EPS, Revenue and year over year

TRANSCRIPTS_DIRECTORY = "/media/raphael/masterspass2/masterSpass/scrape/transcripts/"

TRANSCRIPTS_DIRECTORY = os.fsencode(TRANSCRIPTS_DIRECTORY)

shit_counter = 0

for file in os.listdir(TRANSCRIPTS_DIRECTORY):
    print(file[:-4])
    with open(TRANSCRIPTS_DIRECTORY + file, "r") as to_read:
        lines = to_read.readlines()

        line = ""
        i = 0
        while i < 8:
            if "EPS of" in lines[i]:
                line = lines[i]
            i += 1

        print(line)
    to_read.close()
    split_line = line.split(" ")
    percentage_EPS = 0
    percentage_Rev = 0
    year_over_year = 0
    try:
        EPS_planned = split_line[2].replace("$", "")
        EPS_beat = split_line[5].replace("$", "")
        Rev_planned = split_line[8].replace("$", "")
        Rev_beat = split_line[13].replace("$", "")
        year_over_year = split_line[9].replace("(", "")
        print(EPS_planned + " " + EPS_beat + " " + Rev_planned + " " + Rev_beat + " " + year_over_year)

        percentage_EPS = float(EPS_beat) / float(EPS_planned)
        print(percentage_EPS)

        Rev_planned_number = float(Rev_planned.replace("B", "").replace("M", ""))

        if "B" in Rev_planned:
            Rev_planned_number *= 1000000000
        elif "M" in Rev_planned:
            Rev_planned_number *= 1000000
        Rev_beat_number = float(Rev_beat.replace("B", "").replace("M", ""))
        if "B" in Rev_beat:
            Rev_beat_number *= 1000000000
        elif "M" in Rev_beat:
            Rev_beat_number *= 1000000

        percentage_Rev = Rev_beat_number/Rev_planned_number

        year_over_year = float(year_over_year.replace("%", ""))/100




    except:
        shit_counter += 1
        print("shit")
        print(shit_counter)


    filename = file[:-4].decode('utf-8')
    try:
        with open("/media/raphael/masterspass2/masterSpass/wav_sentences/" + filename + "/earnings_numbers.txt", "w") as file_to_write:
            file_to_write.write(str(percentage_EPS) + "  " + str(percentage_Rev) + " " + str(year_over_year))
        file_to_write.close()
    except:
        print("ops, file not found.")
