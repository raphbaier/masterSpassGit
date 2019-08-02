import os
import json

DIRECTORY_NAME_TRANSCRIPTS = "../scrape/transcripts/"
DIRECTORY = os.fsencode(DIRECTORY_NAME_TRANSCRIPTS)

DIRECTORY_NAME_ALIGNMENTS = "../forced_alignment/alignments/"

counter = 1

#the possible titles of the list of names of company members
IMPORTANT_NAMES = ["Company Participants", "Executives", "Company Representatives", "Corporate Participants"]
#list of important names ends here.
UNIMPORTANT_NAMES = ["Analysts", "Conference Call Participants", "Analyst"]
#End searching for names at the last name, operator
OPERATOR_NAMES = ["Operator"]


def get_important_names_from_transcript(transcript):
    found_important_start = False
    found_unimportant_start = False
    names_list_filled = False
    list_of_important_names = []
    list_of_unimportant_names = []

    for line in transcript.readlines():
        if not names_list_filled:
            if found_important_start:

                for name in UNIMPORTANT_NAMES:
                    if name in line:
                        found_unimportant_start = True
                        found_important_start = False

                if not found_unimportant_start:
                    if line != '\n':
                        name_to_add = line.split('-')[0]
                        if name_to_add[-1] == ' ':
                            name_to_add = name_to_add[:-1]
                        list_of_important_names.append(name_to_add)

            elif found_unimportant_start:
                for name in OPERATOR_NAMES:
                    if name in line:
                        names_list_filled = True
                if not names_list_filled:
                    if line != '\n':
                        name_to_add = line.split('-')[0]
                        if name_to_add[-1] == ' ':
                            name_to_add = name_to_add[:-1]
                        list_of_unimportant_names.append(name_to_add)

            else:
                for name in IMPORTANT_NAMES:
                    if name in line:
                        # we found people belonging to the company
                        found_important_start = True

    return [list_of_important_names, list_of_unimportant_names]



class Sound_Data:

    def __init__(self, transcript_file):
        with open(DIRECTORY_NAME_TRANSCRIPTS + transcript_file + '.txt', 'r') as transcript:
            self.names = get_important_names_from_transcript(transcript)
        with open(DIRECTORY_NAME_ALIGNMENTS + transcript_file + '_alignment.json', 'r') as alignment:
            self.alignment = json.load(alignment)

        all_words_in_alignment = []

        transcript_started = False
        important_started = False


        self.important_times = []
        in_important_time = False
        important_time = []

        counter = 0
        for word in self.alignment['words']:
            transcrib_word = word['word']
            if not transcript_started:
                if transcrib_word == 'Operator':
                    transcript_started = True
            else:
                all_words_in_alignment.append(word)



                #look for important names, i.e. and important speaker starts to speak from here
                important_found = False
                for name in self.names[0]:
                    if transcrib_word.lower() == name.split(" ")[0].replace(".", "").lower():
                        name_counter = 0
                        important_found = True
                        for subname in name.split(" "):
                            if self.alignment['words'][counter+name_counter]['word'].lower() != subname.replace('.', '').lower():
                                important_found = False
                            name_counter += 1
                if important_found:
                    #we found a new important name but are already in another important name
                    if in_important_time:
                        #end current time: go back to the latest timestamp
                        counter_timestamp = counter - 1
                        while self.alignment['words'][counter_timestamp]['case'] != "success":
                            counter_timestamp -= 1
                        important_time.append(self.alignment['words'][counter_timestamp]['end'])
                        self.important_times.append(important_time)
                        important_time = []
                    #we start a new important time
                    #start current time: go forward to the newest timestamp
                    counter_timestamp = counter + 1
                    while self.alignment['words'][counter_timestamp]['case'] != "success":
                        counter_timestamp += 1
                    important_time.append(self.alignment['words'][counter_timestamp]['start'])
                    in_important_time = True

                unimportant_found = False
                for name in self.names[1] + OPERATOR_NAMES:
                    if transcrib_word.lower() == name.split(" ")[0].replace(".", "").lower():
                        name_counter = 0
                        unimportant_found = True
                        for subname in name.split(" "):
                            if self.alignment['words'][counter + name_counter]['word'].lower() != subname.replace('.',
                                                                                                                  '').lower():
                                unimportant_found = False
                            name_counter += 1
                if unimportant_found:
                    #we found an important name and are in an important name: end current time: go back to the latest timestamp
                    if in_important_time:
                        counter_timestamp = counter - 1
                        while self.alignment['words'][counter_timestamp]['case'] != "success":
                            counter_timestamp -= 1
                        important_time.append(self.alignment['words'][counter_timestamp]['end'])
                        self.important_times.append(important_time)
                        important_time = []
                        in_important_time = False
            counter += 1
        #print(self.important_times)
        print("Hallo Feli")

    def get_names(self):
        return self.names

    def get_important_times(self):
        return self.important_times




for file in os.listdir(DIRECTORY):


    filename = os.fsdecode(file)[:-4]
    new_alignment = Sound_Data(filename)
    #print(filename)
    #print(new_alignment.get_names())
    print(new_alignment.get_important_times())