import os
import json
from pydub import AudioSegment
import numpy as np
import Levenshtein

DIRECTORY_NAME_TRANSCRIPTS = "/media/raphael/MasterSpass/masterSpass/scrape/transcripts/"

DIRECTORY = os.fsencode(DIRECTORY_NAME_TRANSCRIPTS)

DIRECTORY_NAME_ALIGNMENTS = "/media/raphael/MasterSpass/masterSpass/forced_alignment/alignments/"


DIRECTORY_NAME_MP3 = "/media/raphael/MasterSpass/masterSpass/mp3/"


DIRECTORY_NAME_OUTPUT = "/media/raphael/MasterSpass/masterSpass/wav/"


DIRECTORY = os.fsencode(DIRECTORY_NAME_MP3)

# length of a sound segment in miliseconds
SEGMENT_LENGTH = 5000

counter = 1

#the possible titles of the list of names of company members
IMPORTANT_NAMES = ["Company Participants", "Executives", "Company Representatives", "Corporate Participants"]
#list of important names ends here.
UNIMPORTANT_NAMES = ["Analysts", "Conference Call Participants", "Analyst"]
#End searching for names at the last name, operator
OPERATOR_NAMES = ["Operator"]


def letters(input):
    valids = []
    for character in input:
        if character.isalpha():
            valids.append(character)
    return ''.join(valids)


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

    print(list_of_important_names)
    print(list_of_unimportant_names)
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

        transcript_words = self.alignment['transcript']
        to_replace = [".", "|", ":", ",", ";", "_", "-", "(", ")", "=", "?", "!", "'", "    ", "   ", "  "]
        for symbol in to_replace:
            transcript_words = transcript_words.replace(symbol, " ")
        #print(transcript_words)
        transcript_words = transcript_words.replace("\n\n", "\n").replace("\n", " \n ")
        transcript_words = transcript_words.split(" ")
        print(transcript_words)

        position_in_transcript = 0
        window_size = 3

        false_words_counter = 0
        for word in self.alignment['words']:
            transcrib_word = word['word']

            #get current position in transcript
            position_in_transcript = min(len(transcript_words)-1, position_in_transcript+1)
            smallest_distance = Levenshtein.distance(transcript_words[position_in_transcript], transcrib_word)
            start_pos = max(0, position_in_transcript - window_size)
            end_pos = min(position_in_transcript + window_size, len(transcript_words))
            new_position = 0
            current_position = start_pos
            for word in transcript_words[start_pos:end_pos]:
                current_distance = Levenshtein.distance(word, transcrib_word)
                if current_distance <= smallest_distance:
                    smallest_distance = current_distance
                    new_position = current_position
                current_position += 1
            position_in_transcript = new_position

            if transcrib_word != transcript_words[position_in_transcript]:
                false_words_counter += 1
            else:
                false_words_counter = 0
            #print("JETZABER")
            #print(transcrib_word)
            #print(transcript_words[position_in_transcript])

            if not transcript_started:
                if transcrib_word == 'Operator':
                    transcript_started = True

                    #print(transcrib_word)
            else:
                all_words_in_alignment.append(word)

                #look for important names, i.e. an important speaker starts to speak from here
                important_found = False
                for name in self.names[0]:
                    #print(name)
                    if transcrib_word.lower() == name.split(" ")[0].replace(".", "").lower():
                        name_counter = 0
                        important_found = True
                        for subname in name.split(" "):

                            name_position = min(len(self.alignment['words']) - 1, counter + name_counter)
                            if self.alignment['words'][name_position]['word'].lower() != subname.replace('.', '').lower():
                                important_found = False
                            name_counter += 1

                        next_paragraph_position = min(len(transcript_words)-1, position_in_transcript + name_counter)
                        before_paragraph_position = max(0, position_in_transcript - 1)

                        #if transcript_words[next_paragraph_position] != "\n" or transcript_words[before_paragraph_position] != "\n":
                        if transcript_words[before_paragraph_position] != "\n":
                            important_found = False



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
                    while self.alignment['words'][counter_timestamp]['case'] != "success" and counter_timestamp < len(self.alignment['words'])-1:
                        counter_timestamp += 1
                    try:
                        important_time.append(self.alignment['words'][counter_timestamp]['start'])
                        in_important_time = True
                    except Exception as e:
                        print(e)

                unimportant_found = False
                for name in self.names[1] + OPERATOR_NAMES:
                    if transcrib_word.lower() == name.split(" ")[0].replace(".", "").lower():
                        name_counter = 0
                        unimportant_found = True
                        for subname in name.split(" "):

                            name_position = min(len(self.alignment['words'])-1, counter+name_counter)
                            if self.alignment['words'][name_position]['word'].lower() != subname.replace('.', '').lower():
                                unimportant_found = False
                            name_counter += 1

                        next_paragraph_position = min(len(transcript_words) - 1, position_in_transcript + name_counter)
                        before_paragraph_position = max(0, position_in_transcript - 1)

                        #if transcript_words[next_paragraph_position] != "\n" or transcript_words[before_paragraph_position] != "\n":
                        if transcript_words[before_paragraph_position] != "\n":
                            unimportant_found = False

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

        if false_words_counter > 10:
            print("Da stimmt was net!!")
            print("DA STIMMT WAS NET")
            print("DA STIMMT WAS NET")
            print("DA STIMMT WAS NET")
            print("DA STIMMT WAS NET")
            print("DA STIMMT WAS NET")
            print(transcript_file)

    def get_names(self):
        return self.names

    def get_important_times(self):
        return self.important_times




counter = 0
for file in os.listdir(DIRECTORY):
    #sonntag, 19:25 gestartet für 300
    if counter < 600:
        filename = os.fsdecode(file)[:-4]
        new_alignment = Sound_Data(filename)
        print(DIRECTORY_NAME_MP3 + filename + ".mp3")

        new_dir = DIRECTORY_NAME_OUTPUT + filename + "/"

        if not os.path.isdir(new_dir):

            os.mkdir(new_dir)

            dir_wav = new_dir + "wav/"
            dir_arff = new_dir + "arff/"

            os.mkdir(dir_wav)
            os.mkdir(dir_arff)

            #print(filename)
            #print(new_alignment.get_names())
            print(new_alignment.get_important_times())
            splits = np.array(new_alignment.get_important_times())
            splits = np.multiply(splits, 1000)

            sound_file = AudioSegment.from_mp3(DIRECTORY_NAME_MP3 + filename + ".mp3")

            counter2 = 0

            for start, end in splits:
                x = start
                while x < end:
                    y = x + SEGMENT_LENGTH
                    if y > end:
                        y = end
                    new_file = sound_file[x : y]
                    new_file.export(dir_wav + str(counter2) + ".wav", format="wav")
                    x = y

                    os.system("SMILExtract -C emobase2010.conf -I "
                              + dir_wav + str(counter2) + ".wav" + " -O "
                              + dir_arff + str(counter2) + ".arff" + " -l 0")



                    counter2 += 1
        #print(splits)
        counter += 1

        #TODO: Alle mit 0 anschauen. hier steht evtl noch der titel hinter der überschrift. generell alle mit auffällig wenigen
        #TODO: evtl werden auch die important und unimportant names nicht richtig ausgelesen, zb weil zwei "-" hinter einem namen stehen, oder "," oder "&"