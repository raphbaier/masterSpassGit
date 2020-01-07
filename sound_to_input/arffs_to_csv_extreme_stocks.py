import os
import pandas as pd
import yfinance as yf
import os
from datetime import datetime
from datetime import timedelta
import arff
from scipy import stats

#from model.cnn.part6_earnings import get_mfcc_from_arff

import numpy as np

arff_files = "/media/raphael/masterspass2/masterSpass/wav_sentences"

DIRECTORY_TRANSCRIPTS = "/media/raphael/masterspass2/masterSpass/scrape/transcripts/"


DAYS_BEFORE = 1
DAYS_AFTER = 1

emotion_m = []
path_m = []

emotion_w = []
path_w = []

stock_growth_list_m = []
stock_growth_list_w = []

paths_list_m = []
paths_list_w = []

#proportion of highest and lowest stocks we want to keep --- 0.5 is the maximum
proportion = 0.3

onlyQAndA = False
onlyExtremes = False
onlyOutliers = False

#Considering the presented numbers at the earnings talk: 0 for not considering, 1 for EPS
#deviation, 2 for revenue deviation, 3 for year-over-year deviation
consider_earnings_numbers = 0

#TODO: import aus model
def get_mfcc_from_arff(file):
    # get mfcc features from arff
    try:
        with open(file) as arff_opened:
            arff_file = arff.load(arff_opened)
        arff_opened.close()
    except:
        with open(
                "/home/raphael/masterSpassGit/model/cnn/input_arff/female/disgust/_03-01-07-01-01-01-02.arff") as file2:
            arff_file = arff.load(file2)
        file2.close()

    counter = 0
    derived = True
    mfcc_list = []

    last_counter = "0"
    new_mfcc = []
    for attribute in arff_file['attributes']:
        attribute_found = False
        if 'mfcc_sma' in attribute[0]:
            if not derived:
                if not "sma_de" in attribute[0]:
                    attribute_found = True
            else:
                attribute_found = True

        if attribute_found:
            if len(arff_file['data']) > 0:
            #if counter < len(arff_file['data'][0]):
                new_mfcc.append(arff_file['data'][0][counter])
            else:
                return None
            """
            attributes = attribute[0].replace("]", "[").split("[")
            if attributes[1] != last_counter:

                mfcc_list.append(new_mfcc)
                last_counter = attributes[1]
                new_mfcc = []
                new_mfcc.append(arff_file['data'][0][counter])

            else:
                new_mfcc.append(arff_file['data'][0][counter])"""
        counter += 1
    new_mfcc = np.array(new_mfcc)
    new_mfcc = new_mfcc.reshape((30, 21))
    mfcc_list.append(new_mfcc)

    return mfcc_list



counter = 0
#counter < 100
for file in os.listdir(arff_files):
    if 0 < counter < 1400: # 100 < counter < 300
        stock_name = file.split("-")[0]


        print(file)

        # find out the date of the earnings call
        with open(DIRECTORY_TRANSCRIPTS + file + ".txt") as transcript:
            date_name = transcript.readlines()[2]
        transcript.close()
        print(stock_name)
        stock_growth = 0
        if "Earnings" in date_name:
            date = date_name.split(" ")[1]
            if date.count("-") == 2:
                try:
                    ticker = yf.Ticker(stock_name)
                    if ticker.info['quoteType'] == "EQUITY":
                        datetime_object = datetime.strptime(date, '%m-%d-%y')
                        start_date = datetime_object - timedelta(days=DAYS_BEFORE)
                        end_date = datetime_object + timedelta(days=DAYS_AFTER)
                        stock_data = yf.download(stock_name, start_date.date(), end_date.date())
                        first_course = stock_data.Close[0]
                        second_course = stock_data.Close[-1]
                        stock_growth = (second_course / first_course) - 1
                        print(stock_growth)


                        if consider_earnings_numbers != 0:
                            with open(arff_files + "/" + file + "/" + "earnings_numbers.txt", "r") as earnings_numbers:
                                deviations = earnings_numbers.read().split(" ")
                            earnings_numbers.close()
                            #the deviation from the "normal" stock course
                            stock_growth = stock_growth - float(deviations[consider_earnings_numbers-1])


                except KeyError:
                    print("Couldn't find symbol.")

        #start from here for the q&a session
        qanda = 0
        try:
            with open(arff_files + "/" + file + "/qAndACount.txt", "r") as qaFile:
                qanda = int(qaFile.read())
            qaFile.close()

            for arff_file in os.listdir(arff_files + "/" + file + "/arff"):
                infos = arff_file.split("_")
                if int(infos[0]) > qanda or not onlyQAndA:
                    if infos[1] == "m":
                        #path_m.append(arff_files + "/" + file + "/arff/" + arff_file)
                        stock_growth_list_m.append(stock_growth)
                        paths_list_m.append(arff_files + "/" + file + "/arff/" + arff_file)
                        #if stock_growth > 0:
                        #    emotion_m.append("POSITIVE")
                        #else:
                        #    emotion_m.append("NEGATIVE")
                    #else:
                        ##path_w.append(arff_files + "/" + file + "/arff/" + arff_file)
                        #stock_growth_list_w.append(stock_growth)
                        #paths_list_w.append(arff_files + "/" + file + "/arff/" + arff_file)
                        ##if stock_growth > 0:
                        ##    emotion_w.append("POSITIVE")
                        ##else:
                        ##    emotion_w.append("NEGATIVE")
                ##emotion.append(stock_growth)
        except:
            print("file not found. Probably no Q&A section found or no earnings numbers for earnings consideration.")
    counter += 1

### only the highest and lowest stock courses
if not onlyExtremes:
    proportion = 0.5

print(stock_growth_list_m)
print(paths_list_m)

stock_growth_list_m = np.array(stock_growth_list_m)
paths_list_m = np.array(paths_list_m)

ordering_m = stock_growth_list_m.argsort()
paths_list_m = paths_list_m[ordering_m]
stock_growth_list_m = stock_growth_list_m[ordering_m]

proportion_m = int(proportion*len(paths_list_m))

paths_list_m_lower = paths_list_m[:proportion_m]
paths_list_m_higher = paths_list_m[-proportion_m:]


### keep only the outliers
def keep_outliers(paths):
    new_paths = []
    paths_list = []
    current_path = paths[0].split("/")[6]
    for path in paths:
        path_info = path.split("/")

        if path_info[6] == current_path:
            new_paths.append(path)
        else:
            current_path = path_info[6]
            new_mfccs = []
            paths_to_check = []
            for path_to_look_at in new_paths:
                mfcc_to_look_at = get_mfcc_from_arff(path_to_look_at)
                if mfcc_to_look_at != None:
                    new_mfccs.append(mfcc_to_look_at)
                    paths_to_check.append(path_to_look_at)
            new_paths = paths_to_check

            arr = np.array(new_mfccs)
            print(arr.shape)
            arr = np.reshape(arr, (arr.shape[0], 630))
            df = pd.DataFrame(arr)

            no_outliers = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
            to_remove_from_paths = []
            for index, row in no_outliers.iterrows():
                print(index)
                to_remove_from_paths.append(index)

            counter = 0
            paths_to_add = []
            for new_path in new_paths:
                if counter not in to_remove_from_paths:
                    #print("ADDDE")
                    paths_to_add.append(new_path)
                counter += 1

            paths_list.append(paths_to_add)


            new_paths = []

    return paths_list

if onlyOutliers:
    paths_list_m_lower = keep_outliers(paths_list_m_lower)
    paths_list_m_higher = keep_outliers(paths_list_m_higher)

#take the upper and lower n

path_m = []
emotion_m = []


for paths_per_earning in paths_list_m_lower:
    for path in paths_per_earning:
        path_m.append(path)
        emotion_m.append("NEGATIVE")
for paths_per_earning in paths_list_m_higher:
    for path in paths_per_earning:
        path_m.append(path)
        emotion_m.append("POSITIVE")


stock_arff_df = pd.DataFrame(emotion_m, columns=['labels'])
stock_arff_df = pd.concat([stock_arff_df, pd.DataFrame(path_m, columns=['path'])], axis=1)
stock_arff_df.labels.value_counts()

stock_arff_df.head()
stock_arff_df.to_csv("arff_data_path_enhanced_m.csv", mode='w', index=False, header=False)


'''
stock_arff_df = pd.DataFrame(emotion_w, columns=['labels'])
stock_arff_df = pd.concat([stock_arff_df, pd.DataFrame(path_w, columns=['path'])], axis=1)
stock_arff_df.labels.value_counts()

stock_arff_df.head()
stock_arff_df.to_csv("arff_data_path_enhanced_w.csv",index=False)'''