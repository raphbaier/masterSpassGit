import os
import pandas as pd
import yfinance as yf
import os
from datetime import datetime
from datetime import timedelta

arff_files = "/media/raphael/masterspass2/masterSpass/wav_sentences"

DIRECTORY_TRANSCRIPTS = "/media/raphael/masterspass2/masterSpass/scrape/transcripts/"


DAYS_BEFORE = 2
DAYS_AFTER = 2

emotion_m = []
path_m = []

emotion_w = []
path_w = []

for file in os.listdir(arff_files):

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


            except KeyError:
                print("Couldn't find symbol.")


    for arff_file in os.listdir(arff_files + "/" + file + "/arff"):
        infos = arff_file.split("_")
        if infos[1] == "m":
            path_m.append(arff_files + "/" + file + "/arff/" + arff_file)
            if stock_growth > 0:
                emotion_m.append("POSITIVE")
            else:
                emotion_m.append("NEGATIVE")
        else:
            path_w.append(arff_files + "/" + file + "/arff/" + arff_file)
            if stock_growth > 0:
                emotion_w.append("POSITIVE")
            else:
                emotion_w.append("NEGATIVE")
        #emotion.append(stock_growth)

stock_arff_df = pd.DataFrame(emotion_m, columns=['labels'])
stock_arff_df = pd.concat([stock_arff_df, pd.DataFrame(path_m, columns=['path'])], axis=1)
stock_arff_df.labels.value_counts()

stock_arff_df.head()
stock_arff_df.to_csv("arff_data_path_m.csv",index=False)



stock_arff_df = pd.DataFrame(emotion_w, columns=['labels'])
stock_arff_df = pd.concat([stock_arff_df, pd.DataFrame(path_w, columns=['path'])], axis=1)
stock_arff_df.labels.value_counts()

stock_arff_df.head()
stock_arff_df.to_csv("arff_data_path_w.csv",index=False)