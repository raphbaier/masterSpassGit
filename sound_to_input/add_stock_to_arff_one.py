#We add the stock difference to the arffs

import yfinance as yf
import os
from datetime import datetime
from datetime import timedelta
import shutil

DIRECTORY_ARFFS = "/media/raphael/MasterSpass/masterSpass/wav/"
DIRECTORY_TRANSCRIPTS = "/media/raphael/MasterSpass/masterSpass/scrape/transcripts/"
DIRECTORY_TO_SAVE = "/media/raphael/MasterSpass/masterSpass/one_arff_with_label/"

#how many days before the earnings talk do we look?
#how many days after the earnings talk do we look?
#difference between both stock courses will make the label
DAYS_BEFORE = 2
DAYS_AFTER = 2



for file in os.listdir(DIRECTORY_ARFFS):
    stock_name = file.split("-")[0]
    #print(stock_name)
    print(file)

    #find out the date of the earnings call
    with open(DIRECTORY_TRANSCRIPTS + file + ".txt") as transcript:
        date_name = transcript.readlines()[2]
    transcript.close()
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
                    stock_growth = (second_course/first_course)-1
                    print(first_course)
                    print(second_course)
                    print(stock_growth)
                    emotion = "lower"
                    if stock_growth > 0:
                        emotion = "higher"
                    new_dir = DIRECTORY_TO_SAVE + file + ".arff"

                    if not os.path.isfile(new_dir):
                        arff_dir = DIRECTORY_ARFFS + file + "/" + "arff/"
                        first_file = True


                        for arff_file in os.listdir(arff_dir):
                            with open(arff_dir + arff_file) as sub_file:
                                if first_file:
                                    sub_file_with_stock = open(new_dir, "w+")
                                    for line in sub_file.readlines():
                                        if line == "@attribute class numeric\n":
                                            sub_file_with_stock.write("@attribute emotion {lower, higher}\n")
                                        elif line[0:7] == "\'noname":
                                            sub_file_with_stock.write(line[:-4] + emotion + "\n")
                                        else:
                                            sub_file_with_stock.write(line)
                                    sub_file_with_stock.close()
                                else:
                                    sub_file_with_stock = open(new_dir, "a")
                                    for line in sub_file.readlines():
                                        if line[0:7] == "\'noname":
                                            sub_file_with_stock.write(line[:-4] + emotion + "\n")
                            sub_file.close()
                            first_file = False

            except KeyError:
                print("Couldn't find symbol.")