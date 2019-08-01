import pyautogui
import sys
import time
import csv
from scrape import imagesearch

import os



time_delay = 1

with open('ma_stocks.csv', mode='r') as infile:
    reader = csv.reader(infile)
    with open('ma_stocks_new.csv', mode='w') as outfile:
        writer = csv.writer(outfile)
        abb_to_stock = {rows[0]:rows[1] for rows in reader}

print(abb_to_stock['A'])


counter = 0

#when to start and to stop
start_count = 0
max_count = 30000

time.sleep(3*time_delay)
with open("filename1.txt") as f:
    content = f.readlines()
    for line in content:
        if start_count < counter < max_count:
            earnings_info = line
            earnings_infos = earnings_info[:-5].split("-")
            print(earnings_infos)
            if not os.path.isfile('transcripts/' + earnings_info[:-5] + '.txt'):

                pos = imagesearch.imagesearch("search_box.png")
                if pos[0] != -1:
                    pyautogui.moveTo(pos[0], pos[1])

                pyautogui.click()

                pyautogui.typewrite(abb_to_stock[earnings_infos[0]] + " (")
                #pyautogui.typewrite("(")

                pyautogui.typewrite(earnings_infos[0] + ")")

                pyautogui.typewrite(" ")
                pyautogui.typewrite(earnings_infos[2])
                pyautogui.typewrite(" ")
                pyautogui.typewrite(earnings_infos[1])
                pyautogui.typewrite(" - earnings call transcript")

                time.sleep(0.2 * time_delay)
                pyautogui.press('enter')

                time.sleep(2.8 * time_delay)
                pyautogui.moveTo(709, 369)
                pyautogui.click()

                time.sleep(2)
                pos = imagesearch.imagesearch("search_for_transcript.png")
                if pos[0] != -1:
                    print("position : ", pos[0], pos[1])
                    pyautogui.moveTo(pos[0], pos[1])
                    pyautogui.click()


                time.sleep(2 * time_delay)
                pyautogui.moveTo(600, 246)
                pyautogui.mouseDown()
                time.sleep(0.5 * time_delay)
                pyautogui.moveTo(736, 379)
                pyautogui.scroll(-3000)
                pyautogui.mouseUp()

                #pyautogui.keyDown('ctrl')
                #pyautogui.press('c')
                #pyautogui.keyUp('ctrl')

                transcript_to_save = os.popen('xsel').read()
                print("OKOKOKOK")
                print(transcript_to_save)
                with open('transcripts/' + earnings_info[:-5] + '.txt', 'w') as file_to_write:
                    file_to_write.write(transcript_to_save)
                file_to_write.close()


                counter += 1
                time.sleep(1 * time_delay)

        counter += 1
print(counter)