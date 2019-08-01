from scrape import imagesearch
import pyautogui


pos = imagesearch.imagesearch("search_for_transcript.png")
if pos[0] != -1:
    print("position : ", pos[0], pos[1])
    pyautogui.moveTo(pos[0], pos[1])
else:
    print("image not found")