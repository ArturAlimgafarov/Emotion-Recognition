import cv2
import os

path = os.path.abspath('') + '\\sad'
for filename in os.listdir(path):
    image = cv2.imread('sad\\' + filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (50, 50))
    cv2.imwrite('sad\\' + filename, gray)