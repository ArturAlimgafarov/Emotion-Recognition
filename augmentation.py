import cv2
import random

def rotate(image, angle):
    h, w, _ = image.shape
    center = (w // 2, h // 2)
    mtx = cv2.getRotationMatrix2D(center, angle, 1)
    return cv2.warpAffine(image, mtx, (w, h))

face_cscd = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

image = cv2.imread('dataset\\calm_39.jpg')
image1 = rotate(image, random.randint(5, 10))
image2 = rotate(image, -random.randint(5, 10))

cv2.imshow('Face', image)
cv2.waitKey(0)

cv2.imshow('Face', image1)
cv2.waitKey(0)

cv2.imshow('Face', image2)
cv2.waitKey(0)

cv2.destroyAllWindows()