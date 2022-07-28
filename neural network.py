import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
import cv2

model = load_model('emotion_nn_2.h5')
class_names = ['angry', 'happy', 'neutral', 'sad']

capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
face_cscd = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
while True:
    _, img = capture.read()
    faces = face_cscd.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face_image = img[y:y+h, x:x+w]
        cv2.imwrite('test\\images\\image.jpg', face_image)

        new_image = image_dataset_from_directory('test', batch_size=256, image_size=(48, 48))
        prediction = model.predict(new_image)

        max_width = 50
        y1 = 20
        x1 = 5
        for i in range(len(prediction[0])):
            class_name = class_names[i]
            value = prediction[0][i]
            width = value * max_width
            cv2.putText(img, class_name, (x1, y1 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.rectangle(img, (x1 + 60, y1 + 5), (x1 + 60 + int(width), y1 + 5 + 10), (255, 255, 255), -1)
            y1 += 20

        emotion_name = class_names[np.argmax(prediction)]

        cv2.putText(img, emotion_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.imshow('Camera', img)

    key = cv2.waitKey(2) & 0xFF
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()