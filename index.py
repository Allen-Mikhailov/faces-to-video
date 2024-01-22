import cv2
import matplotlib.pyplot as plt
import os

image_folder = "./images/"

def find_face(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    face = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=6, minSize=(100, 100)
    )

    x, y, w, h = 0, 0, 0, 0
    largestArea = 0

    for (rx, ry, rw, rh) in face:
        area = rw*rh
        if (largestArea < area):
            x, y, w, h = rx, ry, rw, rh

    return (x, y, w, h)

loaded_images = [cv2.imread(image_folder+img) for img in os.listdir("./images")]
face_rects = [find_face(img) for img in loaded_images]

# for (x, y, w, h) in face:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

cv2.imwrite("./test.jpeg", loaded_images[0])