import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageOps
import numpy as np

image_folder = "./images/"

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    # return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    # return np.asarray(img)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def find_face(img):
    # gray sca,e processing
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=6, minSize=(100, 100)
    )

    x, y, w, h = -1, -1, -1, -1
    largestArea = 0

    for (rx, ry, rw, rh) in face:
        area = rw*rh
        if (largestArea < area):
            x, y, w, h = rx, ry, rw, rh

    return (x, y, w, h)

image_paths = [image_folder+img for img in os.listdir("./images")]
loaded_images = [ImageOps.exif_transpose(Image.open(img)) for img in image_paths]
loaded_image_arrays = [convert_from_image_to_cv2(img) for img in loaded_images]
face_rects = [find_face(img) for img in loaded_image_arrays]
max_sizes = []
valid_images = [rect[0] != -1 for rect in face_rects]

# Large number that is surely never true
sw, sh = 10000, 10000

scale_const = 1.5 / 2

for i in range(len(loaded_images)):
    if not valid_images[i]:
        print(image_paths[i]+" does not have a valid face :(")
        continue

    size = loaded_image_arrays[i].shape
    frect = face_rects[i]

    max_dim  = 10000

    left = max(frect[0]-frect[2]*scale_const, 0)
    right = min(frect[0]+frect[2]*scale_const, size[0])

    top = max(frect[1]-frect[3]*scale_const, 0)
    bottom = min(frect[1]+frect[3]*scale_const, size[1])

    max_dim = min(frect[0] - left, max_dim)
    max_dim = min(right - frect[0], max_dim)
    max_dim = min(frect[1] - top, max_dim)
    max_dim = min(bottom - frect[1], max_dim)

    nh = bottom-top

    max_sizes.append(max_dim)

    sw = min(sw, right-left)
    sh = min(sh, bottom-top)

print(sw, sh)

# for (x, y, w, h) in face:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

loaded_images[0].save("test2.jpeg")

open_cv_image = np.array(loaded_images[0])
open_cv_image = open_cv_image[:, :, ::-1].copy()
cv2.imwrite("./test.jpeg", open_cv_image)

cropped_images = []
for i in range(len(loaded_images)):
    img = loaded_images[i]
    rect = face_rects[i]

    crop = (rect[0]-sw/2, rect[1]-sh/2, rect[0]-sw/2, rect[0]-sw/2)
    cropped_images.append(img.crop(crop))

cropped_images[0].save("./test3.jpeg")