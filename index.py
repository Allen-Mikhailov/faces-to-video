import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageOps
import numpy as np
import moviepy.editor as editor
from pillow_heif import register_heif_opener

register_heif_opener()

image_folder = "./images/"

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    # return np.asarray(img)
    # first = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

k = 0
def find_face(img):
    global k
    # gray sca,e processing
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=6, minSize=(300, 300)
    )

    x, y, w, h = -1, -1, -1, -1
    largestArea = 0

    for (rx, ry, rw, rh) in face:
        area = rw*rh
        if (largestArea < area):
            x, y, w, h = rx, ry, rw, rh

    if (w != -1):
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 4)

    name = image_paths[k].split("/")[-1].split(".")[0]
    cv2.imwrite("./test_images/"+name+".jpeg", img)
    k += 1

    return (x, y, w, h)

image_paths = [image_folder+img for img in os.listdir("./images")]
loaded_images = [ImageOps.exif_transpose(Image.open(img)) for img in image_paths]
loaded_image_arrays = [convert_from_image_to_cv2(img) for img in loaded_images]
face_rects = [find_face(img) for img in loaded_image_arrays]
face_pos = []
max_sizes = []
valid_images = [rect[0] != -1 for rect in face_rects]

# Large number that is surely never true
lw, lh = 0, 0

scale_const = 1.5 

for i in range(len(loaded_images)):
    if not valid_images[i]:
        print(image_paths[i]+" does not have a valid face :(")
        face_pos.append((0, 0))
        max_sizes.append(0)
        continue

    size = (loaded_image_arrays[i].shape)
    size = (size[1], size[0])
    frect = face_rects[i]

    w = frect[2] * scale_const
    h = frect[3] * scale_const
    x = frect[0] + frect[2]/2
    y = frect[1] + frect[3]/2

    face_pos.append((x, y))

    max_dim  = 10000

    left = max(x-w/2, 0)
    right = min(x+w/2, size[0])

    top = max(y-h/2, 0)
    bottom = min(y+h/2, size[1])

    max_dim = min(x - left, max_dim)
    max_dim = min(right - x, max_dim)
    max_dim = min(y - top, max_dim)
    max_dim = min(bottom - y, max_dim)

    max_sizes.append(max_dim)

    lw = max(lw, right-left)
    lh = max(lh, bottom-top)

resize = (int(lw), int(lh))

cropped_images = []
for i in range(len(loaded_images)):
    if not valid_images[i]:
        continue
    img = loaded_images[i]
    rect = face_rects[i]
    pos = face_pos[i]

    max_size = max_sizes[i]


    crop = (
        int(pos[0]-max_size), 
        int(pos[1]-max_size), 
        int(pos[0]+max_size), 
        int(pos[1]+max_size)
        )

    cropped_images.append(np.array(img.crop(crop).resize(resize)))
  
loops = 4
clip_images = []
for i in range(loops):
    clip_images += cropped_images

clip = editor.ImageSequenceClip(clip_images, fps = 10) 
  
# showing  clip  
clip.write_videofile("./test.mp4")

# cropped_images[0].save("./test3.jpeg")