from keras.models import load_model
from keras.preprocessing import image
from skimage import color, exposure, transform,io
import numpy as np
IMG_SIZE = 48
import os
import cv2

def preprocess_img(img):
    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)

    # central square crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0] // 2, img.shape[1] // 2
    img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
              centre[1] - min_side // 2:centre[1] + min_side // 2,
              :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # roll color axis to axis 0
    img = np.rollaxis(img, -1)

    return img


model = load_model("model.h5")
new_image = image.load_img("7.png")
new_image=preprocess_img(new_image)
image = np.expand_dims(new_image, axis=0)
pred = model.predict(image)
print(np.argmax(pred))