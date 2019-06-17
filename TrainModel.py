import numpy as np
from skimage import color, exposure, transform,io
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import backend as K
K.set_image_data_format('channels_first')
import os
import glob

NUM_CLASSES = 43
IMG_SIZE = 48

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

def get_class(img_path):
    return int(img_path.split('\\')[-2])

root_dir = 'GTSRB/Training/images/'
imgs = []
labels = []

all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
np.random.shuffle(all_img_paths)
for img_path in all_img_paths:
    img = preprocess_img(io.imread(img_path))
    label = get_class(img_path)
    imgs.append(img)
    labels.append(label)

X = np.array(imgs, dtype='float32')
# Make one hot targets
Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]

def cnn_model():
    model = Sequential([Conv2D(16, (3, 3), padding='same',
                        input_shape=(3, IMG_SIZE, IMG_SIZE),
                               activation='relu'),
                        Conv2D(16, (3, 3), activation='relu'),
                        MaxPooling2D(pool_size=(2, 2)),
                        Dropout(0.2),
                        Conv2D(32, (3, 3), padding='same',
                               activation='relu'),
                        Conv2D(32, (3, 3), activation='relu'),
                        MaxPooling2D(pool_size=(2, 2)),
                        Dropout(0.2),
                        Conv2D(64, (3, 3), padding='same',
                               activation='relu'),
                        Conv2D(64, (3, 3), activation='relu'),
                        MaxPooling2D(pool_size=(2, 2)),
                        Dropout(0.2),
                        Flatten(),
                        Dense(512, activation='relu'),
                        Dropout(0.5),
                        Dense(NUM_CLASSES, activation='softmax')])
    return model

model = cnn_model()
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

def lr_schedule(epoch):
    return 0.01 * (0.01 ** int(epoch / 10))

batch_size = 50
epochs = 30

model.fit(X, Y,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          callbacks=[LearningRateScheduler(lr_schedule),
                     ModelCheckpoint('model.h5', save_best_only=True)]
          )