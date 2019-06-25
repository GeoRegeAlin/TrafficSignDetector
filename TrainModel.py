import numpy as np
from keras.utils import plot_model
from skimage import color, exposure, transform,io
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import backend as K
K.set_image_data_format('channels_first')
import os
import glob


def preprocess_img(img):
    img = transform.resize(img, (48, 48))
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
Y = np.eye(43, dtype='uint8')[labels]

def cnn_model():
    model = Sequential([Conv2D(16, (3, 3), padding='same',
                        input_shape=(3, 48, 48),
                               activation='relu'),
                        Conv2D(16, (3, 3), activation='relu'),
                        MaxPooling2D(pool_size=(2, 2)),
                        Dropout(0.5),
                        Conv2D(32, (3, 3), padding='same',
                               activation='relu'),
                        Conv2D(32, (3, 3), activation='relu'),
                        MaxPooling2D(pool_size=(2, 2)),
                        Dropout(0.5),
                        Conv2D(64, (3, 3), padding='same',
                               activation='relu'),
                        Conv2D(64, (3, 3), activation='relu'),
                        MaxPooling2D(pool_size=(2, 2)),
                        Dropout(0.5),
                        Flatten(),
                        Dense(512, activation='relu'),
                        Dropout(0.5),
                        Dense(43, activation='softmax')])
    return model


model = cnn_model()
plot_model(model, to_file='convolutional_neural_network.png')
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

def lr_schedule(epoch):
    return 0.01 * (0.01 ** int(epoch / 10))

batch_size = 50
epochs = 10

model.fit(X, Y,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          callbacks=[LearningRateScheduler(lr_schedule),
                     ModelCheckpoint('model.h5', save_best_only=True)]
          )