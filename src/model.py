import numpy as np
from pandas import DataFrame
import argparse
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument('--train', action='store_true', help='Train model')
parser.add_argument('--display', action='store_true', help='Run pretrained model for prediction')
parser.add_argument('--all', action='store_true', help='Train model and run it for prediction')

args = parser.parse_args()
if not args.train and not args.display:
    args.all = True

emotions = {'angry': 0, 'disgusted': 1, 'fearful': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprised': 6}
emotions_id = {0: 'angry', 1: 'disgusted', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
size = 48

# <---- MODEL ---->
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(size, size, 1)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),

    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')
])

if args.all or args.train:
    # <---- MAKE DATAFRAME ---->
    data = []
    label = []

    for dirname, _, filenames in os.walk('../data'):
        print(dirname)
        for filename in tqdm(filenames):
            if filename.endswith('.png'):
                image = Image.open(os.path.join(dirname, filename))
                resized_image = image.resize((size, size))
                image_array = np.array(resized_image)

                data.append(image_array)
                label.append(emotions[dirname.split('/')[-1]])

    data = np.array(data)
    label = to_categorical(np.array(label))

    xtrain, xtest, ytrain, ytest = train_test_split(data, label, test_size=0.3)
    xtrain = np.expand_dims(xtrain, -1)
    xtest = np.expand_dims(xtest, -1)

    # <---- MODEL BUILDING ---->
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(xtrain, ytrain, epochs=50, batch_size=64, validation_split=0.1)
    test_loss, test_acc = model.evaluate(xtest, ytest)
    print('Accuracy on test data:', test_acc)

    model.save_weights('model.weights.h5')

if args.all or args.display:
    # <---- MODEL LOADING ---->
    model.load_weights('model.weights.h5')


    # <---- START VIDEO CAPTURE ---->
    cv2.ocl.setUseOpenCL(False)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise Exception('Camera can not be opened')

    while True:
        ret, frame = cap.read()
        if not ret:
            raise Exception('Could not get a frame')

        # <---- FACE DETECTION ---->
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

        # <---- DRAWING FACE RECTANGLE ---->
        for (x, y, w, h) in faces:
            w += 50
            h += 50
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # <---- EMOTION PREDICTION ---->
            gray_image = gray_image[y:y+h, x:x+w]
            if gray_image.size > 0:
                gray_image = cv2.resize(gray_image, (size, size), interpolation=cv2.INTER_CUBIC)
                gray_image = np.expand_dims(np.expand_dims(gray_image, -1), 0)

                prediction = model.predict(gray_image)
                maxindex = int(np.argmax(prediction))
                cv2.putText(frame, emotions_id[maxindex], (x+50, y-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video Stream', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()