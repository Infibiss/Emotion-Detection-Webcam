import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os

parser = argparse.ArgumentParser()

parser.add_argument('-train', action='store_true', help='Train model')
parser.add_argument('-display', action='store_true', help='Run pretrained model for prediction')
parser.add_argument('-all', action='store_true', help='Train model and run it for prediction')

args = parser.parse_args()
if not args.train and not args.display:
    args.all = True

# <---- PARAMS ---->
emotions_id = {0: 'angry', 1: 'disgusted', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
img_size = (224, 224)
channels = 3
class_count = len(emotions_id)
img_shape = (img_size[0], img_size[1], channels)
batch_size = 25
epochs = 10

if args.all or args.train:
    # <---- MODEL STRUCTURE ---->
    model = tf.keras.applications.efficientnet.EfficientNetB0(include_top=True, weights=None, classes=class_count, input_shape=img_shape)

    # <---- MAKE DATAFRAME ---->
    filepaths = []
    labels = []

    res = set()
    for dirname, _, filenames in os.walk('../data'):
        print(dirname)
        for filename in tqdm(filenames):
            if filename.endswith('.png'):
                filepaths.append(os.path.join(dirname, filename))
                labels.append(dirname.split('/')[-1])

    fSer = pd.Series(filepaths, name='filepaths')
    lSer = pd.Series(labels, name='labels')
    df = pd.concat([fSer, lSer], axis=1)

    train_df, test_df = train_test_split(df, train_size=0.8, shuffle=True)
    valid_df, test_df = train_test_split(test_df, train_size=0.6, shuffle=True)

    # Augmentation?
    tr_gen = ImageDataGenerator()
    ts_gen = ImageDataGenerator()
    train_gen = tr_gen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                           class_mode='categorical',
                                           color_mode='rgb', shuffle=True, batch_size=batch_size)
    valid_gen = ts_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                           class_mode='categorical',
                                           color_mode='rgb', shuffle=True, batch_size=batch_size)
    test_gen = ts_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                          class_mode='categorical',
                                          color_mode='rgb', shuffle=False, batch_size=batch_size)

    # <---- CLASS WEIGHTS ---->
    classes = np.array(train_gen.classes)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(classes),
        y=classes
    )
    class_weights_dict = dict(enumerate(class_weights))

    # <---- MODEL COMPILING & TRAINING ---->
    model.compile(loss='categorical_crossentropy', optimizer=Adamax(learning_rate=0.001), metrics=['accuracy'])
    # model.summary()
    # exit(0)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model_accuracy.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    history = model.fit(x=train_gen, epochs=epochs, class_weight=class_weights_dict, callbacks=[checkpoint], verbose=1, validation_data=valid_gen, validation_steps=None, shuffle=False)

    # <---- MODEL TRAINING RESULTS ---->
    epochs_x = [i + 1 for i in range(epochs)]

    plt.figure(figsize=(20, 8))
    plt.style.use('fivethirtyeight')

    tr_loss, val_loss = history.history['loss'], history.history['val_loss']
    index_loss = np.argmin(val_loss)
    loss_lowest, loss_label = val_loss[index_loss], f'Best epoch= {str(index_loss + 1)}'
    plt.subplot(1, 2, 1)
    plt.plot(epochs_x, tr_loss, 'r', label='Training loss')
    plt.plot(epochs_x, val_loss, 'g', label='Validation loss')
    plt.scatter(index_loss + 1, loss_lowest, s=120, c='blue', label=loss_label)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    tr_acc, val_acc = history.history['accuracy'], history.history['val_accuracy']
    index_acc = np.argmax(val_acc)
    acc_highest, acc_label = val_acc[index_acc], f'Best epoch= {str(index_acc + 1)}'
    plt.subplot(1, 2, 2)
    plt.plot(epochs_x, tr_acc, 'r', label='Training Accuracy')
    plt.plot(epochs_x, val_acc, 'g', label='Validation Accuracy')
    plt.scatter(index_acc + 1, acc_highest, s=120, c='blue', label=acc_label)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # <---- MODEL TESTING ---->
    test_loss, test_acc = model.evaluate(test_gen)
    print('Accuracy on test data:', test_acc)

    # <---- MODEL SAVING ---->
    model.save('model.keras')

if args.all or args.display:
    # <---- MODEL LOADING ---->
    model = models.load_model('model.keras')

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
                gray_image = cv2.resize(gray_image, img_size, interpolation=cv2.INTER_CUBIC)
                gray_image = np.expand_dims(np.expand_dims(gray_image, -1), 0)

                prediction = model.predict(gray_image)
                max_index = int(np.argmax(prediction))
                cv2.putText(frame, emotions_id[max_index], (x+50, y-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video Stream', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()