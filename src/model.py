import numpy as np
import cv2
from tensorflow.keras import models

import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical

# Список для хранения данных
data = []

for dirname, _, filenames in os.walk('../data/train'):
    for filename in tqdm(filenames):
        if filename.endswith('.png'):
            img_path = os.path.join(dirname, filename)
            # Открытие и изменение размера изображения
            img = Image.open(img_path)
            img = img.resize((224, 224))  # Изменение размера
            img_array = np.array(img).flatten()  # Преобразование в 1D массив
            img_array = img_array / 255.0  # Нормализация значений пикселей (0-1)
            # Добавление целевой переменной (эмоции)
            data.append(np.append(img_array, dirname))  # Добавление эмоции в конец массива

# Создание DataFrame и сохранение в CSV
df = pd.DataFrame(data)
print(df)
exit(0)
df.to_csv('emotion_dataset.csv', index=False, header=False)  # Сохранение без индекса и заголовка

# <---- PARAMS ---->
emotions_id = {0: 'angry', 1: 'disgusted', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
img_size = (224, 224)

# <---- LOAD MODEL ---->
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
