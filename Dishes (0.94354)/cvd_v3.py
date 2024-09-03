# 사용할 모듈 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import os
import zipfile

from pathlib import Path
from tqdm import tqdm
from rembg import remove, new_session

import tensorflow as tf
import keras

with zipfile.ZipFile('/kaggle/input/platesv2/plates.zip', 'r') as zip_obj:
   zip_obj.extractall('/kaggle/working/')
   
data_root = '/kaggle/working/plates/'

session = new_session()
labels = ['cleaned', 'dirty']

def get_image_and_label_batch(dataset, n, labels=None):
    plt.figure(figsize=(3 * min(n, 8), 4 * (n // 8 + 1)))
    if labels is None:
        images, labels = next(dataset)
    else:
        images, _ = next(dataset)
    for i, (img, l) in enumerate(zip(images[:n], labels[:n])):
        ax = plt.subplot(n // 8 + 1, min(n, 8), i + 1)
        plt.imshow(img.astype('uint8'))
        plt.title(l)
        plt.axis("off")

# 파일 별 위치 탐지
for dir_name in ['train', 'val']:
    for l in labels:
        os.makedirs(os.path.join(dir_name, l), exist_ok=True)

# 파일 열어보기 (훈련용 파일)
for l in labels:
    for i, file in enumerate(tqdm(Path(f"/kaggle/working/plates/train/{l}").glob('*.jpg'))):
        input_path = str(file)
        if i % 5 == 0:
            output_path = f"/kaggle/working/val/{l}/{file.stem}.jpg"
        else:
            output_path = f"/kaggle/working/train/{l}/{file.stem}.jpg"
        with open(input_path, 'rb') as i:
            with open(output_path, 'wb') as o:
                input = i.read()
                output = remove(input, session=session)
                o.write(output)
                os.makedirs("test/unknown", exist_ok=True)

session = new_session()

# 파일 열어보기 (테스트용 파일)
for i, file in enumerate(tqdm(Path("/kaggle/working/plates/test").glob('*.jpg'))):
        input_path = str(file)
        output_path = f"/kaggle/working/test/unknown/{file.stem}.jpg"
        with open(input_path, 'rb') as i:
            with open(output_path, 'wb') as o:
                input = i.read()
                output = remove(input, session=session)
                o.write(output)

# keras 사용해 훈련용 이미지 전처리 실행
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip = True
        )

train_ds = train_datagen.flow_from_directory(
        "./train",
        target_size=(224, 224),
        keep_aspect_ratio=True,
        batch_size=32,
        class_mode='binary',
        shuffle=True)

# keras 사용해 검증용 이미지 전처리 실행
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

val_ds = val_datagen.flow_from_directory(
        "./val",
        target_size=(224, 224),
        keep_aspect_ratio=True,
        batch_size=8,
        class_mode='binary',
        shuffle=False)

get_image_and_label_batch(train_ds, 4)

get_image_and_label_batch(val_ds, 4)

base_model = keras.applications.ResNet152(
    weights="imagenet",  # ImageNet에서 미리 훈련된 이미지 기준 
    input_shape=(224, 224, 3),
    include_top=False,  # 맨 위에 ImageNet 분류기를 두지 않는다
)

# 기본 모델을 동결 (추가적 학습 차단)
base_model.trainable = False

# 새로운 모델을 생성, 최상층에 배치
inputs = keras.Input(shape=(224, 224, 3))

# ResNet 전처리 메소드 중 생성한 모델에 맞는 것 사용
x = keras.applications.resnet.preprocess_input(inputs)

# 기반 모델은 배치정규화 레이어를 포함
# 파인 튜닝 시 해당 레이어를 동결하여 영향을 받지 않게끔 한다
x = base_model(x, training=False)

# 모델에서 사용할 레이어의 패러미터를 설정
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(400, activation='relu')(x)
x = keras.layers.Dropout(0.25)(x)  # Regularize with dropout
outputs = keras.layers.Dense(1, activation = 'sigmoid')(x)

# 딥 러닝 위해 사용할 모델 지정
model = keras.Model(inputs, outputs)
model.compile(loss ='binary_crossentropy', 
              optimizer = keras.optimizers.Adam(learning_rate = 0.0003, amsgrad = True), 
              metrics = ['binary_accuracy'])

# 만들어진 모델의 개요 출력
model.summary()

# 학습 완료 시 일찍 종료할 수 있게끔 설정 
cb_early_stopper = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 7)
hist = model.fit(train_ds,
          validation_data = val_ds,
          epochs = 200,
          callbacks = [cb_early_stopper])

history_frame = pd.DataFrame(hist.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()

shutil.copytree('plates/test', 'test/unknown', dirs_exist_ok=True)

# keras 사용해 테스트용 이미지 전처리 실행
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
test_ds = test_datagen.flow_from_directory(
        './test',
        target_size = (224, 224),
        keep_aspect_ratio = True,
        batch_size = 32,
        shuffle = False)
test_ds.reset()

# 만들어진 모델과 주어진 이미지 기반으로 예측 실행
preds = model.predict(test_ds, verbose=True)
preds[:10]

test_ds.reset()
get_image_and_label_batch(test_ds, 4, labels=preds)

labels = ['dirty' if x > 0.5 else 'cleaned' for x in preds]
labels[:8]

# 샘플 출력 데이터를 기반으로 제출용 csv 파일을 출력
submission_df = pd.read_csv('/kaggle/input/platesv2/sample_submission.csv')
submission_df['label'] = labels
submission_df

submission_df.to_csv('submission.csv', index=False)