
# 모듈 불러오기
import numpy as np 
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import cv2
import random
import multiprocessing

import shutil
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Add, Dense, Activation, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout 
from keras import Model
from keras.applications import ResNet50, MobileNetV2, ResNet101
from keras.optimizers import Adam, SGD 
from keras.preprocessing.image import ImageDataGenerator, save_img

import os
for dirname, _, filenames in os.walk('../plates'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# 청소
if (os.path.exists('data_augment')):
    shutil.rmtree('data_augment')
if (os.path.exists('data_test')):
    shutil.rmtree('data_test')

# 데이터 분류용 폴더 생성
os.makedirs('data_augment/plates/train/cleaned')
os.makedirs('data_augment/plates/train/dirty')
os.makedirs('data_augment/valid/plates/train/cleaned')
os.makedirs('data_augment/valid/plates/train/dirty')

# 테스트 파일용 폴더 생성
os.makedirs('data_test/plates/test')

# 데이터 가공

## 화상 표준화
def image_standardization(img):
    return tf.image.per_image_standardization(
        img
    )
    
## 그레이스케일로 변환
def image_grayscale(img):
    
    # PIL형을 openCV형으로 변환하는 작업 진행
    img = np.array(img, dtype=np.uint8)
    img = img[:, :, ::-1]
    
    # 그레이스케일 파일 반환
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # PIL형 반환
    return cv2.cvtColor(img_gray, cv2.COLOR_BGR2RGB)

## 배경 제거
def grabCutFirst(img):

    # PIL형을 openCV형으로 변환하는 작업 진행
    img = np.array(img, dtype = np.uint8)
    img = img[:, :, ::-1]
    
    # 이미지 크기 정의
    height, width = img.shape[:2]
    rect = (15, 15, width - 30, height - 30)
    
    # 배경 제거용 마스크 졍의
    mask = np.zeros(img.shape[:2], np.uint8)
    
    # 배경, 전경 추출 (grabcut)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    
    mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')
    output_img = img * mask2[:, :, np.newaxis]  

    # 이미지에서 배경 제거 실행 
    background = img - output_img

    # 흑백 반전
    background[np.where((background > [0, 0, 0]).all(axis = 2))] = [255, 255, 255]

    # PIL형 반환 
    return cv2.cvtColor(background + output_img, cv2.COLOR_BGR2RGB)

## 이미지 크롭
def crop(img, l):

    img = Image.fromarray(img.astype(np.uint8))
    
    # 크롭 실행
    l2 = l // 2     # 기존 이미지의 절반으로 나눈다
    w, h = img.size # 이미지의 너비와 높이 지정 
    w2 = w // 2     # 기존 너비의 절반 
    h2 = h // 2     # 기존 높이의 절반
    img = img.crop((w2 - l2, h2 - l2, w2 + l2, h2 + l2))

    # 이미지 리사이즈하여 반환
    img = img.resize((w, h))
    return img

## 학습을 위한 이미지 가공 메소드
def image_transform_for_training(org_image_dir_path, crop_size_list, rotation_range, sum_data_num, valid_data_num):
    
    print('이미지 학습을 위한 가공: ' + org_image_dir_path + ' --> ' + str(crop_size_list[0]))
    
    # "ImageDataGenerator" 인스턴스 생성 
    datagen = ImageDataGenerator(
           rotation_range = rotation_range,
           width_shift_range = 0,
           height_shift_range = 0,
           shear_range = 0,
           zoom_range = 0,
           horizontal_flip = False,
           vertical_flip = False,
           preprocessing_function = image_standardization)
    
    i = 0
    valid_iter = random.sample(range(sum_data_num), int(valid_data_num))
    for org_image_file_name in os.listdir(org_image_dir_path):
        
        root, ext = os.path.splitext(org_image_file_name)
        if (ext != '.jpg'):
            continue

        # 학습 및 검증용 이미지 디렉토리와 주소 결합
        image_dir_path = org_image_dir_path
        if (i in valid_iter):
            image_dir_path = 'valid/' + org_image_dir_path

        print('이미지 학습을 위해 가공하는 파일: ' + org_image_file_name)
            
        # 이미지 파일을 PIL형식으로 열기
        img = Image.load_img(org_image_dir_path + '/' + org_image_file_name)
        
        # PIl형식은 numpy의 ndarray형식으로 교환
        img = Image.img_to_array(img)
        
        # 배경소거
        x = grabCutFirst(img)
        
        # (height, width, 3) -> (1, height, width, 3)
        x = x.reshape((1,) + x.shape)
        
        # 학습용 이미지 파일 생성 작업 진행
        j = 0
        for d in datagen.flow(x, batch_size = 1):
            grab_cut_img = grabCutFirst(d[0])
            
            for l in crop_size_list:
                crop_img = Image.img_to_array(crop(grab_cut_img, l))
                
                std_img = np.array(image_standardization(crop_img))
                std_img = crop_img
                
                gray_img = image_grayscale(std_img)
                gray_img = std_img
                
                # 학습용 데이터를 지정된 주소에 맞추어 저장
                save_img('data_augment/' + image_dir_path + '/' + root + '_' + str(l) + '_' + str(j * rotation_range) + ext, Image.fromarray(gray_img.astype(np.uint8)))
            j += 1
            
            if ((360/rotation_range) <= j):
                break
        i += 1

    print('학습을 위해 이미지 변환 완료: ' + org_image_dir_path + ' - ' + str(crop_size_list[0]))
    
## 테스트를 위한 이미지 가공 메소드
def image_transfrom_for_test(org_image_dir_path, crop_size_list):
    
    print('테스트를 위한 이미지 가공 중: ' + org_image_dir_path + ' - ' + str(crop_size_list[0]))
    
    for org_image_file_name in os.listdir(org_image_dir_path):

        root, ext = os.path.splitext(org_image_file_name)
        if (ext != '.jpg'):
            continue
    
        print('테스트를 위해 현재 가공 중인 파일: ' + org_image_file_name)
    
        img = Image.load_img(org_image_dir_path + '/' + org_image_file_name)
        img = Image.img_to_array(img)
        
        # 배경 소거
        img = grabCutFirst(img) 
        
        for l in crop_size_list:

            if (os.path.exists('data_test/' + org_image_dir_path + '/' + str(l)) == False):
                os.makedirs('data_test/' + org_image_dir_path + '/' + str(l))

            crop_img =  Image.img_to_array(crop(img, l))
            
            std_img = np.array(image_standardization(crop_img))
            std_img = crop_img
            
            gray_img = image_grayscale(std_img)
            gray_img = std_img
            
            # 테스트용 데이터를 지정된 주소에 맞추어 저장
            save_img('data_test/' + org_image_dir_path + '/' + str(l) + '/' + org_image_file_name,Image.fromarray(gray_img.astype(np.uint8)))
    
    print('테스트 위해 이미지 변환 완료: ' + org_image_dir_path + ' - ' + str(crop_size_list[0]))

# 딥 러닝 모델 정의 
def get_model():
    ## https://keras.io/ja/applications/#resnet50
    input_shape = image_size + (3,)
    model_res = ResNet101(include_top=False, input_shape=input_shape, weights='imagenet')
    
    # 추가적 레이어 정의 
    x = model_res.output
    
    x = Flatten()(x)

    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = Dropout(.5)(x)
    
    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = Dropout(.5)(x)
    
    x = Dense(128)(x)
    x = Activation('relu')(x)
    x = Dropout(.5)(x)

    x = Dense(1)(x)
    
    outputs = Activation('sigmoid')(x)

    # 전송 모델은 학습시키지 않는다
    # 추가 레이어 이외의 레이어를 고정 (매개 변수 프리즈)
    for l in model_res.layers[1:]:
        l.trainable = False
    
    # 회전 모델과 추가 레이어를 합성
    model = Model(model_res.input, outputs)
    
    return model

# 학습 진행 
def learning(key, model):

    ## 정의된 모델 컴파일 진행 (필요한 옵션 선택해도 무방할 것)
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=Adam(decay=0.1), loss='binary_crossentropy', metrics=['binary_accuracy'])

    ## 학습 절차의 이행
    ### 조기에 중지할 경우를 정의
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=30
    )
    ### 모델 보존 설정
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        '/tmp/checkpoint_' + str(key), 
        monitor='val_binary_accuracy', 
        save_best_only=True
        # 최상의 val_binary_accuarcy 결과 저장
    )
    ### 피팅 절차의 이행
    return model.fit(
        train_ds,
        validation_data=val_ds, 
        epochs=epochs, 
        callbacks=[callback, checkpoint]
    )
    
# 테스트 결과를 생성
def create_test_generator(l):
    test_datagen = ImageDataGenerator()
    return test_datagen.flow_from_directory(  
        'data_test/plates/test',
        classes=[str(l)],
        target_size = image_size,
        batch_size = 100,
        shuffle = False,        
        class_mode = None)  
    
# 가공용 크롭 사이즈 종류
crop_size_training_list = [91, 171, 251]
crop_size_test_list = [91, 171, 251]

# 생성 이미지의 회전 각도
rotation_range = 90

# 가공 및 생성 작업 처리할 리스트 생성
processes = []

## 학습할 이미지 리스트에 저장
for l in crop_size_training_list:  
    processes.append(multiprocessing.Process(target = image_transform_for_training, args = ('plates/train/cleaned', [l], rotation_range, 20, 20 * 0.3,)))
    processes.append(multiprocessing.Process(target = image_transform_for_training, args = ('plates/train/dirty', [l], rotation_range, 20, 20 * 0.3,)))
    
## 테스트용 이미지 맞는 리스트에 저장
for l in crop_size_test_list:
    processes.append(multiprocessing.Process(target = image_transfrom_for_test, args = ('plates/test', [l],)))

## 전체 프로세스 개시
for p in processes:
    p.start()

## 프로세스 완료될 때까지 대기
for p in processes:
    p.join()  
    
# 이미지 크기 및 학습 횟수 등 뱐수 정의 과정
image_size = (224, 224)
batch_size = 4
epochs = 30

# 학습옹 데이터
train_ds = tf.keras.preprocessing.image_dataset_from_directory (
    "data_augment/plates/train",
    seed=1307,
    image_size=image_size,
    batch_size=batch_size,
)

# 검증용 데이터
val_ds = tf.keras.preprocessing.image_dataset_from_directory (
    "data_augment/valid/plates/train",
    seed=1307,
    image_size=image_size,
    batch_size=batch_size,
)

# 학습 데이터 | 시각적 확인
plt.figure(figsize=(20, 20))
for images, labels in train_ds.take(1):
    for i in range(batch_size):
        ax = plt.subplot(7, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
        
# 검증용 데이터 | 시각적 확인
plt.figure(figsize=(20, 20))
for images, labels in val_ds.take(1):
    for i in range(batch_size):
        ax = plt.subplot(7, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
        
# 모델용 인스턴스 (필요시 주석 수정해 늘일 것)
models = {}
models[0] = get_model()
models[1] = get_model()
models[2] = get_model()
models[0].summary()

# 학습결과 보존 
results = {}
for key, model in models.items():
    print('=== model-' + str(key) + ' fiting ===')
    results[key] = learning(key, model)
    
# 학습 결과의 개요를 시각화
result = results[0]
his_range = len(result.history['loss'])

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, his_range+1), result.history['binary_accuracy'], label="training")
plt.plot(range(1, his_range+1), result.history['val_binary_accuracy'], label="validation")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, his_range+1), result.history['loss'], label="training")
plt.plot(range(1, his_range+1), result.history['val_loss'], label="validation")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
    
# 테스트 결과를 저장 
test_generators = {}
for l in crop_size_test_list:
    test_generators[str(l)] = create_test_generator(l)
    
model = models[0]

## 학습용 세트 생성
test_generator = test_generators[str(crop_size_test_list[0])]
test_generator.reset()

## 정확도 체크
for d in test_generator:
    for i in range(30):
        print(model.predict([d[i][None,...]]))
        plt.imshow(d[i].astype(np.uint8))
        plt.show()
    break

# 예상
predicts = {}

for key, model in models.items():
    for key_gen, test_generator in test_generators.items():
        # 생성된 테스트 결과 리셋 
        test_generator.reset()
        # 예측 
        predicts['model:' + str(key) + ' - inputsize:' + str(key_gen)] = pd.Series(
            np.ravel( # 일차원화
                model.predict_generator(
                    test_generator, 
                    steps = len(test_generator.filenames)
                )
            )
        )
        
# 예상 결과 저장용 데이터프레임 생성        
predicts_df = pd.DataFrame(predicts)
predicts_df.head(30)

# 제출 결과 샘플 불러오기 
sub_df = pd.read_csv('../input/platesv2/sample_submission.csv')

# 샘플에 맞추어 제출용 파일 만들기
f = lambda x: 'dirty' if x > 0.5 else 'cleaned'
sub_df['label'] = pd.DataFrame(
    np.mean(
        predicts_df, 
        axis=1
    )
)
sub_df['label'] = sub_df['label'].apply(f)
sub_df.head(30)

sub_df['label'].value_counts()
sub_df.to_csv('submission.csv', index=False)