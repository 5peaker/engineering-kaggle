# 사용할 모듈 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
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