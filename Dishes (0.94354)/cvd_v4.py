import numpy as np 
import pandas as pd

import os
import zipfile
import cv2
import shutil 
import torch
import torchvision

from torchvision import transforms, models
from matplotlib import pyplot as plt
from tqdm import tqdm

# 예상 파일 경로 지정
print(os.listdir('../input'),'\n') 
with zipfile.ZipFile('../input/platesv2/plates.zip', 'r') as zip_obj:
        zip_obj.extractall('/kaggle/working/') 
        
print(os.listdir('/kaggle/working/'))

data_root = '/kaggle/working/plates/' 
print(data_root)
print(os.listdir(data_root))

train_dir = 'train' 
val_dir = 'val' 
class_names = ['cleaned', 'dirty']

# 파일 별 위치 탐지
for dir_name in [train_dir, val_dir]:
    for class_name in class_names:
        os.makedirs(os.path.join(dir_name, class_name), exist_ok=True)

# 파일 열어보기 (훈련용 파일)
for class_name in class_names:
    source_dir = os.path.join(data_root, 'train', class_name)
    for i, file_name in enumerate(tqdm(os.listdir(source_dir))):
        if i % 6 != 0:
            dest_dir = os.path.join(train_dir, class_name) 
        else:
            dest_dir = os.path.join(val_dir, class_name)
        shutil.copy(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))

## 배경 제거 및 크롭 기능 함수 모은 클래스 정의
class Remove_background_and_crop:
    
    # 탐지할 범위를 초기화하는 함수 
    def __init__(self, img):
        self.x00 = 0
        self.x00 = 0
        self.r00 = 0
        self.img = img
        self.mask = img
    
    # 이미지 크롭
    def crop(self): 
            c_r_crop = (1.42 * self.r00 / 2)
            self.img = self.img[int(self.y00) - int(c_r_crop) : int(self.y00) + int(c_r_crop), int(self.x00) - int(c_r_crop) : int(self.x00) + int(c_r_crop)]
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            
            crop = self.img
            cv2.imwrite(image_folder,crop)
            h,w = self.img.shape[:2] 
            c = min(h, w)   
              
            for i in range (5, int(c/3), 5): 
                    crop_img = self.img[i : h-i, i : w-i]    
                    cv2.imwrite(image_folder[:-4] + '_Crop_' + str(i) + '.jpg', crop_img)
                    
            image1 = self.img[0 : int(h//2), 0 : int(w//2)]
            cv2.imwrite(image_folder[:-4] + 'image1' + '.jpg', image1)
            
            image2 = self.img[0 : int(h//2) : h, int(w//2) : w]
            cv2.imwrite(image_folder[:-4] + 'image2' + '.jpg', image2)
            
            image3 = self.img[int(h//2) : h, 0 : int(w//2)]
            cv2.imwrite(image_folder[:-4] + 'image3' + '.jpg', image3)
            
            image4 = self.img[0 : int(h//2), int(w//2) : w]
            cv2.imwrite(image_folder[:-4] + 'image4' + '.jpg', image4)
    
    # Crop() 함수 정상동작 확인
    def crop_test(self):
        c_r_crop = (1.42 * self.r00 / 2)
        self.img = self.img[int(self.y00) - int(c_r_crop) : int(self.y00) + int(c_r_crop), int(self.x00) - int(c_r_crop) : int(self.x00) + int(c_r_crop)]
        self.img = cv2.cvtColor(self.img, cv2. COLOR_BGR2RGB)
        cv2.imwrite(image_folder, self.img)       
    
    # 이미지 내에서 원의 여부를 체크하고, 있을 경우 원의 위치를 특정
    def find_circle(self):
        output = self.img.copy()    
        img = cv2.convertScaleAbs(self.img, alpha = 1.2, beta = 0.0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10, param1=10, param2=5, minRadius=40, maxRadius=250)
        
        if circles is not None: 
            print('중심점 좌표 출력:',self.x00,self.y00)
            circles = np.round(circles[0, :]).astype("int")
            print("중심점 좌표 및 지름 기반으로 작도 가능한 원: ")
            print(circles) 
                             
            for x, y, r in circles:
                if ((self.x00-15) < x < (self.x00+15)) and ((self.y00-15) < y < (self.y00+15)):
                    if r > self.r00: 
                        self.x00 = x
                        self.y00 = y
                        self.r00 = r
                        print('조건에 일치하는 변수: ', x, y, r)
                    
            if self.r00 == 0: 
                print('일치하는 접시 확인되지 않음','\n')
                ret,thresh = cv2.threshold(self.mask,235,255,0)
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)   
                c = max(contours, key=cv2.contourArea)
                (self.x00, self.y00), self.r00 = cv2.minEnclosingCircle(c)
            print('선택된 중심점 좌표:', self.x00, self.y00, self.r00)
    
    # 좌표 확인         
    def findCoordinates(self):  
        ret, thresh = cv2.threshold(self.mask,235,255,0)        
        M = cv2.moments(thresh)
        self.x00 = int(M["m10"] / M["m00"])
        self.y00 = int(M["m01"] / M["m00"])
    
    # 이미지 안에서 중심이 어딘지 확인    
    def findCenter(self):
        c_r_crop = 124
        self.img = self.img[int(self.y00) - int(c_r_crop) : int(self.y00) + int(c_r_crop), int(self.x00) - int(c_r_crop) : int(self.x00) + int(c_r_crop)]
    
    # 배경 삭제            
    def remove_background(self):  
        mainRectSize = .08
        fgSize = .01
        
        img = self.img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        new_h, new_w = img.shape[:2]
        mask = np.zeros(img.shape[:2], np.uint8)
        
        bg_w = round(new_w * mainRectSize)
        bg_h = round(new_h * mainRectSize)
        bg_rect = (bg_w, bg_h, new_w - bg_w, new_h - bg_h)
        
        fg_w = round(new_w * (1 - fgSize) / 2)
        fg_h = round(new_h * (1 - fgSize) / 2)
        fg_rect = (fg_w, fg_h, new_w - fg_w, new_h - fg_h)
        
        cv2.rectangle(mask, fg_rect[:2], fg_rect[2:4], color=cv2.GC_FGD, thickness=-1)
        bgdModel1 = np.zeros((1, 65), np.float64)
        fgdModel1 = np.zeros((1, 65), np.float64)
        
        cv2.grabCut(img, mask, bg_rect, bgdModel1, fgdModel1, 3, cv2.GC_INIT_WITH_RECT)
        cv2.rectangle(mask, bg_rect[:2], bg_rect[2:4], color=cv2.GC_PR_BGD, thickness=bg_w * 3)
        cv2.grabCut(img, mask, bg_rect, bgdModel1, fgdModel1, 10, cv2.GC_INIT_WITH_MASK)   
        
        mask_result = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
        masked = cv2.bitwise_and(img, img, mask=mask_result)
        masked[mask_result < 2] = [255, 255, 255] 
        
        self.img = masked
        self.mask = mask_result

# 정의된 함수를 통해 이미지를 전처리하는 작업 진행 

## 더러운 접시 
for image_index in range (20):
    print ("'Dirty' 카테고리 내에서 처리 완료: ","{0:04}".format(image_index),"/0019", end="\r")
    image_folder = '/kaggle/working/plates/train/dirty/{0:04}.jpg'.format(image_index) 
    img = cv2.imread(image_folder)
    
    out_img  = Remove_background_and_crop(img)
    out_img.remove_background()
    out_img.findCoordinates()
    out_img.find_circle()
    out_img.crop()    
    
print ("\n\r", end="")    

# 깨끗한 접시
for image_index in range (20):
    print ("'Clean' 카테고리 내에서 처리 완료: ","{0:04}".format(image_index),"/0019", end="\r")
    image_folder = '/kaggle/working/plates/train/cleaned/{0:04}.jpg'.format(image_index) 
    img = cv2.imread(image_folder)
    
    out_img  = Remove_background_and_crop(img)
    out_img.remove_background()
    out_img.findCoordinates()
    out_img.find_circle()
    out_img.crop()
    
print ("\n\r", end="")

# 테스트용 이미지 
for image_index in range (744):
    print ("테스트 데이터 내에서 처리 완료: ","{0:04}".format(image_index),"/0743", end="\r")
    image_folder = '/kaggle/working/plates/test/{0:04}.jpg'.format(image_index) 
    img = cv2.imread(image_folder)
    
    out_img  = Remove_background_and_crop(img)
    out_img.remove_background()
    out_img.findCoordinates()
    out_img.find_circle()
    out_img.crop_test()
    
print ("\n\r", end="") 

# -- 이미지 전처리 절차 종료 -- 

# pytorch 기반, transform 메소드 사용해 이미지 변환

## 학습용 데이터 
train_transforms = transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.09, p=0.75, interpolation=3, fill=255),
    transforms.Resize((224, 224)),    
    transforms.ColorJitter(hue=(-0.5,0.5)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(), 
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

## 검증용 데이터 
val_transforms = transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.1, p=0.8, interpolation=3, fill=255),
    transforms.Resize((224, 224)),
    transforms.ColorJitter(hue=(-0.5,0.5)),
    transforms.RandomHorizontalFlip(),     
    transforms.RandomVerticalFlip(), 
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])   

## 데이터 세트 전체 대상
dataset_transforms = {
                      'orig': transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),

                      '140': transforms.Compose([
    transforms.CenterCrop(140),
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
                     '135': transforms.Compose([
    transforms.CenterCrop(135),
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]), 
                      '130': transforms.Compose([
    transforms.CenterCrop(130),
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
                      '125': transforms.Compose([
    transforms.CenterCrop(125),
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
                      '120': transforms.Compose([
    transforms.CenterCrop(120),
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
                      '115': transforms.Compose([
    transforms.CenterCrop(115),
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
                      '110': transforms.Compose([
    transforms.CenterCrop(110),
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
                      '105': transforms.Compose([
    transforms.CenterCrop(105),
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
                      '100': transforms.Compose([
    transforms.CenterCrop(100),
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
                     '95': transforms.Compose([
    transforms.CenterCrop(95),
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
                       '90': transforms.Compose([
    transforms.CenterCrop(90),
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
                       '85': transforms.Compose([
    transforms.CenterCrop(85),
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
                       '80': transforms.Compose([
    transforms.CenterCrop(80),
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
                      '75': transforms.Compose([
    transforms.CenterCrop(75),
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),                                         
                       '70': transforms.Compose([
    transforms.CenterCrop(70),
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),                                                           
                     }
 
train_dataset = torchvision.datasets.ImageFolder(train_dir, train_transforms)
val_dataset = torchvision.datasets.ImageFolder(val_dir, val_transforms)

batch_size = 16 
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size)

# 입력받은 이미지가 어떤 내용인지 보여준다 
def show_input(input_tensor, title=''):
    image = input_tensor.permute(1, 2, 0).numpy() 
    image = std * image + mean 
    plt.imshow(image.clip(0, 1))
    plt.title(title)
    plt.show()
    plt.pause(0.1)
 
X_batch, y_batch = next(iter(train_dataloader)) 
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# 이미지를 출력하고, 이미지의 이름은 해당 파일의 이름으로 한다
for x_item, y_item in zip(X_batch, y_batch):
    show_input(x_item, title=class_names[y_item])
    
# 이미 훈련된 모델 불러오기 
model = models.resnet152(pretrained=True) 

for param in model.parameters(): 
   param.requires_grad = False 

# cuda가 사용 가능할 때는 cuda 사용하고, 그렇지 않을 때는 cpu를 사용 
model.fc = torch.nn.Linear(model.fc.in_features, 2) 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 최적화기와 게획기 정의, 호출
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), amsgrad=True, lr=0.001) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 모델 훈련 방법을 정의
def train_model(model, loss, optimizer, scheduler, num_epochs):
 
    # 손실 데이터 및 정확도 데이터 기록용 
    loss_hist = {'train':[], 'val':[]}
    acc_hist = {'train':[], 'val':[]}
 
    for epoch in range(num_epochs):
        # 진행도 측정용 함수 
        print("Epoch {}/{}: ".format(epoch, num_epochs - 1), end="")
        
        # 훈련 단계이면 훈련을 진행, 훈련 단계가 아니면(끝났으면) 평가를 진행
        for phase in ['train', 'val']:
            if phase == 'train': 
                dataloader = train_dataloader 
                scheduler.step()
                model.train()  
            else: 
                dataloader = val_dataloader 
                model.eval()  
                
            running_loss = 0. 
            running_acc = 0.
 
            for inputs, labels in tqdm(dataloader):
                inputs = inputs.to(device) 
                labels = labels.to(device) 
 
                # 모든 모델 패러미터 재료를 0으로 설정 (기존의 모델 패러미터가 새로운 모델 패러미터와 뒤섞이는 것을 미연에 방지)
                optimizer.zero_grad() 
 
                with torch.set_grad_enabled(phase == 'train'): 
                    preds = model(inputs) 
                    loss_value = loss(preds, labels)
                    preds_class = preds.argmax(dim=1) 
                
                    # 훈련 단계일 때만 재료들을 최적화기에 넘긴다 
                    if phase == 'train':
                        loss_value.backward() 
                        optimizer.step() # 다음 번 절차로
 
                running_loss += loss_value.item() 
                running_acc += (preds_class == labels.data).float().mean().data.cpu().numpy()  
 
            epoch_loss = running_loss / len(dataloader)  
            epoch_acc = running_acc / len(dataloader)
            
            print("{} Loss: {:.4f} Acc: {:.4f} ".format(phase, epoch_loss, epoch_acc), end="")
            
            # 손실과 정확도 역사에 현재 epoch의 손실, 정확도를 기록
            loss_hist[phase].append(epoch_loss)
            acc_hist[phase].append(epoch_acc)
        
    return model, loss_hist, acc_hist

# 훈련 이행
model, loss, acc = train_model(model, loss, optimizer, scheduler, num_epochs=30); 

# 시각화 표현 (정확도)
plt.rcParams['figure.figsize'] = (14, 7)

for experiment_id in acc.keys():
    plt.plot(acc[experiment_id], label=experiment_id)
    
plt.legend(loc='upper left')
plt.title('Model Accuracy')
plt.xlabel('Epoch num', fontsize=15)
plt.ylabel('Accuracy value', fontsize=15);
plt.grid(linestyle='--', linewidth=0.5, color='.7')

# 시각화 표현 (손실)
plt.rcParams['figure.figsize'] = (14, 7)

for experiment_id in loss.keys():
    plt.plot(loss[experiment_id], label=experiment_id)
    
plt.legend(loc='upper left')
plt.title('Model Loss')
plt.xlabel('Epoch num', fontsize=15)
plt.ylabel('Loss function value', fontsize=15)
plt.grid(linestyle='--', linewidth=0.5, color='.7')

test_dir = 'test'
shutil.copytree(os.path.join(data_root, 'test'), os.path.join(test_dir, 'unknown'))

# 훈련된 결과 기반으로 사후 정확도 추적
class ImageFolderWithPaths(torchvision.datasets.ImageFolder): 
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index) 
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
    
df = pd.DataFrame

for (i,tranforms) in dataset_transforms.items():
    test_dataset = ImageFolderWithPaths('/kaggle/working/test', tranforms) 
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0) 
    
    model.eval() 
    test_predictions = [] 
    test_img_paths = [] 
    
    for inputs, labels, paths in tqdm(test_dataloader): 
        inputs = inputs.to(device) 
        labels = labels.to(device)  
        with torch.set_grad_enabled(False):
            preds = model(inputs) 
        test_predictions.append(
            torch.nn.functional.softmax(preds, dim=1)[:,1].data.cpu().numpy()) 
        test_img_paths.extend(paths)
    test_predictions = np.concatenate(test_predictions)
    
    submission_df = pd.DataFrame.from_dict({'id': test_img_paths, 'label': test_predictions})
    submission_df['id'] = submission_df['id'].str.replace('/kaggle/working/test/unknown/', '')
    submission_df['id'] = submission_df['id'].str.replace('.jpg', '')
    submission_df.set_index('id', inplace=True)
    
    try:
        df = df.merge(submission_df, how='inner', on='id') 
    except BaseException: 
        df = submission_df 
    # 최종 제출용 파일로 가공하는 과정    
    submission_df['label'] = submission_df['label'].map(lambda pred: 'dirty' if pred > 0.50 else 'cleaned')
    submission_df.to_csv('submission_predict_{0}.csv'.format(i))

df['mean'] = df.mean(axis=1)
df.drop(df.columns[:-1], axis='columns', inplace=True)
df['label'] = df['mean'].map(lambda pred: 'dirty' if pred > 0.50 else 'cleaned')
df.drop(df.columns[:-1], axis='columns', inplace=True)
df.head(10)

df.to_csv('submission.csv')