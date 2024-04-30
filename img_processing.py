import dlib
import numpy as np
import cv2
import os
from tqdm import tqdm

# Load Model
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('model/dlib_face_recognition_resnet_model_v1.dat')

# Define image folder
image_folder = 'img_data'

all_descriptors = []

for filename in tqdm(os.listdir(image_folder)):
    if filename.endswith(".jpg"):
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 얼굴 검출
        dets = detector(image_rgb, 1)
        for k, d in enumerate(dets):
            shape = sp(image_rgb, d)
            face_descriptor = facerec.compute_face_descriptor(image_rgb, shape)
            np_face_descriptor = np.array(face_descriptor)

            ## 모든 특징을 리스트에 추가
            all_descriptors.append(np_face_descriptor)

# 전체 특징 리스트를 파일로 저장
np.save(os.path.join(image_folder, 'all_face_descriptors.npy'), all_descriptors)