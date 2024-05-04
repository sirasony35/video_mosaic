import dlib
import numpy as np
import cv2

# dlib 모델 로드
detector = dlib.get_frontal_face_detector()  # 얼굴 검출기
sp = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('model/dlib_face_recognition_resnet_model_v1.dat')

# 특정 인물의 얼굴 특징 로드
known_face_descriptor = np.load('img_data/all_face_descriptors.npy')

def apply_mosaic(img, rect, level=15):
    """
    주어진 얼굴 영역(rect)에 모자이크를 적용하는 함수
    """
    x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
    face_img = img[y:y+h, x:x+w]
    face_img = cv2.resize(face_img, (level, level), interpolation=cv2.INTER_LINEAR)
    face_img = cv2.resize(face_img, (w, h), interpolation=cv2.INTER_NEAREST)
    img[y:y+h, x:x+w] = face_img

def recognize_faces(frame):
    """
    프레임에서 얼굴을 검출하고 인식하여 특정 인물의 얼굴에 모자이크 처리
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        shape = sp(gray, face)
        face_descriptor = facerec.compute_face_descriptor(frame, shape)
        distance = np.linalg.norm(np.array(face_descriptor) - known_face_descriptor)
        if distance < 0.3:  # 유사도 임계값 설정
            apply_mosaic(frame, face)

# 비디오 파일 로드 및 처리
cap = cv2.VideoCapture('video_data/yoonbin_3.mp4')
out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    recognize_faces(frame)
    out.write(frame)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
