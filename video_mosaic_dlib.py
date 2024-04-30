import cv2
import dlib
import numpy as np

## Dlib의 얼굴 인식기와 얼굴 랜드마크 검출기 로드
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('model/dlib_face_recognition_resnet_model_v1.dat')

# 여러 개의 얼굴 특징 로드 (Numpy 배열)
known_face_descriptors = np.load('img_data/all_face_descriptors.npy')

def find_faces(img):
    dets = detector(img, 1)
    return dets

def extract_face_descriptor(img, face):
    shape = sp(img, face)
    face_descriptor = facerec.compute_face_descriptor(img, shape)
    return np.array(face_descriptor)

def apply_mosaic(img, face):
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    face_img = img[y:y+h, x:x+w]
    face_img = cv2.resize(face_img, (w // 10, h // 10))
    face_img = cv2.resize(face_img, (w, h), interpolation=cv2.INTER_AREA)
    img[y:y+h, x:x+w] = face_img

def is_match(face_descriptor, threshold=0.7):
    distances = np.linalg.norm(known_face_descriptors - face_descriptor, axis=1)
    return np.any(distances < threshold)

# 비디오 파일 로드
cap = cv2.VideoCapture('video_data/yoonbin_with_mom.mp4')

# 출력 비디오의 프레임 크기
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# 결과 비디오 파일 생성
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 24.0, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = find_faces(rgb_frame)

    for face in faces:
        descriptor = extract_face_descriptor(rgb_frame, face)

        # 특정 인물과의 거리가 임계값보다 작을 경우 모자이크 처리
        if is_match(descriptor):
            apply_mosaic(frame, face)

    out.write(frame) # 처리된 프레임을 결과 비디오 파일에 저장
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

