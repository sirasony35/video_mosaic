import cv2
import numpy as np

def apply_mosaic(img, boxes, level=15):
    for (x, y, w, h) in boxes:
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (level, level), interpolation=cv2.INTER_LINEAR)
        roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_NEAREST)
        img[y:y+h, x:x+w] = roi
    return img

# Haar Cascade 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 비디오 로드
cap = cv2.VideoCapture('video_data/yoonbin_3.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# 출력 비디오 준비
out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 24.0, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 얼굴 검출
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 모자이크 처리
    frame = apply_mosaic(frame, faces)

    out.write(frame)
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
