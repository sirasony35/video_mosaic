import cv2
import numpy as np


def load_yolo():
    net = cv2.dnn.readNet("model/yolov3.weights", "model/yolov3.cfg")
    layers_names = net.getLayerNames()
    # OpenCV 4.x에서 getUnconnectedOutLayers()가 array of indices를 반환
    unconnected_out_layers = net.getUnconnectedOutLayers()
    output_layers = [layers_names[i[0] - 1] if isinstance(i, np.ndarray) else layers_names[i - 1] for i in unconnected_out_layers]
    return net, output_layers


def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs


def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.5 and class_id == 0:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = center_x - w / 2
                y = center_y - h / 2
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes


def apply_mosaic(img, boxes, level=15):
    for box in boxes:
        x, y, w, h = box
        x, y, w, h = int(x), int(y), int(w), int(h)

        # 이미지 경계를 벗어나지 않도록 조정
        x_end = min(x + w, img.shape[1])
        y_end = min(y + h, img.shape[0])
        w = x_end - x
        h = y_end - y

        # ROI 추출
        roi = img[y:y_end, x:x_end]
        if roi.size == 0:
            continue  # 빈 영역을 처리하는 경우, 이를 건너뛰기

        # 모자이크 처리
        roi = cv2.resize(roi, (level, level), interpolation=cv2.INTER_LINEAR)
        roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_NEAREST)

        # 모자이크 적용
        img[y:y_end, x:x_end] = roi

    return img


# Load YOLO
net, output_layers = load_yolo()

# Load video
cap = cv2.VideoCapture('video_data/yoonbin_3.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Prepare output video writer
out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 24.0, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape
    boxes = detect_objects(frame, net, output_layers)[1]
    boxes = get_box_dimensions(boxes, height, width)
    frame = apply_mosaic(frame, boxes)

    out.write(frame)
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
