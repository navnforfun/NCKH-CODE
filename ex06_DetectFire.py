import cv2
from ultralytics import YOLO
import supervision as sv
import time
start_time = time.time()
time_now = start_time
my_time = 0
# cap = cv2.VideoCapture("http://192.168.1.228:81/stream")
cap = cv2.VideoCapture("http://192.168.1.241:4747/video")
# cap = cv2.VideoCapture(0)

model = YOLO("best.pt")
box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=0.5)

while True:
    time_now = time.time() - start_time
    print(time_now)
    # Skip frame
    # if((time_now - my_time ) <1):
    #     ret,frame = cap.read()
    #     print("bo qua. time skip: " , (time_now - my_time ))
    #     continue
    my_time = time_now
    ret,frame = cap.read()
    result = model(frame,agnostic_nms=True)[0]
    detections = sv.Detections.from_yolov8(result)
    detections = [detection for detection in sv.Detections.from_yolov8(result) if detection[1] > 0.6]
    print(detections)
    labels = [
        f"{model.model.names[class_id]} {confidence:0.2f}"
        for _,confidence, class_id,_
        in detections
    ]
    frame = box_annotator.annotate(scene = frame,detections=detections,labels=labels)
    cv2.imshow('yolov8',frame)

    # print(frame.shape)
    if(cv2.waitKey(30) == 27):
        cap.release()
        cv2.destroyAllWindows()
        break