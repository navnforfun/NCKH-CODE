#  delay time reset box

from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import supervision as sv
import time

model = YOLO('best.pt')
app = Flask(__name__)

# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("http://192.168.1.228:81/stream")
# cap = cv2.VideoCapture("http://192.168.1.241:4747/video")
model = YOLO("best.pt")
box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=0.5)





def gen_frames():  # generate frame by frame from camera
    cap = cv2.VideoCapture("http://192.168.1.241:4747/video")
    # cap = cv2.VideoCapture(0)
    start_time = time.time()
    time_now = start_time
    my_time = 0
    while True:
        ret,frame = cap.read()
        # cv2.imshow('frame', frame) 
        time_now = time.time() - start_time
        print(time_now)
        if((time_now - my_time ) <1):
            print("bo qua. time skip: " , (time_now - my_time ))
        else:
            print("===detect===")
            my_time = time_now
            if not ret:
                break
            result = model(frame,agnostic_nms=True)[0]
            detections = sv.Detections.from_yolov8(result)
            detections = [detection for detection in sv.Detections.from_yolov8(result) if detection[1] > 0.6]
            # print(detections)
            labels = [
                f"{model.model.names[class_id]} {confidence:0.2f}"
                for _,confidence, class_id,_
                in detections
            ]
            frame = box_annotator.annotate(scene = frame,detections=detections,labels=labels)
        # print("abc")
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/videeo_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)