#  delay time reset box
import requests
from flask import Flask, render_template, Response,request
import cv2
from ultralytics import YOLO
import supervision as sv
import time
import dweepy
import threading

model = YOLO('models/model_v11.pt')
app = Flask(__name__)
box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=0.5)

#  send notice to dweet
def alertNotice(state):
    dweepy.dweet_for("dnu_cntt1504_nhom2_notice",{"state":"Fire","duration":"3000","warning":"yes"})
    requests.get("http://localhost:5257/Emails/MySendMailAPI?notice=fire")


def gen_frames(conf,lever,src):  # generate frame by frame from camera
    # cap.release()
    # cap = cv2.VideoCapture("http://192.168.1.241:4747/video")
    # cap = cv2.VideoCapture("http://14.161.31.172:81/asp/video.cgi")
    cap = cv2.VideoCapture(src)
    # skip frame tránh lag
    start_time = time.time()
    time_now = start_time
    my_time = 0
    detections = None
    labels = []
    while True:

        ret,frame = cap.read()
        # cv2.imshow('frame', frame) 
        time_now = time.time() - start_time
        # print(time_now)
        isDraw = True
        # skip frame tránh lag
        if((time_now - my_time ) <2):
            # print("bo qua. time skip: " , (time_now - my_time ))
            pass
        else:
            isDraw =False
            my_time = time_now
            if not ret:
                break
            result = model(frame,agnostic_nms=True,verbose=False)[0]
            detections = sv.Detections.from_yolov8(result)
            # check confident
            detections = [detection for detection in sv.Detections.from_yolov8(result) if detection[1] > (conf/100)]
            if(detections != None):
                # Optional: Calculate area
                for detection in detections:
                    bounding_box_coords = detection[0]  # Extract coordinates
                    width = bounding_box_coords[2] - bounding_box_coords[0]  # Calculate width
                    height = bounding_box_coords[3] - bounding_box_coords[1]  # Calculate height
                    area = width * height
                    # tính độ lớn cơ bản của lửa so với 1/4 màn hình 
                    check = (area/76000*100)>lever
                    if(check):
                        print(f"=== chay to: {area/76000*100} vs {lever} ===")
                        threading.Thread(target=alertNotice,args=("Fire",)).start()     
                        print(f"Bounding box area: {area:.2f} pixels squared")
                        isDraw= True
                    else:
                        isDraw= False
         
            labels = [
                f"{model.model.names[class_id]} {confidence:0.2f}"
                for _,confidence, class_id,_
                in detections
            ]
            if(isDraw):
                frame = box_annotator.annotate(scene = frame,detections=detections,labels=labels)
            else:
                detections = None
        
   
        if(detections != None):
            frame = box_annotator.annotate(scene = frame,detections=detections,labels=labels)
        
        try:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        except:
            pass
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
@app.route('/video_feed')
def video_feed():
    print(request.query_string)
    conf =  request.args.get('conf')
    lever =  request.args.get('lever')
    src =  request.args.get('src')
    if conf == None:
        conf = 50
    else:
        conf = int(conf)
    if lever == None:
        lever = 20
    else:
        lever = int(lever)
 
    if src == None or src==""  :
        src = 0
    print("=====")
    print(src)
    
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(conf,lever,src), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    # app.run(debug=True)
    app.run()