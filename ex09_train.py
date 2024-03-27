# pip install roboflow --quiet
# !yolo task=detect mode=train model=yolov8x.pt data=FireDetection_ver2-2\data.yaml epochs=300 imgsz=800 plots=True

# yolo task=detect mode=train model=yolov8x.pt data=C:\Users\Administrator\Documents\python\NCKH_FireDetect\FireDetection_ver2-2\data.yaml epochs=300 imgsz=800 plots=True
# !pip install roboflow

# data
# !pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="mSXH4RQO2rxJMSZmPMQ5")
project = rf.workspace("vn-na-7glre").project("firedetection_ver2")
dataset = project.version(4).download("yolov8")


# from ultralytics import YOLO
 
# # Load the model.
# model = YOLO('yolov8n.pt')
 
# # Training.
# results = model.train(
#    data=r'C:\Users\Administrator\Documents\python\NCKH_FireDetect\FireDetection_ver2-2\data.yaml',
#    imgsz=640,
#    epochs=100,

# )
