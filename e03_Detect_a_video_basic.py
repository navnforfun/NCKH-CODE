# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 21:32:53 2023

@author: ngocanh
"""

from ultralytics import  YOLO
import  cv2
import supervision as sv
import numpy as np
model  = YOLO('./model_v8.pt')
VIDEO_PATH = r"Data\fire8.mp4"
SAVE_FILE = r"runs\results\fire6_v3.mp4"
# old
# results = model.predict(r'C:\Users\Administrator\Documents\python\NCKH_FireDetect\Data\fire8.mp4',show=True,save = True, save_txt = True)
# # model.predict()b  
# # print(results[0].tojson())
# for r in results:
#     print(r.boxes)


# Way 2
# def process_frame(frame: np.ndarray, _) -> np.ndarray:
#     results = model(frame, imgsz=1280)[0]
#     # cv2.imshow('video',frame)

#     detections = sv.Detections.from_yolov8(results)

#     box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=1)
#     if results.boxes.id is not None:
#         detections.tracker_id = results.boxes.id.cpu().numpy().astype(int)
#     labels = [f"{tracker_id} {model.names[class_id]} {confidence:0.2f}" for _, confidence, class_id, tracker_id in detections]
#     frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
#     return frame

# sv.process_video(source_path=VIDEO_PATH, target_path=SAVE_FILE, callback=process_frame)


box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=1)
for result in  model.track(source=VIDEO_PATH,stream=True,save=True,show=True):
    frame = result.orig_img
    detections = sv.Detections.from_yolov8(result)
    detections = detections[detections.class_id != 0]
    if result.boxes.id is not None:
        detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
    labels = [
        f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        for _,confidence,class_id,tracker_id
        in detections
    ]
    frame = box_annotator.annotate(scene=frame, detections=detections,labels=labels)
    # cv2.imshow("yolov8",frame)
    if(cv2.waitKey(30)==27):
        cv2.destroyAllWindows()
        break

