from ultralytics import  YOLO
import yaml
import  cv2
model  = YOLO('models/model_v10.pt')
img = r"Data/test/x (4).jpg"
save_dir = "runs\model_v8" 
model.predict(img,save = True,show=True,project="xxx", name="yyy")    