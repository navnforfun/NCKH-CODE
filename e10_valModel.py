from ultralytics import  YOLO
import yaml
import  cv2
model  = YOLO('models/model_v3.pt')
for i in range(1,31):
    img = f"Data/test/x ({i}).jpg"
    model.predict(img,save = True,show=True,project="model_v3")    