from ultralytics import  YOLO
import yaml
import  cv2
# model  = YOLO('yolov8m.pt')
# model.predict(r'D:\DNU\NghienCuuKhoaHoc\Code\dog-facts-cat-facts.webp',save = True, save_txt = True)
lis = open(r"D:\DNU\NghienCuuKhoaHoc\Code\runs\detect\predict4\labels\dog-facts-cat-facts.txt").readlines()
li = lis[0].split()
xc,yc,nw,nh = float(li[1]),float(li[2]),float(li[3]),float(li[4])
img = cv2.imread(r"D:/DNU/NghienCuuKhoaHoc/Code/dog-facts-cat-facts.webp")
h,w = img.shape[0],img.shape[1]

# position

# x center
xc *=w 

# y center
yc *= h

# width box
nw *= w 

# height box
nh *=h

top_left = (int(xc - nw/2),int(yc-nh/2))
bottom_right =  (int(xc + nw/2),int(yc+nh/2))

img = cv2.rectangle(img, top_left,   bottom_right,(0,255,0),3)
cv2.imshow("winname", img)
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()