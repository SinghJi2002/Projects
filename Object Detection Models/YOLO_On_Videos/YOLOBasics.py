#for explaination refer to webcam explaination. both are the same.
from ultralytics import YOLO
import cv2
import cvzone
import math
model= YOLO("../YoloWeights/yolov8n.pt")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]
# cap=cv2.VideoCapture(0)
# cap.set(3,640)
# cap.set(4,480)
cap=cv2.VideoCapture("../videos/people1.mp4")
while True:
    sucess,img=cap.read()
    results=model(img,stream=True)
    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            w,h=x2-x1,y2-y1


            conf=math.ceil(box.conf[0]*100)
            label=box.cls[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cvzone.putTextRect(img,f'{classNames[int(label)]} {conf}',(max(0,x1),max(35,y1)),scale=0.8,thickness=1)
            #cv2.putText(img,f'{conf}',(x1,y1-10),cv2.FONT_HERSHEY_PLAIN,1,(255,0,255),2)

    cv2.imshow("Image",img)
    cv2.waitKey(1)