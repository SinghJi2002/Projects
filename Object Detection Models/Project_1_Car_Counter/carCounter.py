import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
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

cap=cv2.VideoCapture("../videos/cars.mp4")

mask=cv2.imread("mask.png") #mask being applied at the top of feed to define which region of feed to be detected by yolo and not.

tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)

limits=[400,297,673,297] #x1,height,x2,height

totalCount=[]

while True:
    #reading and storing the feed.
    sucess,img=cap.read()
    #masking parts of feed to make the model work better
    imgRegion=cv2.bitwise_and(img,mask)
    #feeding the masked frame to the model
    results=model(imgRegion,stream=True)
    detections = np.empty((0, 5))
    #analysing each frame and producing neccessary attributes
    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            w,h=x2-x1,y2-y1

            conf=math.ceil(box.conf[0]*100)
            label=classNames[int(box.cls[0])]

            if((label=="car" or label=="truck" or label=="motorbike" or label=="bus") and conf>40):
                #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                #cvzone.putTextRect(img,f'{label} {conf}',(max(0,x1),max(35,y1)),scale=0.8,thickness=1,offset=3)
                #cv2.putText(img,f'{conf}',(x1,y1-10),cv2.FONT_HERSHEY_PLAIN,1,(255,0,255),2)
                currentArray=np.array([x1,y1,x2,y2,conf])
                detections=np.vstack((detections,currentArray))
    resultsTraker = tracker.update(detections)

    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(255,0,0),5)
    
    for res in resultsTraker:
        x1,y1,x2,y2,id=res
        x1,y1,x2,y2,id=int(x1),int(y1),int(x2),int(y2),int(id)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cvzone.putTextRect(img, f'{id}', (max(0, x1), max(35, y1)), scale=0.8, thickness=1, offset=3)

        cx,cy=(x1+x2)//2,(y1+y2)//2
        cv2.circle(img,(cx,cy),1,(0,0,255),cv2.FILLED)

        if(limits[0]<cx<limits[2] and limits[1]-10<cy<limits[1]+10):
            if(id not in totalCount):
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
    cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
    cv2.imshow("Image",img)
    cv2.waitKey(1)