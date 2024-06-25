#We are basically using the video detection code
from ultralytics import YOLO
import cv2
import cvzone
import math
model= YOLO("../YoloWeights/best.pt")
#Below are predefined yolo classes we are using these just for inference.
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']
colorClass=[(0,255,0),(0,255,0),(255,0,0),(255,0,0),(255,0,0),(0,0,255),(0,0,255),(0,255,0),(0,0,255),(0,0,255)]
#Since working with webcam, following thing need to be defined.
cap=cv2.VideoCapture("../videos/ppe-3-1.mp4")#Which camera to use for capturing the videos. By default, 0 is used for internal web cam.
#Below we are setting dimension of window that will appear when the web cam is lauched.
#While true remain true until web cam is on
while True:
    sucess,img=cap.read()#cap.read captures frames read from web cam and stores them in img. success is a boolean which stores true
    #if web cam is on and store false when web cam is off.
    results=model(img,stream=True) #Here we are passing the frames captured by webcam to the model for analysis. Since frames are
    #continously being recorded, stream is set to true.

    #Here is a bit of context of how the video is analysed by yolo. We feed the stream to model, model detects, classifies the objects
    #in the frame with varying confidence. But unlike photos, where the boxes where displayed with class of object in the box and the
    #confidence level, for video analysis it stores such values in different attributes, but it doesn't display it on the stream. The
    #code below is for displaying all those details onto the stream.
    for r in results:
        #Lets say webcam supplies a feed to model, that has a human and a smartphone. So there will be two boxes for that particular
        #frame. To access these boxes we use the boxes attribute. Note at any time results will hold multiple frames, hence to iterate
        #frame r iterator is being used.
        boxes=r.boxes
        for box in boxes:
            #Now each box of the frame has various attributes, such as xyxy(co-ordinates of leftmost-top and rightmosy-bottom points of
            # box),score(confidence),id(numeral) associated to class and many more.

            #xyxy returns a array(0,4), which stores the coordinates. We are storing those coordinates in variables x1,x2,y1,y2,
            #converting them to int and then drawing a rectangular box using these coordinates using cv2 which is displayed in live feed.
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            w,h=x2-x1,y2-y1


            #next we are accessing confidence via conf attribute. confidence is in range(0,1) hence we convert it to percentage,
            #take the ceiling int.
            conf=math.ceil(box.conf[0]*100)
            #yolo detects object in the box and associates an id with it for identification. cls attribute returns that id.
            label=box.cls[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), colorClass[int(label)], 3)
            #using cvzone this time, we are drawing a rectangle and displaying these values alongside the boxes.
            cvzone.putTextRect(img,f'{classNames[int(label)]} {conf}',(max(0,x1),max(35,y1)),scale=0.8,thickness=1)
            #cv2.putText(img,f'{conf}',(x1,y1-10),cv2.FONT_HERSHEY_PLAIN,1,(255,0,255),2)

    #feed to be shown live.
    cv2.imshow("Image",img)
    cv2.waitKey(1)