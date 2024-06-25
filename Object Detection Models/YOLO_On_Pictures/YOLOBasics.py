from ultralytics import YOLO #ultralytics is required to import YOLO
import cv2 #cv2 is required when working with media. Here it fixes the problem of
#image disappearing after appearing itself.
model=YOLO('../YoloWeights/yolov8x.pt') #Here we have downloaded the YOLO 8 Xtra Large
#model and saved it in YoloWeights folder
results=model('images/pic3.jpeg',show=True)
#Above we are sending the model the image to analyse. YOLO will detect the objects,
#classify them and display accuracy of detection and classification.
cv2.waitKey(0)
