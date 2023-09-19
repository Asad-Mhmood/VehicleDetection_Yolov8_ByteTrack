import math
import cv2
import cvzone
import numpy as np


from ultralytics import YOLO
cap = cv2.VideoCapture('4.mp4') # Change this video name according to your requirement

model = YOLO("yolov8m.pt")


classNames = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]



limits = [430,600,1550,600] # line 4  Change these limits according to your requirement
counter={
    'car':0,
    'motorcycle':0,
    'truck':0,
    'bus':0,
    'bicycle':0

}
totalcounts=[]


while True:

    success,img = cap.read()

    img = cv2.resize(img,(1600,900))
    result = model.track(img,stream=True,tracker="bytetrack.yaml",persist=True)



    for r in result:


        boxes = r.boxes

        for box in boxes:
            #print(box)
            #Bounding Box
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)

            #Bounding Box 2
            w,h = x2-x1,y2-y1
            #x1,y1,w,h = box.xywh[0]
            #bbox = int(x1), int(y1), int(w), int(h)
            bbox = x1,y1,w,h
            #cvzone.cornerRect(img,bbox,l=5,rt=3)


            #Confidence
            conf = math.ceil((box.conf[0]*100))/100
            #cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(20,y1)),scale=1,thickness=1)



            #Class Name
            cls=int(box.cls[0])
            currentClass=classNames[cls]

            id = int(box.id[0])



            if currentClass == 'car' or currentClass == 'motorcycle' or currentClass == 'bus' or currentClass == 'truck' or currentClass=='bicycle':
                pass
                cvzone.cornerRect(img,bbox,l=9,rt=5)
                cvzone.putTextRect(img, f'id : {id} {classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.8, thickness=1, offset=3)
                cx, cy = x1 + w // 2, y1 + h // 2
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)


                if limits[0] < cx < limits[2] and limits[1]-20 < cy < limits[1]+20:
                    # if currentClass == 'car' or currentClass == 'motorcycle' or currentClass == 'bus' or currentClass == 'truck' or currentClass=='bicycle':

                    if totalcounts.count(id) == 0:
                        totalcounts.append(id)
                        counter[currentClass] += 1
                        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (255, 255, 255), 5)
                        cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

                #cvzone.cornerRect(img, bbox, l=9, rt=5)
                #cvzone.putTextRect(img, f'{id}', (max(0, x1), max(35, y1)), scale=0.8,thickness=1, offset=3)
                #currentArray = np.array([x1, y1, x2, y2, conf])





    y_offset = 80
    for vehicle_type, count in counter.items():

        cvzone.putTextRect(img, f'{vehicle_type.capitalize()} : {count}', (50, y_offset),thickness=2)
        y_offset += 50


    cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break





cap.release()
cap.destroyAllWindows()
