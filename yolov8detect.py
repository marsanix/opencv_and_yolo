from ultralytics import YOLO
import cv2
import math 
from face_recognition3 import facerec
import face_recognition
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Inisialisasi dictionary untuk menyimpan id dan nama
id_names = {}

# Mendapatkan daftar file dalam direktori dataset
dataset_dir = 'dataset/'
files = os.listdir(dataset_dir)

# Memproses setiap file dalam direktori dataset
for file in files:
    if file.endswith('.jpg'):
        # Membagi nama file untuk mendapatkan id dan nama
        parts = file.split('.')
        id = int(parts[1])  # Mengambil id dari nama file
        name = parts[0]     # Mengambil nama dari nama file (tanpa id)

        # Menambahkan id dan nama ke dalam dictionary
        if id not in id_names:
            id_names[id] = name

# Mengurutkan id_names berdasarkan id
id_names = dict(sorted(id_names.items()))

# Membuat array names berdasarkan id_names
names = ['None'] + [id_names[i] for i in range(1, len(id_names) + 1)]
#names = ['None'] + [id_names.get(i, 'Unknown') for i in range(1, len(id_names) + 1)]

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

minW = 0.1 * cap.get(3)
minH = 0.1 * cap.get(4)

# model
model = YOLO("yolov8n.pt")

# object classes
classNames = ["person"]


while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("cls", cls)

            try:
            
                if(classNames[cls] == "person"):

                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)
                    print("Class name -->", classNames[cls])

                    facerec(img, minW, minH, faceCascade, recognizer, names)
                    # faceRecognize(img, known_faces, known_names)

            
            except IndexError as e:
                print("ERROR", e.args)

    cv2.imshow('Webcam', img)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()