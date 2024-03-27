import cv2
import numpy as np
import os 

def facerec(img, minW, minH, faceCascade, recognizer, names):
    

    font = cv2.FONT_HERSHEY_SIMPLEX

    # # Initialize and start realtime video capture
    # cam = cv2.VideoCapture(0)
    # cam.set(3, 640) # set video widht
    # cam.set(4, 480) # set video height

    # # Define min window size to be recognized as a face
    # minW = 0.1 * cam.get(3)
    # minH = 0.1 * cam.get(4)


        #ret, img = cam.read()
        #img = cv2.flip(img, 1) # Flip vertically

    try:     
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))

        for (x, y, w, h) in faces:

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)

            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            # Check if confidence is less than 100 ==> "0" is perfect match 
            if confidence < 100:
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            
            cv2.putText(img, str(id), (x+5, y-5), font, 1, (255, 255, 255), 1)
            cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (255, 255, 0), 1)  

    except IndexError as e:
        print("ERROR REC", e.args)
        
        # cv2.imshow('camera', img) 

        # k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        # if k == 27:
        #     break

    # Do a bit of cleanup
    # print("\n [INFO] Exiting Program and cleanup stuff")
    # cam.release()
    # cv2.destroyAllWindows()
