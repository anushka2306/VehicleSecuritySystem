import cv2
import os
import numpy as np
import facerecognition as fr

test_img=cv2.imread('C:\\Users\HP\\Desktop\\opencv\\opencv-master\\test_images\\anu1.jpg')#test_img path
faces_detected,gray_img=fr.faceDetection(test_img)
print("faces_detected:",faces_detected)

#for (x, y, w, h) in faces_detected:
#    img = cv2.rectangle(test_img, (x, y), (x+w, y+h), (0, 255, 0), 5)

faces,faceID=fr.labels_for_training_data('C:\\Users\\HP\\Desktop\\opencv\\opencv-master\\training_images')
face_recognizer=fr.train_classifier(faces,faceID)
face_recognizer.save('C:\\Users\\HP\\Desktop\\opencv\\opencv-master\\trainingData.yml')
#face_recognizer=cv2.face.LBPHFaceRecognizer_create()
#face_recognizer.read('C:\\Users\\HP\\Desktop\\opencv\\opencv-master\\trainingData.yml')

name={0:"Alia",1:"Anushka"}#creating dictionary containing names for each label

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
    print("confidence:",confidence)
    print("label:",label)
    fr.draw_rect(test_img,face)
    predicted_name = name[label]
    if (confidence > 37):  # If confidence more than 37 then don't print predicted face text on screen
        continue
    fr.put_text(test_img, predicted_name, x, y)

resized_img=cv2.resize(test_img,(int(test_img.shape[1]), int(test_img.shape[0])))
cv2.imshow("face dtecetion tutorial",resized_img)
cv2.waitKey(0)#Waits indefinitely until a key is pressed
cv2.destroyAllWindows()
