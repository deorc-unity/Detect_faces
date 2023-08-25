#This module is used for image processing
import cv2


#The xml file contains all the features defined for a face
face_cas = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#Reading the image that you want to use
#The image must be present in the same directory as of the code and the xml file
img = cv2.imread("news.jpg")

#Gray scaling the image 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cas.detectMultiScale(gray, scaleFactor = 1.15, minNeighbors = 3)

#This code generates a rectangular box around the detected face
for x, y, w, h in faces:
    img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)

#using to define the dimension properties of the new image 
res = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
cv2.imshow("New Image",res)
cv2.waitKey(0)

#If you press any key the image window will close
cv2.destroyAllWindows

#Prints the dimension of the faces in array format in the terminal
print(faces)