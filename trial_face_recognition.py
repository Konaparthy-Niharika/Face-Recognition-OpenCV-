import numpy as np
import cv2
import os
import urllib.request
import imutils

haarcascade_file='haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haarcascade_file)
datasets='Datasets'
print('Training....')
(images,labels,names,id) = ([],[],{},0)

for (subdirs,dirs,files) in os.walk(datasets):
    for subdir in dirs:
        names[id]=subdir
        subjectpath=os.path.join(datasets,subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label=id
            images.append(cv2.imread(path,0))
            labels.append(int(label))
        id+=1

(images,labels) = [np.array(lis) for lis in  [images,labels]]
print(images,labels)
(width,height) = (130,100)

model = cv2.face.LBPHFaceRecognizer_create()
#model = cv2.face.FisherFaceRecognizer_create()

model.train(images,labels)

counts = 0

url='http://192.168.0.105:8080/shot.jpg'

while True:
    imagePath = urllib.request.urlopen(url)
    imageNp = np.array(bytearray(imagePath.read()),dtype=np.uint8)
    image = cv2.imdecode(imageNp,-1)
    image2 = imutils.resize(image,width=450)
    grayimage=cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(grayimage,1.3,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(image2,(x,y),(x+w,y+h),(0,255,0),2)
        face = grayimage[y:y+h , x:x+w]
        face_resize = cv2.resize(face,(width,height))

        prediction = model.predict(face_resize)
        cv2.rectangle(image2,(x,y),(x+w, y+h),(0,255,0),2)
        if prediction[1] < 100:
            cv2.putText(image2,'%s - %0.f' %(names[prediction[0]],prediction[1]),(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255))
            print(names[prediction[0]])
            counts = 0
        else:
            counts+=1
            cv2.putText(image2,'UNKNOWN',(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255))
            if counts > 100:
                print('Unknown Person')
                cv2.imwrite('unknown.jpg',image2)
                counts = 0


    cv2.imshow('FaceRecognition',image2)
    key=cv2.waitKey(10)
    if key == 27:
        break
cv2.destroyAllWindows()
