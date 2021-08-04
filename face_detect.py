import cv2 as cv

resized = cv.imread('people.jpg')

# resized = cv.resize(img, (img.shape[1]//2, img.shape[0]//2))
# cv.imshow('Messi', resized)

gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Messi', gray)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors=2)

print(len(faces_rect))

for (x,y,w,h) in faces_rect:
    cv.rectangle(resized, (x,y), (x+w, y + h), (0, 255, 0), thickness=2)


cv.imshow('Detected faces', resized)


cv.waitKey(0)