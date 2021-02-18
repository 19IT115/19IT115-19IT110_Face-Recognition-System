import cv2

cascade_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
captureface = cv2.VideoCapture(0)

while True:
    # It will Capture frame-by-frame
    ret, frame = captureface.read()

    gray = cv2.cvtColor(frame, 0)
    detections = cascade_classifier.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
    if(len(detections) > 0):
        (x,y,w,h) = detections[0]
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)


    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# It will release the capture
captureface.release()
cv2.destroyAllWindows()
