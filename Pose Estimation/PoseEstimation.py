from unittest import result
import cv2
import mediapipe as mp
from numpy import imag

cap = cv2.VideoCapture('./1.mp4')
#cap = cv2.VideoCapture(0)
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
while True:
    success, image = cap.read()
    image = cv2.resize(image,(1280,1020))
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(image, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, landmark in enumerate(results.pose_landmarks.landmark):
            height, width, channel = image.shape
            cx, cy = int(landmark.x * width), int(landmark.y * height)
            cv2.circle(image, (cx,cy), 3,(0,0,0), cv2.FILLED)

    cv2.imshow("Image", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()