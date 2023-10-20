import cv2
import dlib

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(0)

while True:
  ret, frame = cap.read()
  if not ret:
    break
  
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = detector(gray)
  
  for face in faces:
    landmarks = predictor(gray, face)
    for n in range(68):
      x, y = landmarks.part(n).x, landmarks.part(n).y
      cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
      
  cv2.imshow("Facial Landmark Detection")
  
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()