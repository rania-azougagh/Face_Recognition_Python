import cv2
import os

# Initialize recognizer and load the trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Mapping IDs to Names (Index 1 = User 1, Index 2 = User 2)
names = ['None', 'Rania', 'User_2'] 

cap = cv2.VideoCapture(0)

print("\n System is live. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Face detection with optimized parameters
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=10, minSize=(100, 100))

    for (x, y, w, h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        # Confidence: Lower value means higher accuracy in LBPH
        if confidence < 75:
            user_name = names[id]
            accuracy = f"{round(100 - confidence)}%"
            color = (0, 255, 0) # Green for identified
        else:
            user_name = "Unknown"
            accuracy = ""
            color = (0, 0, 255) # Red for unknown

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{user_name} {accuracy}", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    cv2.imshow('Face Recognition Project', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
