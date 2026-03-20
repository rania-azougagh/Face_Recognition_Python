import cv2
import os

# Create dataset directory if it doesn't exist to store face images
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Initialize the camera and the face detector
cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Input a numeric ID for the person (e.g., 1 for Rania, 2 for Brother)
user_id = input('\n Enter User ID (numeric) and press <Enter>: ')
count = 0

print("\n Starting face capture. Please look at the camera...")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        # Save the captured face image into the dataset folder
        # Format: User.[ID].[SampleNumber].jpg
        file_path = f"dataset/User.{user_id}.{count}.jpg"
        cv2.imwrite(file_path, gray[y:y+h, x:x+w])
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"Captured: {count}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow('Data Collection', frame)

    # Stop after 100 images or if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 100:
        break

print("\n Successfully captured and saved images.")
cap.release()
cv2.destroyAllWindows()
