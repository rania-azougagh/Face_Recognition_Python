import cv2
import numpy as np
import os
from PIL import Image

# Path for the captured face images
path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]     
    face_samples = []
    ids = []

    for image_path in image_paths:
        # Load image and convert to grayscale
        PIL_img = Image.open(image_path).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        
        # Get user ID from the filename
        user_id = int(os.path.split(image_path)[-1].split(".")[1])
        
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y+h, x:x+w])
            ids.append(user_id)

    return face_samples, ids

print("\n  Training the model. Please wait...")
faces, ids = get_images_and_labels(path)
recognizer.train(faces, np.array(ids))

# Save the trained model to a file
recognizer.write('trainer.yml') 

print(f"\n  Training complete. {len(np.unique(ids))} faces recognized.")
