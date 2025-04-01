import cv2
import os
import face_recognition
from PIL import Image
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        
    def load_encoding_images(self, images_path):
        """
        Load encoding images from path
        """
        if not os.path.exists(images_path):
            raise FileNotFoundError(f"Directory '{images_path}' does not exist")
            
        valid_images = 0
        
        for img_file in os.listdir(images_path):
            if img_file.split('.')[-1].lower() not in ['jpg', 'jpeg', 'png']:
                continue
                
            img_path = os.path.join(images_path, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Warning: Could not read image {img_file}")
                continue
                
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_img)
            
            if len(face_encodings) == 0:
                print(f"Warning: No face found in {img_file}")
                continue
                
            # Get filename without extension
            name = os.path.splitext(img_file)[0]
            
            self.known_face_encodings.append(face_encodings[0])
            self.known_face_names.append(name)
            valid_images += 1
            
        if valid_images == 0:
            raise ValueError("No valid face images found in the directory")
            
        print(f"Successfully loaded {valid_images} face encodings")

    # Rest of the class remains the same...
    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                
            face_names.append(name)
        
        face_locations = np.array(face_locations)
        face_locations = face_locations * 4
        return face_locations.tolist(), face_names