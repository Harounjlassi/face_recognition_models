import cv2
import os
import face_recognition
import numpy as np
import mysql.connector
from datetime import datetime
import threading

class PriorityFaceDetector:
    def __init__(self, db_config, images_path):
        self.db_config = db_config
        self.images_path = images_path
        self.known_face_encodings = []
        self.known_face_info = []
        self.priority_colors = {
            'HIGH': (0, 0, 255),    # Red
            'MEDIUM': (0, 165, 255),# Orange
            'LOW': (0, 255, 0),     # Green
            'DEFAULT': (255, 255, 255) # White
        }
        self.lock = threading.Lock()
        
        # Initialize database connection and load encodings
        self.db_connection = mysql.connector.connect(**self.db_config)
        self.load_encoding_images()
        self.update_task_priorities()

    def load_encoding_images(self):
        """Load face encodings from images and map to database users"""
        cursor = self.db_connection.cursor(dictionary=True)
        valid_images = 0

        try:
            # Get all users from database
            cursor.execute("SELECT id, username FROM user")
            users = cursor.fetchall()
            
            # Create username to user mapping (case insensitive)
            user_map = {user['username'].lower(): user for user in users}
            
            # Process each image in the directory
            for img_file in os.listdir(self.images_path):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                # Extract username from filename (without extension)
                base_name = os.path.splitext(img_file)[0]
                username_from_file = base_name.split('.')[0].lower()
                
                # Find matching user
                user = user_map.get(username_from_file)
                if not user:
                    print(f"Warning: No database user found for image {img_file}")
                    continue
                
                # Process the image
                img_path = os.path.join(self.images_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not read image {img_file}")
                    continue
                
                # Convert to RGB and find face encodings
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(rgb_img)
                
                if len(face_encodings) == 0:
                    print(f"Warning: No face found in {img_file}")
                    continue
                
                # Store user info and encoding
                with self.lock:
                    self.known_face_info.append({
                        'user_id': user['id'],
                        'username': user['username'],
                        'encoding': face_encodings[0]
                    })
                    self.known_face_encodings.append(face_encodings[0])
                    valid_images += 1
                    print(f"Mapped {img_file} to user {user['username']} (ID: {user['id']})")

            if valid_images == 0:
                raise ValueError("No valid face images could be mapped to users")
                
            print(f"Successfully loaded {valid_images} face encodings")

        except mysql.connector.Error as err:
            print(f"Database error: {err}")
            raise
        finally:
            cursor.close()

    def update_task_priorities(self):
        """Fetch latest task priorities from database"""
        cursor = self.db_connection.cursor(dictionary=True)
        try:
            cursor.execute("""
                SELECT u.id as user_id, t.priorite 
                FROM user u
                LEFT JOIN projet_tache t ON u.id = t.assignee_id
            """)
            
            self.task_priorities = {}
            for row in cursor.fetchall():
                user_id = row['user_id']
                priority = row['priorite'].upper() if row['priorite'] else 'DEFAULT'
                self.task_priorities[user_id] = priority
                
            print("Updated task priorities from database")
        except Exception as e:
            print(f"Database error: {e}")
        finally:
            cursor.close()

    def detect_faces(self, frame):
        """Detect faces and return with priority information"""
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_info = []
        for face_encoding in face_encodings:
            # Compare with known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            user_id = None
            priority = 'DEFAULT'
            
            if True in matches:
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    matched_info = self.known_face_info[best_match_index]
                    name = matched_info['username']
                    user_id = matched_info['user_id']
                    priority = self.task_priorities.get(user_id, 'DEFAULT')
            
            face_info.append({
                'name': name,
                'user_id': user_id,
                'priority': priority
            })
        
        # Scale face locations back to original size
        face_locations = [(top*4, right*4, bottom*4, left*4) for (top, right, bottom, left) in face_locations]
        
        return face_locations, face_info

    def process_frame(self, frame):
        """Process each frame and draw priority-colored rectangles"""
        # Update priorities every 30 frames (~1 second at 30fps)
        if cv2.getTickCount() % 30 == 0:
            self.update_task_priorities()
        
        # Detect faces and their priorities
        face_locations, face_info = self.detect_faces(frame)
        
        # Draw rectangles and labels
        for (top, right, bottom, left), info in zip(face_locations, face_info):
            color = self.priority_colors.get(info['priority'], self.priority_colors['DEFAULT'])
            
            # Draw rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label background
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            
            # Draw text
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f"{info['name']} : {info['priority']} priority tasks)", 
                    (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return frame

    def run(self):
        """Main loop for real-time processing"""
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            frame = self.process_frame(frame)
            
            # Display
            cv2.imshow('Priority Face Detection', frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

    def __del__(self):
        if hasattr(self, 'db_connection') and self.db_connection.is_connected():
            self.db_connection.close()

# Configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'club-sync'
}

# Run the detector
detector = PriorityFaceDetector(db_config, images_path="images/")
detector.run()