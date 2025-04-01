import cv2
from simple_facerec import SimpleFacerec

# Initialize with verbose output
print("Initializing face recognition system...")
sfr = SimpleFacerec()

# Load encodings with progress feedback
print("Loading face encodings from images folder...")
try:
    sfr.load_encoding_images("images/")
except Exception as e:
    print(f"Error loading images: {e}")
    exit()

# Camera setup with auto-detection
def find_working_camera():
    for i in range(4):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Found camera at index {i}")
            return cap
    return None

print("Searching for camera...")
cap = find_working_camera()
if not cap:
    print("Error: No working camera found!")
    exit()

# Main recognition loop
print("Starting face recognition. Press ESC to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame")
        break
    
    # Process frame
    face_locations, face_names = sfr.detect_known_faces(frame)
    
    # Visualize results
    for (y1, x2, y2, x1), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow("Face Recognition", frame)
    
    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
print("System shut down successfully")