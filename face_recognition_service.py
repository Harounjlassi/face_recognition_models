import cv2
import os

# 1. Load the Haar Cascade classifier
alg = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)

# 2. Create directory to store faces if it doesn't exist
output_dir = "stored-faces"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 3. Load the input image
file_name = "hq720.jpg"  # Replace with your image path
img = cv2.imread(file_name)

if img is None:
    print(f"Error: Could not load image {file_name}")
    exit()

# 4. Convert to grayscale (face detection works better on grayscale)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Fixed conversion

# 5. Detect faces
faces = haar_cascade.detectMultiScale(
    gray_img, 
    scaleFactor=1.05, 
    minNeighbors=5,  # Increased for better accuracy
    minSize=(30, 30)  # Reduced minimum size
)

# 6. Process and store each face
for i, (x, y, w, h) in enumerate(faces):
    # Crop the face region
    cropped_face = img[y:y+h, x:x+w]
    
    # Save the face
    output_path = os.path.join(output_dir, f"face_{i}.jpg")
    cv2.imwrite(output_path, cropped_face)
    print(f"Saved face to {output_path}")

# 7. Show detection results (optional)
if len(faces) > 0:
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow("Detected Faces", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No faces detected in the image.")