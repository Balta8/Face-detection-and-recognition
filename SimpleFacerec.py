import face_recognition
import cv2
import os
import glob
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25  # Resize frame for faster processing

    def load_encoding_images(self, images_path):
        """
        Load images from the path and encode faces
        :param images_path: Path to the folder with images
        """
        # Collect image paths
        images_path = glob.glob(os.path.join(images_path, "*.*"))
        print(f"{len(images_path)} encoding images found.")

        # Process each image
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Extract filename (without extension) as the name label
            filename = os.path.splitext(os.path.basename(img_path))[0]

            # Encode the face
            try:
                img_encoding = face_recognition.face_encodings(rgb_img)[0]
                self.known_face_encodings.append(img_encoding)
                self.known_face_names.append(filename)
                print(f"Encoded {filename}")
            except IndexError:
                print(f"Face not detected in {filename}, skipping.")
                
        print("Encoding images loaded.")

    def detect_known_faces(self, frame):
        """
        Detect and recognize known faces in the given frame
        :param frame: The video frame to process
        :return: Face locations and their corresponding names
        """
        # Resize frame to improve processing speed
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Compare face encodings with known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # Find the best match
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                
            face_names.append(name)
        
        # Scale back face locations to match the original frame size
        face_locations = (np.array(face_locations) / self.frame_resizing).astype(int)
        
        return face_locations, face_names

   
    
        





