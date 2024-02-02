import torch
import numpy as np
import cv2
import os
from facenet_pytorch import MTCNN

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def capture_faces(cam, detector, face_id, image_directory, image_count=500, max_images_per_face=3):
    count = 0
    last_face_centroid = None
    face_threshold = 50   
    
    print("\n[INFO] Initializing face capture. Look at the camera and wait...")

    while True:
        ret, img = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes, _ = detector.detect(rgb_img)

        if boxes is not None:
            faces_centroids = [(int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)) for box in boxes]

            if last_face_centroid is None:
                last_face_centroid = faces_centroids[0]  

            distances = [np.linalg.norm(np.array(last_face_centroid) - np.array(centroid)) for centroid in faces_centroids]
            closest_face_index = np.argmin(distances)

            if distances[closest_face_index] < face_threshold:

                box = boxes[closest_face_index]
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                face = rgb_img[y1:y2, x1:x2]
                face_image_path = f"{image_directory}/{face_id}_{count}.jpg"
                cv2.imwrite(face_image_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, f'{count}', (img.shape[1] - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('image', img)
                last_face_centroid = faces_centroids[closest_face_index] 

        count += 1

        k = cv2.waitKey(100) % 0xff
        if k == 27 or count >= image_count: 
            break

    print("\n[INFO] Success! Exiting program and closing windows.")
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    create_directory('images')
    detector = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # Setare latime
    cam.set(4, 480)  # Setare inaltime
    face_id = input("\nEnter user name and press <return>: ")
    image_directory = './images'
    capture_faces(cam, detector, face_id, image_directory)
