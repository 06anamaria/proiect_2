import cv2
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision import transforms
from PIL import Image
import numpy as np

from custom_dataset import CustomDataset

def main():

    print('Loading model')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset_directories = ['images']  # sau orice alte directoare doriți să le adăugați
    dataset = CustomDataset(directories=dataset_directories, transform=transform)

    # Creare model nou
    model = InceptionResnetV1(pretrained='vggface2', num_classes=len(dataset.labels), classify=True).eval()

    # Încărcare ponderi într-un nou dicționar
    checkpoint = torch.load('trained_models/face_recognition_model.pth')
    new_state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith('logits'):
            continue  # Ignoram stratul logits, deoarece are un număr diferit de clase
        new_state_dict[key] = value

    # Încărcare dicționar de stare în modelul nou
    model.load_state_dict(new_state_dict, strict=False)

    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print('Setting up video capture')
    detector = MTCNN(keep_all=True, device=device)
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # Width
    cam.set(4, 480)  # Height


    threshold = 0.3
    while True:
        ret, img = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes, _ = detector.detect(rgb_img)

        if(boxes is not None):
                for i, box in enumerate(boxes):

                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    face = rgb_img[y1:y2, x1:x2]
                    img_height, img_width = img.shape[:2]
                    x1 = max(0, min(x1, img_width))
                    y1 = max(0, min(y1, img_height))
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    if face.size == 0:
                        continue

                    # Convert ROI to PIL image and process
                    roi_pil = Image.fromarray(face)
                    roi_tensor = transform(roi_pil).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        result = model(roi_tensor).detach().cpu()
                    max_value = torch.max(result)
                    label = dataset.labels[torch.argmax(result)]

                    """ if max_value > threshold:
                        cv2.putText(img, f'{label} ({max_value:.2f})', (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    else:
                        cv2.putText(img, 'unknown', (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2) """
                        
                    
                    if max_value < threshold:
                        cv2.putText(img, f'{label} ({max_value:.2f})', (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    else:
                        cv2.putText(img, 'unknown', (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                    """ if label is not None:
                        cv2.putText(img, f'{label} ({max_value:.2f})', (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    else:
                        cv2.putText(img, 'unknown', (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2) """



                    normalized_embedding = (result - result.min()) / (result.max() - result.min())
                    embedding_bar = np.tile(normalized_embedding, (10, 1))

                    cv2.imshow(f'visualization_{i}', embedding_bar)

        cv2.imshow('camera', img)

        k = cv2.waitKey(10) & 0xff
        if k == 27:  # 'ESC' key
            break

    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
