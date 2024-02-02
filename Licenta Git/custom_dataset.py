import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, directories, transform=None):
        self.directories = directories
        self.transform = transform
        self.images = self._load_images()  
        self.labels = self._extract_labels()

        unique = sorted(set(self.labels))
        print(f'Found {len(unique)} persons in the dataset:\n {str(unique)}')
        self.images_by_person = {label: [] for label in unique}
        for img, label in zip(self.images, self.labels):
            self.images_by_person[label].append(img)

        self.triplets = self._create_triplets()

    def get_label(self, img_path):
        label = os.path.basename(img_path).split('_')[0]
        return label

    @property
    def anchors(self):
        return [triplet[0] for triplet in self.triplets]

    @property
    def positives(self):
        return [triplet[1] for triplet in self.triplets]

    @property
    def negatives(self):
        return [triplet[2] for triplet in self.triplets]

    def compute_embeddings(self, model, device):
        model.eval()  
        embeddings = []
        labels = []

        with torch.no_grad():
            for img_path in self.images:
                img = self._load_image(img_path)
                img = img.to(device)
                embedding = model(img.unsqueeze(0)).squeeze(0) 
                embeddings.append(embedding.cpu())
                labels.append(self.get_label(img_path))

        return torch.stack(embeddings), labels

    def get_hard_negatives(self, embeddings, labels):
        distance_matrix = torch.cdist(embeddings, embeddings, p=2)
        
        # For each anchor-positive pair, find the hard negative
        hard_negatives = []
        for anchor_idx, anchor_label in enumerate(labels):
            # Mask to select only negatives for this anchor
            mask = torch.tensor([label != anchor_label for label in labels])
            # Get the hardest negative for this anchor
            distances_to_negatives = distance_matrix[anchor_idx][mask]
            if len(distances_to_negatives) > 0:
                hard_negative_idx = distances_to_negatives.argmin().item()
                hard_negatives.append(hard_negative_idx)
            else:
                hard_negatives.append(None)
        return hard_negatives

    def update_triplets_with_hard_negatives(self, hard_negatives):
            # Update the dataset's triplets with the identified hard negatives
            for idx, hard_negative_idx in enumerate(hard_negatives):
                if hard_negative_idx is not None and idx < len(self.triplets):
                    anchor, positive, _ = self.triplets[idx]
                    hard_negative_image = self.images[hard_negative_idx]
                    self.triplets[idx] = (anchor, positive, hard_negative_image)



    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_path, positive_path, negative_path = self.triplets[idx]

        anchor_img = self._load_image(anchor_path)
        positive_img = self._load_image(positive_path)
        negative_img = self._load_image(negative_path)

        return anchor_img, positive_img, negative_img

    def _load_image(self, img_path):
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

    def _load_images(self):
        images = []
        for directory in self.directories:
            images.extend(glob.glob(os.path.join(directory, '*.jpg')))
        return images

    def _extract_labels(self):
        labels = [os.path.basename(img).split('_')[0] for img in self.images]
        return labels

    def _split_images(self, images):
        np.random.shuffle(images)
        midpoint = len(images) // 2
        return zip(images[:midpoint], images[midpoint:])

    def _get_random_label(self, exclude=None):
        labels = [label for label in self.labels if label != exclude]
        if not labels:
            return exclude
        else:
            return np.random.choice(labels)

    def _create_triplets(self):
        triplets = []
        for label, images in self.images_by_person.items():
            if len(images) < 2:
                continue
            anchors_positives = self._split_images(images)
            for anchor, positive in anchors_positives:
                negative_label = self._get_random_label(exclude=label)
                negative_image = np.random.choice(self.images_by_person[negative_label])
                triplets.append((anchor, positive, negative_image))
        return triplets
