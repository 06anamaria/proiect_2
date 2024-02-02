import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
import matplotlib.pyplot as plt

from custom_dataset import CustomDataset

def plot_and_log(losses, learning_rate, num_epochs, batch_size=None, momentum=None, timestamp=None):

    train_losses, val_losses = zip(*losses)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o', color='b', label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, marker='o', color='r', label='Validation Loss')

    title = f'Loss per Epoch\nLearning Rate: {learning_rate}'
    if batch_size is not None:
        title += f', Batch Size: {batch_size}'
    if momentum is not None:
        title += f', Momentum: {momentum}'
    if time is not None:
        title += f', Time: {timestamp}'    

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_dir_path = 'trained_models/plots/'
    log_dir_path = 'trained_models/logs/'
    if not os.path.exists(plot_dir_path):
        os.makedirs(plot_dir_path)
    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)

    existing_plots = os.listdir(plot_dir_path)
    existing_logs = os.listdir(log_dir_path)
    file_id = 1
    while f'loss_plot_{file_id}.png' in existing_plots or f'training_log_{file_id}.txt' in existing_logs:
        file_id += 1
    plot_file_path = os.path.join(plot_dir_path, f'loss_plot_{file_id}.png')
    log_file_path = os.path.join(log_dir_path, f'training_log_{file_id}.txt')

    plt.savefig(plot_file_path)
    plt.close()  

    with open(log_file_path, 'w') as log_file:
        
        config_info = f"Start training: Epochs: {num_epochs}, LR: {learning_rate}, Batch Size: {batch_size}, Momentum: {momentum}, Time: {timestamp}\n"
        log_file.write(config_info)
        
        
        for epoch, (train_loss, val_loss) in enumerate(losses, 1):
            epoch_info = f'Epoch {epoch}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}\n'
            log_file.write(epoch_info)

def main():

    model_save_path = 'trained_models'
    os.makedirs(model_save_path, exist_ok=True)
    

    train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),  
    transforms.RandomRotation(15), 
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
    transforms.RandomResizedCrop(256), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

    batch_size = 8

    dataset_directories = ['images', 'images_2', 'random']
    dataset = CustomDataset(directories=dataset_directories, transform=train_transforms)

    train_size = int(0.8 * len(dataset))  # 80% for training
    validation_size = len(dataset) - train_size  # 20% for validation
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    validation_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = InceptionResnetV1(pretrained='vggface2', num_classes=len(dataset.labels), classify=True).train()

    device = torch.device('cuda' if torch.cuda.device_count() else 'cpu')
    model.to(device)

    learning_rate = 0.005
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    num_epochs = 200  
    losses = []
    
    HARD_NEGATIVE_MINING_INTERVAL = 5  # Interval for hard negative mining


    start_time = time.time()

    for epoch in range(num_epochs):
        # Hard negative mining
        if epoch % HARD_NEGATIVE_MINING_INTERVAL == 0:
            print(f"Epoch {epoch}: Computing embeddings for hard negative mining.")
            embeddings, labels = dataset.compute_embeddings(model, device)
            hard_negatives = dataset.get_hard_negatives(embeddings, labels)
            dataset.update_triplets_with_hard_negatives(hard_negatives) 
        

        model.train()
        train_running_loss = 0.0
        for anchor_imgs, positive_imgs, negative_imgs in train_loader:
            anchor_imgs = anchor_imgs.to(device)
            positive_imgs = positive_imgs.to(device)
            negative_imgs = negative_imgs.to(device)
             
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            anchor_embeddings = model(anchor_imgs)
            positive_embeddings = model(positive_imgs)
            negative_embeddings = model(negative_imgs)

            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_running_loss += loss.item()

        train_epoch_loss = train_running_loss / len(train_loader.dataset)

        with torch.no_grad():
            model.eval()
            validation_running_loss = 0.0

            for anchor_imgs, positive_imgs, negative_imgs in validation_loader:
                anchor_imgs = anchor_imgs.to(device)
                positive_imgs = positive_imgs.to(device)
                negative_imgs = negative_imgs.to(device)
                
                anchor_embeddings = model(anchor_imgs)
                positive_embeddings = model(positive_imgs)
                negative_embeddings = model(negative_imgs)
                
                loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
                validation_running_loss += loss.item()
            
            validation_epoch_loss = validation_running_loss / len(validation_loader.dataset)

        # Log and plot the epoch results
        losses.append((train_epoch_loss, validation_epoch_loss))  
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_epoch_loss:.6f}, Validation Loss: {validation_epoch_loss:.6f}')
 
    end_time = time.time()
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    timestamp = f'{int(hours)}:{int(minutes):02d}:{seconds:.2f}'
    print(f'Finished training, took {timestamp}')

    plot_and_log(losses, learning_rate, num_epochs, batch_size, None, timestamp)


    torch.save(model.state_dict(), os.path.join(model_save_path, 'face_recognition_model.pth'))
    
    print('Training complete')

if __name__ == "__main__":
    main()
