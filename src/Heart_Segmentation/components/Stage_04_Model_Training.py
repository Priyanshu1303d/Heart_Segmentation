import torch.nn as nn
from monai.networks.nets import UNet
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader , Dataset
from src.Heart_Segmentation import logger
import os
import nibabel as nib
import torch
from src.Heart_Segmentation.config.configuration import ModelTrainingConfig


class HeartSegmentationDataset(Dataset):
    def __init__(self, images_dir : str , labels_dir : str):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.nii.gz')]
        logger.info(f"Found {len(self.image_files)} files in {images_dir}")
        if not self.image_files:
            raise ValueError(f"No .nii.gz files found in {images_dir}")

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_file)
        label_path = os.path.join(self.labels_dir, image_file)
        
        image = nib.load(image_path).get_fdata()
        label = nib.load(label_path).get_fdata()
        
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # [1, 128, 128, 64]
        label = torch.tensor(label, dtype=torch.long)  # [128, 128, 64]
        
        return image, label, image_file



class ModelTraining:
    def __init__(self , config : ModelTrainingConfig ):
        super().__init__()

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 3D U-Net from MONAI (not pretrained, simpler than SMP's ResNet)
        self.model = UNet(
            spatial_dims=3,  # 3D data
            in_channels=1,   # Grayscale
            out_channels=3,  # 3 classes (adjust if needed)
            channels=(16, 32, 64, 128, 256),  # Feature maps per layer
            strides=(2, 2, 2, 2),  # Downsampling steps
            num_res_units=2  # Residual units
        ).to(self.device)

        #loss function and optimizer init
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters() , lr = self.config.params_learning_rate , weight_decay=self.config.params_weight_decay)

        #DataLoader object
        dataset = HeartSegmentationDataset(self.config.images_dir, self.config.labels_dir)
        self.dataloader = DataLoader(dataset , batch_size=self.config.params_batch_size, shuffle=False , pin_memory= True)


    def train(self):
        logger.info(f"-------------Started Training----------")
        self.model.train()
        running_loss = 0.0  # Initialized here
        
        for epoch in range(self.config.params_epochs):
            for images, labels, _ in self.dataloader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                y_pred = self.model(images)  # [batch_size, classes, 128, 128, 64]
                loss = self.loss_function(y_pred, labels)

                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            avg_loss = running_loss / len(self.dataloader)
            logger.info(f"Epoch [{epoch+1}/{self.config.params_epochs}], Loss: {avg_loss:.4f}")
        
        # Save the model
        torch.save(self.model.state_dict(), self.config.model_save_path)
        logger.info(f"Model saved to {self.config.model_save_path}")

    def predict(self, image: torch.Tensor) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            image = image.unsqueeze(0).to(self.device)  # Add batch dimension
            output = self.model(image)
            pred = torch.argmax(output, dim=1).cpu().numpy()
        return pred[0]

    def visualize_predictions(self, num_samples: int = 3):
        dataset = HeartSegmentationDataset(self.config.images_dir, self.config.labels_dir)
        samples = [dataset[i] for i in range(min(num_samples, len(dataset)))]
        
        for image, label, filename in samples:
            pred = self.predict(image)
            
            slice_idx = image.shape[-1] // 2
            image_slice = image[0, :, :, slice_idx].numpy()
            label_slice = label[:, :, slice_idx].numpy()
            pred_slice = pred[:, :, slice_idx]
            
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(image_slice, cmap="gray")
            plt.title(f"Image: {filename}")
            plt.axis("off")
            plt.subplot(1, 3, 2)
            plt.imshow(image_slice, cmap="gray")
            plt.imshow(label_slice, cmap="jet", alpha=0.5)
            plt.title("Ground Truth")
            plt.axis("off")
            plt.subplot(1, 3, 3)
            plt.imshow(image_slice, cmap="gray")
            plt.imshow(pred_slice, cmap="jet", alpha=0.5)
            plt.title("Prediction")
            plt.axis("off")
            plt.tight_layout()
            plt.show()

    

