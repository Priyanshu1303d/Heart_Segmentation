
from monai.networks.nets import UNet
from torch.utils.data import Dataset , DataLoader
from Heart_Segmentation import logger
import nibabel as nib
import torch
import torch.nn as nn
import os
from Heart_Segmentation.constants import *
from src.Heart_Segmentation.config.configuration import ModelOptimizationConfig
from src.Heart_Segmentation.config.configuration import ConfigurationManager



class HeartSegmentationDataset(Dataset):
    def __init__(self, images_dir: str, labels_dir: str):
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
    



class ModelOptimization:
    def __init__(self, config: ModelOptimizationConfig):
        self.config = config
        #### self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # DataLoader initialization
        train_dataset = HeartSegmentationDataset(self.config.images_dir, self.config.labels_dir)
        self.train_loader = DataLoader(train_dataset, batch_size=self.config.params_batch_size, shuffle=True, pin_memory=True)
        self.test_loader = DataLoader(train_dataset, batch_size=self.config.params_batch_size, shuffle=False, pin_memory=True)  # Assuming same dataset for simplicity

        # Model initialization (assuming a custom Model_Optimization class exists)
        # Load the trained U-Net model
        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2
        )  #####.to(self.device)
        self.model.load_state_dict(torch.load(self.config.model_save_path))
        self.model.eval()

        # Loss and optimizer
        self.loss_function = nn.CrossEntropyLoss()
        if self.config.params_optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.params_learning_rate, weight_decay=self.config.params_weight_decay)
        elif self.config.params_optimizer_name == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.params_learning_rate, weight_decay=self.config.params_weight_decay)
        else:
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.config.params_learning_rate, weight_decay=self.config.params_weight_decay)

    def train(self):
        for epoch in range(self.config.params_epochs):
            self.model.train()
            for batch_features, batch_labels, _ in self.train_loader:
                #### batch_features, batch_labels = batch_features.to(self.device), batch_labels.to(self.device)
                y_pred = self.model(batch_features)
                loss = self.loss_function(y_pred, batch_labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def evaluate(self):
        self.model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for batch_features, batch_labels, _ in self.test_loader:
               ####  batch_features, batch_labels = batch_features.to(self.device), batch_labels.to(self.device)
                y_pred = self.model(batch_features)
                _, predicted = torch.max(y_pred, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        accuracy = correct / total
        return accuracy

    def save_best_model(self, accuracy):
        if not os.path.exists(self.config.best_model_path) or accuracy > self.best_accuracy:
            torch.save(self.model.state_dict(), self.config.best_model_path)
            self.best_accuracy = accuracy
            logger.info(f"Saved best model with accuracy {accuracy:.4f} to {self.config.best_model_path}")


def objective(trial):
    config = ConfigurationManager()
    config = config.get_model_optimization_config(trial)
    
    model_opt = ModelOptimization(config)
    model_opt.train()
    accuracy = model_opt.evaluate()
    
    # Save the best model
    model_opt.save_best_model(accuracy)
    
    return accuracy