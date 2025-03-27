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
import json
from src.Heart_Segmentation.config.configuration import ModelEvaluationConfig
from src.Heart_Segmentation.utils.common import save_json


# 2. Dataset (Reusing HeartSegmentationDataset from training)
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
    


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the trained U-Net model
        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2
        ).to(self.device)
        self.model.load_state_dict(torch.load(self.config.model_save_path))
        self.model.eval()
        
        # DataLoader for evaluation
        dataset = HeartSegmentationDataset(self.config.images_dir, self.config.labels_dir)
        self.dataloader = DataLoader(dataset, batch_size=self.config.params_batch_size, shuffle=False, pin_memory=True)

    def calculate_dice_score(self, pred: torch.Tensor, target: torch.Tensor, smooth=1e-5):
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        intersection = (pred_flat * target_flat).sum()
        return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

    def calculate_iou(self, pred: torch.Tensor, target: torch.Tensor, smooth=1e-5):
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        return (intersection + smooth) / (union + smooth)

    def save_metrics_to_json(self, metrics: dict):
        """Save evaluation metrics to a JSON file."""
        json_path = self.config.model_metrics_json
        try:
            with open(json_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Metrics saved to {json_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics to {json_path}: {str(e)}")
            raise e

    def evaluate(self):
        """Run evaluation, compute metrics, and save them to JSON."""
        logger.info("-------------Started Evaluation----------")
        dice_scores = []
        iou_scores = []
        accuracies = []
        confidences = []

        with torch.no_grad():
            for images, labels, _ in self.dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                confidence = torch.max(probs, dim=1)[0].mean().cpu().item()
                confidences.append(confidence)

                for pred, label in zip(preds, labels):
                    pred_binary = (pred > 0).float()
                    label_binary = (label > 0).float()
                    
                    dice = self.calculate_dice_score(pred_binary, label_binary)
                    iou = self.calculate_iou(pred_binary, label_binary)
                    accuracy = (pred == label).float().mean().cpu().item()
                    
                    dice_scores.append(dice.cpu().item())
                    iou_scores.append(iou.cpu().item())
                    accuracies.append(accuracy)

        # Average metrics
        avg_dice = np.mean(dice_scores)
        avg_iou = np.mean(iou_scores)
        avg_accuracy = np.mean(accuracies) * 100
        avg_confidence = np.mean(confidences) * 100

        # Log results
        logger.info(f"Evaluation Results:")
        logger.info(f"Average Dice Score: {avg_dice:.4f}")
        logger.info(f"Average IoU Score: {avg_iou:.4f}")
        logger.info(f"Average Accuracy: {avg_accuracy:.2f}%")
        logger.info(f"Average Confidence Score: {avg_confidence:.2f}%")

       # Save metrics to JSON using the imported save_json function
        metrics = {
            "average_dice_score": avg_dice,
            "average_iou_score": avg_iou,
            "average_accuracy_percent": avg_accuracy,
            "average_confidence_score_percent": avg_confidence
        }
        save_json(path=self.config.model_metrics_json, data=metrics)

        return avg_dice, avg_iou, avg_accuracy, avg_confidence