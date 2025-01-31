import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.metrics import jaccard_score

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_list = os.listdir(image_dir)
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_list[idx])
        mask_name = os.path.join(self.mask_dir, self.image_list[idx].replace('.jpg', '_mask.png'))  # Adjust mask file extension
        
        image = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')
        
        if self.transform:
            image = self.transform(image)
            # Note: Mask should not be normalized like images
            mask = torch.from_numpy(np.array(mask)).unsqueeze(0).float() / 255.0
        
        return image, mask

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    # Note: Depth transform would need actual implementation or integration with another dataset source
])

# Load dataset
image_dir = r'C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\input'  # Replace with actual path
mask_dir = r'C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\input_label'    # Replace with actual path
dataset = SegmentationDataset(image_dir, mask_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Helper function to calculate IOU
def calculate_iou(pred_mask, true_mask, num_classes=21):
    iou_list = []
    for i in range(num_classes):
        pred = (pred_mask == i).flatten()
        true = (true_mask == i).flatten()
        intersection = np.sum(np.logical_and(pred, true))
        union = np.sum(np.logical_or(pred, true))
        iou_list.append(intersection / union if union else 0)
    return np.mean(iou_list)

# Implementations for enhancements:

# 2. Self-Supervised Learning (simplified example for rotation prediction)
def self_supervised_pretask(image):
    # Simulate a rotation prediction task
    angles = [0, 90, 180, 270]
    rotated_images = [transforms.functional.rotate(image, angle) for angle in angles]
    # In real implementation, you would train with this as a pretext task
    return rotated_images

# 5. Domain Adaptation (pseudo code)
def adapt_to_new_domain(model, new_domain_data_loader):
    # Example: Fine-tune model on new domain data
    model.train()
    for batch in new_domain_data_loader:
        images, _ = batch
        outputs = model(images)
        # Loss computation and backpropagation would be here
    model.eval()

# 6. Interactive Segmentation (pseudo code)
def interactive_refinement(original_image, mask, user_feedback):
    # Placeholder for user interaction, e.g., clicking or drawing
    return mask  # For now, return the original mask

# 7. Interpretability (simplified Grad-CAM visualization)
def visualize_model_attention(image, model, target_layer):
    # Grad-CAM would require backpropagation, not feasible in eval mode without full setup
    pass  # This would need a complex setup for real implementation

# Function to save image
def save_image(image, path):
    plt.imsave(path, image)

# Function to plot accuracy
def plot_accuracy(baseline_acc, enhanced_acc, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(baseline_acc, label='Baseline Accuracy')
    plt.plot(enhanced_acc, label='Enhanced Accuracy')
    plt.xlabel('Image Index')
    plt.ylabel('Mean IoU')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

# Setup paths
output_dir = r'C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\final_thesis_10_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Lists to store accuracy for plotting
baseline_accuracies = []
enhanced_accuracies = []

# Models for comparison
baseline_model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
baseline_model.eval()

enhanced_model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
enhanced_model.eval()

# Inference loop
for idx, (image, mask) in enumerate(data_loader):
    image = image.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    mask = mask.squeeze(0).numpy()
    
    # Baseline model prediction
    with torch.no_grad():
        baseline_output = baseline_model(image)['out'][0]
    baseline_pred = torch.argmax(baseline_output, dim=0).cpu().numpy()
    baseline_iou = calculate_iou(baseline_pred, mask)
    baseline_accuracies.append(baseline_iou)
    
    # Enhanced model prediction with some enhancements
    # self_supervised_pretask(image)  # Not used in this example due to lack of training phase
    with torch.no_grad():
        enhanced_output = enhanced_model(image)['out'][0]
    enhanced_pred = torch.argmax(enhanced_output, dim=0).cpu().numpy()
    enhanced_iou = calculate_iou(enhanced_pred, mask)
    enhanced_accuracies.append(enhanced_iou)
    
    # Save predicted images
    save_image(baseline_pred, os.path.join(output_dir, f"baseline_image_{idx}.png"))
    save_image(enhanced_pred, os.path.join(output_dir, f"enhanced_image_{idx}.png"))

# Plot and save accuracy comparison
plot_accuracy(baseline_accuracies, enhanced_accuracies, os.path.join(output_dir, "accuracy_comparison.png"))

print("Segmentation results and accuracy plot saved successfully.")