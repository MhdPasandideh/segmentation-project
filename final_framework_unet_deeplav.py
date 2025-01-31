import os
import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp
from torchvision import transforms, models
import matplotlib.pyplot as plt
import matplotlib

# Input and ground truth folders (replace with your actual paths)
input_folder = r"C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\input"
gt_folder = r"C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\input_label"

# Check if folders exist
if not os.path.exists(input_folder):
    raise FileNotFoundError(f"Input folder not found: {input_folder}")
if not os.path.exists(gt_folder):
    raise FileNotFoundError(f"Ground truth folder not found: {gt_folder}")

# Set device: CPU if CUDA isn't available or if CUDA issues arise
device = torch.device("cpu")  # Forces usage of CPU even if CUDA is available

# Load a pre-trained MobileNetV3 model from torchvision.models
model = models.mobilenet_v3_large(weights='IMAGENET1K_V1')  # Use 'weights' argument

# Replace the classification head with a segmentation head
num_classes = 3  # Adjust based on your dataset
model.classifier[1] = torch.nn.Conv2d(1280, num_classes, kernel_size=1) 

model.to(device)
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ResNet
])

def load_image(image_path):
    """Loads an image and applies transformations."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.imread(image_path)  # Read image in BGR format
    if image is None:
        raise ValueError(f"Unable to read the image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return image

def apply_segmentation(image, model, transform):
    """Applies the segmentation model to the image."""
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        output = model(image)  # Pass the image through the entire model
        output = output[0].argmax(0).cpu().numpy()  # Get the class prediction
    return output

def apply_color_map(mask, num_classes=3):
    """Applies a color map to the segmentation mask."""
    # Define a custom colormap (adjust as needed)
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])  # Example: Red, Green, Blue
    colored_mask = colors[mask]
    return colored_mask

def segment_and_save(input_folder, gt_folder, output_folder="segmented_output"):
    """Segments images, compares with ground truth (optional), and saves results."""
    input_filenames = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])
    gt_filenames = sorted([f for f in os.listdir(gt_folder) if f.endswith('.png')])

    if len(input_filenames) == 0 or len(gt_filenames) == 0:
        raise ValueError("No images found in input or ground truth folders.")

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for input_file, gt_file in zip(input_filenames, gt_filenames):
        input_image_path = os.path.join(input_folder, input_file)
        gt_image_path = os.path.join(gt_folder, gt_file)

        input_image = load_image(input_image_path)
        gt_image = load_image(gt_image_path) 

        pred_mask = apply_segmentation(input_image, model, transform)
        colored_pred_mask = apply_color_map(pred_mask)

        # Save the color-mapped prediction
        output_path = os.path.join(output_folder, f"seg_{input_file}")
        cv2.imwrite(output_path, colored_pred_mask)

        # Optionally, show the color-mapped mask
        plt.imshow(colored_pred_mask)
        plt.title(f"Segmentation result: {input_file}")
        plt.show()

# Run segmentation
try:
    segment_and_save(input_folder, gt_folder)
except Exception as e:
    print(f"Error during processing: {e}")