import os
import numpy as np
from PIL import Image
from transformers import pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Directories for dataset and output
image_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\input"
output_base_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid"
labels_dir = r"C:\Users\Lenovo\Dropbox\PC\Desktop\camvid dataset\CamVid\camvid\input_label"
os.makedirs(output_base_dir, exist_ok=True)

# List of pretrained Segformer models
models = [
    {"name": "nvidia/segformer-b1-finetuned-cityscapes-1024-1024", "output_subdir": "segformer_b1"},
    {"name": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024", "output_subdir": "segformer_b2"},
    {"name": "nvidia/segformer-b3-finetuned-cityscapes-1024-1024", "output_subdir": "segformer_b3"},
    {"name": "nvidia/segformer-b4-finetuned-cityscapes-1024-1024", "output_subdir": "segformer_b4"},
    {"name": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024", "output_subdir": "segformer_b5"},
]

# Function to calculate IoU
def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0

# Function to calculate evaluation metrics
def calculate_metrics(pred, true):
    pred_flat = pred.flatten()
    true_flat = true.flatten()
    tn, fp, fn, tp = confusion_matrix(true_flat, pred_flat, labels=[0, 1]).ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) != 0 else 0
    
    return accuracy, sensitivity, specificity, f1_score

# Table for results
evaluation_results = []

# Generate masks and evaluate each model
for model_info in models:
    model_name = model_info["name"]
    output_subdir = os.path.join(output_base_dir, model_info["output_subdir"])
    os.makedirs(output_subdir, exist_ok=True)

    print(f"\nProcessing images with model: {model_name}")
    pipe = pipeline("image-segmentation", model=model_name)
    
    iou_scores = []
    accuracies = []
    sensitivities = []
    specificities = []
    f1_scores = []

    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)

        if not (image_name.endswith(".png") or image_name.endswith(".jpg") or image_name.endswith(".jpeg")):
            continue

        # Load image and process with the model
        image = Image.open(image_path)
        result = pipe(image)
        pred_mask = np.array(result[0]['mask'])

        # Load ground truth label
        label_path = os.path.join(labels_dir, image_name)
        if not os.path.exists(label_path):
            print(f"Ground truth not found for {image_name}. Skipping.")
            continue
        true_mask = np.array(Image.open(label_path).convert("L"))

        # Normalize masks for compatibility
        pred_mask = (pred_mask > 0).astype(np.uint8)
        true_mask = (true_mask > 0).astype(np.uint8)

        # Resize masks to match dimensions
        if pred_mask.shape != true_mask.shape:
            pred_mask = np.array(Image.fromarray(pred_mask).resize(true_mask.shape[::-1], Image.NEAREST))

        # Calculate metrics
        iou = calculate_iou(pred_mask, true_mask)
        accuracy, sensitivity, specificity, f1_score = calculate_metrics(pred_mask, true_mask)

        # Append results
        iou_scores.append(iou)
        accuracies.append(accuracy)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        f1_scores.append(f1_score)

    # Aggregate results for the current model
    mean_iou = np.mean(iou_scores) if iou_scores else 0
    mean_accuracy = np.mean(accuracies) if accuracies else 0
    mean_sensitivity = np.mean(sensitivities) if sensitivities else 0
    mean_specificity = np.mean(specificities) if specificities else 0
    mean_f1_score = np.mean(f1_scores) if f1_scores else 0

    # Store results in table
    evaluation_results.append({
        "Model": model_name,
        "Mean IoU": round(mean_iou, 4),
        "Accuracy": round(mean_accuracy, 4),
        "Sensitivity": round(mean_sensitivity, 4),
        "Specificity": round(mean_specificity, 4),
        "F1 Score": round(mean_f1_score, 4)
    })

# Print the evaluation results as a table
print("\nPerformance Evaluation Table:")
print("{:<25} {:<10} {:<10} {:<12} {:<12} {:<10}".format("Model", "Mean IoU", "Accuracy", "Sensitivity", "Specificity", "F1 Score"))
for result in evaluation_results:
    print("{:<25} {:<10} {:<10} {:<12} {:<12} {:<10}".format(
        result["Model"], result["Mean IoU"], result["Accuracy"], 
        result["Sensitivity"], result["Specificity"], result["F1 Score"]
    ))
