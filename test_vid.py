import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np

# Load Faster R-CNN with ResNet-50 backbone
def get_model(num_classes):
    # Load pre-trained Faster R-CNN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# Initialize the model
num_classes = 6  # Background + chair + person + table

# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# Load the trained model
model = get_model(num_classes)
model.load_state_dict(torch.load("Models\\fasterrcnn_resnet50_epoch_50.pth"))
model.to(device)
model.eval()  # Set the model to evaluation mode


def prepare_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Open image
    image_tensor = F.to_tensor(image).unsqueeze(0)  # Convert image to tensor and add batch dimension
    return image_tensor.to(device)




video_path = "test_vid.mp4"
cap = cv2.VideoCapture(video_path)



# # `prediction` contains:
# # - boxes: predicted bounding boxes
# # - labels: predicted class labels
# # - scores: predicted scores for each box (confidence level)
COCO_CLASSES = {0: "car", 1: "Bus", 2: "Car", 3: "Motorcycle",4:"Pickup",5:"Truck"}

def get_class_name(class_id):
    return COCO_CLASSES.get(class_id, "Unknown")



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
    image_tensor = F.to_tensor(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(image_tensor)

    # Draw boxes
    boxes = prediction[0]['boxes'].detach().cpu().numpy()
    labels = prediction[0]['labels'].detach().cpu().numpy()
    scores = prediction[0]['scores'].detach().cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:
            x1, y1, x2, y2 = map(int, box)
            class_name = get_class_name(label)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f'{class_name} {score:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    cv2.imshow("Faster R-CNN Video Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




