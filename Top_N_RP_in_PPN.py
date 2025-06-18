import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load custom-trained model with 6 classes (background + 5)
def get_finetuned_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=False)  # Do NOT use pretrained weights
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

num_classes = 6  # Based on your training setup
model = get_finetuned_model(num_classes)

# Load checkpoint
model.load_state_dict(torch.load("Models\\fasterrcnn_resnet50_epoch_50.pth", map_location=device))
model.to(device)
model.eval()

# Load image and transform
image_path = 'test_img.jpg'  # Replace with your image path
image = Image.open(image_path).convert('RGB')
transform = transforms.Compose([transforms.ToTensor()])
img_tensor = transform(image).unsqueeze(0).to(device)

# Extract proposals from RPN
with torch.no_grad():
    images, _ = model.transform(img_tensor)
    features = model.backbone(images.tensors)
    proposals, _ = model.rpn(images, features)
N = 200
# Visualize top 100 RPN proposals
rpn_boxes = proposals[0][:N].cpu()  # Move to CPU for plotting
img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img_np)
for box in rpn_boxes:
    x1, y1, x2, y2 = box
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             linewidth=1.5, edgecolor='lime', facecolor='none')
    ax.add_patch(rect)

plt.title(f"Top {N} RPN Proposals from Fine-Tuned Faster R-CNN")
plt.axis('off')
plt.show()
