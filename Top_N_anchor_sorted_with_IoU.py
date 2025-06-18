import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from torchvision import transforms
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ========== Config ==========
image_path = r'Dataset\No_Apply_Grayscale\No_Apply_Grayscale\Vehicles_Detection.v8i.coco\test\frame_7929_jpg.rf.02b89050254d7e4c8c8995125a555cfe.jpg'
coco_annotation_path = r'Dataset\No_Apply_Grayscale\No_Apply_Grayscale\Vehicles_Detection.v8i.coco\test\_annotations.coco.json'
image_id = 6  # Replace with the correct COCO image ID
model_weights_path = 'Models\\fasterrcnn_resnet50_epoch_50.pth'
num_classes = 6  # Background + 5 classes
top_n_proposals = 100
N = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_iou(boxA, boxesB):
    x1 = torch.max(boxA[0], boxesB[:, 0])
    y1 = torch.max(boxA[1], boxesB[:, 1])
    x2 = torch.min(boxA[2], boxesB[:, 2])
    y2 = torch.min(boxA[3], boxesB[:, 3])

    inter_area = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxesB[:, 2] - boxesB[:, 0]) * (boxesB[:, 3] - boxesB[:, 1])
    union = areaA + areaB - inter_area
    iou = inter_area / union
    return iou

# ========== Load Model ==========
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

model = get_model(num_classes)
model.load_state_dict(torch.load(model_weights_path, map_location=device))
model.to(device)
model.eval()

# ========== Load Image ==========
image = Image.open(image_path).convert('RGB')
transform = transforms.Compose([transforms.ToTensor()])
img_tensor = transform(image).unsqueeze(0).to(device)

# ========== Load Ground Truth Boxes ==========
coco = COCO(coco_annotation_path)
ann_ids = coco.getAnnIds(imgIds=image_id)
anns = coco.loadAnns(ann_ids)

# Convert COCO bbox (x, y, w, h) to (x1, y1, x2, y2)
gt_boxes = torch.tensor([
    [x, y, x + w, y + h] for ann in anns for x, y, w, h in [ann['bbox']]
], dtype=torch.float32).to(device)

# ========== Get RPN Proposals ==========
with torch.no_grad():
    images, _ = model.transform(img_tensor)
    features = model.backbone(images.tensors)
    proposals, _ = model.rpn(images, features)
    # Compute IoUs for all proposals
    all_proposals = proposals[0]
    ious_with_gt = torch.tensor([compute_iou(box, gt_boxes).max().item() for box in all_proposals])

    # Get indices of Top-10 proposals by IoU
    top_iou_indices = torch.topk(ious_with_gt, N).indices
    rpn_boxes = all_proposals[top_iou_indices]

    # Print for reference
    print(f"Top {N} IoUs:", ious_with_gt[top_iou_indices])


# ========== Compute IoU ==========
def compute_iou(boxA, boxesB):
    x1 = torch.max(boxA[0], boxesB[:, 0])
    y1 = torch.max(boxA[1], boxesB[:, 1])
    x2 = torch.min(boxA[2], boxesB[:, 2])
    y2 = torch.min(boxA[3], boxesB[:, 3])

    inter_area = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxesB[:, 2] - boxesB[:, 0]) * (boxesB[:, 3] - boxesB[:, 1])
    union = areaA + areaB - inter_area
    iou = inter_area / union
    return iou

# ========== Visualization ==========
img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(img_np)

# Plot anchor boxes (RPN proposals) with IoU coloring
for box in rpn_boxes:
    ious = compute_iou(box, gt_boxes)
    max_iou = ious.max().item()
    
    # Choose color based on IoU
    if max_iou >= 0.7:
        color = 'red'      # Positive anchor
    elif max_iou >= 0.3:
        color = 'orange'   # Medium IoU
    else:
        color = 'blue'     # Negative anchor

    x1, y1, x2, y2 = box.tolist()
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             linewidth=1.5, edgecolor=color, facecolor='none')
    ax.add_patch(rect)

# Plot ground truth boxes
for box in gt_boxes:
    x1, y1, x2, y2 = box.tolist()
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             linewidth=1.5, edgecolor='lime', facecolor='none')
    ax.add_patch(rect)

max_ious = [compute_iou(box, gt_boxes).max().item() for box in rpn_boxes]
print(f"Max IoU in top {len(rpn_boxes)} proposals: {max(max_ious):.3f}")


plt.title("Anchors Colored by IoU: Red (≥0.7), Orange (0.3–0.7), Blue (<0.3), GT: Green")
plt.axis('off')
plt.tight_layout()
plt.show()
