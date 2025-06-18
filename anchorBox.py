import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Original image size
image_height, image_width = 800, 800

# Assume feature map stride (e.g., FPN stride=16)
stride = 16

# Select one location in feature map, say (25, 25)
center_x = 25 * stride
center_y = 25 * stride

# Anchors defined by 3 scales and 3 aspect ratios
scales = [128, 256, 512]
ratios = [0.5, 1.0, 2.0]

def generate_anchors(center_x, center_y, scales, ratios):
    anchors = []
    for scale in scales:
        for ratio in ratios:
            h = scale * (ratio ** 0.5)
            w = scale / (ratio ** 0.5)
            x1 = center_x - w / 2
            y1 = center_y - h / 2
            x2 = center_x + w / 2
            y2 = center_y + h / 2
            anchors.append([x1, y1, x2, y2])
    return anchors

# Generate
anchors = generate_anchors(center_x, center_y, scales, ratios)

# Plot anchors
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, image_width)
ax.set_ylim(0, image_height)
ax.invert_yaxis()

for box in anchors:
    rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                             linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

ax.plot(center_x, center_y, 'bo')  # center point
plt.title("Anchor Boxes at a Single Location")
plt.show()

# Function to compute IoU
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area

# Define one ground truth box (x1, y1, x2, y2)
gt_box = [300, 300, 500, 500]  # A ground truth box

# Anchor settings (centered at some feature map location)
center_x, center_y = 400, 400
scales = [128, 256, 512]
ratios = [0.5, 1.0, 2.0]

# Generate anchors at one point
def generate_anchors(cx, cy, scales, ratios):
    anchors = []
    for scale in scales:
        for ratio in ratios:
            h = scale * (ratio ** 0.5)
            w = scale / (ratio ** 0.5)
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            anchors.append([x1, y1, x2, y2])
    return anchors

anchors = generate_anchors(center_x, center_y, scales, ratios)

# Compute IoUs and assign colors
anchor_colors = []
for anchor in anchors:
    iou = compute_iou(anchor, gt_box)
    if iou >= 0.7:
        anchor_colors.append('red')  # Positive
    elif iou <= 0.3:
        anchor_colors.append('blue')  # Negative
    else:
        anchor_colors.append('orange')  # Neutral

# Plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, 800)
ax.set_ylim(0, 800)
ax.invert_yaxis()
ax.set_title("IoU-based Anchor Labeling")

# Plot Ground Truth box
gt_rect = patches.Rectangle((gt_box[0], gt_box[1]), gt_box[2] - gt_box[0], gt_box[3] - gt_box[1],
                            linewidth=3, edgecolor='green', facecolor='none', label='Ground Truth')
ax.add_patch(gt_rect)

# Plot anchors
for anchor, color in zip(anchors, anchor_colors):
    rect = patches.Rectangle((anchor[0], anchor[1]), anchor[2]-anchor[0], anchor[3]-anchor[1],
                             linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)

plt.legend(handles=[gt_rect], loc='upper right')
plt.show()