import numpy as np
import cv2
from picamera2 import Picamera2, Preview
import matplotlib.pyplot as plt
import os 
import torch
import cv2
import pathlib 
import time
temp = pathlib.PosixPath
pathlib.WindowsPath = pathlib.PosixPath
from models.common import DetectMultiBackend

print("Hello World: Finished Importing Modules")


def filter_predictions(predictions, confidence_threshold=0.25, iou_threshold=0.45):
    """
    Filters YOLOv5 model output to keep only boxes with a high enough confidence
    and applies Non-Maximum Suppression (NMS) to filter overlapping boxes.
    """
    # Apply a confidence threshold on the objectness score (5th column)
    # Only keep predictions with confidence > confidence_threshold
    filtered_predictions = predictions[predictions[:, 4] > confidence_threshold]
    # Use YOLOv5 NMS (PyTorch built-in for YOLO) with IoU filtering
    # The `torchvision.ops.nms` function can be used here, but YOLOv5 provides a helper function
    filtered_boxes = []

    for pred in filtered_predictions:
        # `torchvision.ops.nms` expects boxes in [x1, y1, x2, y2] format, confidences as scores, and iou_threshold
        boxes = pred[:, :4]
        scores = pred[:, 4]
        indices = torch.ops.torchvision.nms(boxes, scores, iou_threshold)

        # Collect NMS-filtered boxes
        filtered_boxes.append(pred[indices])

    return filtered_boxes

def calculate_iou(gt_box, pred_box):
    x1, y1, w1, h1 = gt_box
    x2, y2, w2, h2 = pred_box

    # Convert to corner coordinates
    gt_box_corners = (x1, y1, x1 + w1, y1 + h1)
    pred_box_corners = (x2, y2, x2 + w2, y2 + h2)

    # Calculate intersection coordinates
    inter_x1 = max(gt_box_corners[0], pred_box_corners[0])
    inter_y1 = max(gt_box_corners[1], pred_box_corners[1])
    inter_x2 = min(gt_box_corners[2], pred_box_corners[2])
    inter_y2 = min(gt_box_corners[3], pred_box_corners[3])

    # Calculate intersection area
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection_area = inter_width * inter_height

    # Calculate union area
    gt_area = w1 * h1
    pred_area = w2 * h2
    union_area = gt_area + pred_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

def non_max_suppression(pred_boxes, scores, iou_threshold=0.5):
    # Convert to numpy array for easier manipulation
    pred_boxes = np.array(pred_boxes)
    scores = np.array(scores)

    # Get indices of boxes sorted by scores
    indices = np.argsort(scores)[::-1]

    # List to hold selected boxes
    selected_boxes = []

    while len(indices) > 0:
        # Get the index of the box with the highest score
        current = indices[0]
        selected_boxes.append(pred_boxes[current])

        # Calculate IoU with the current box
        ious = []
        for index in indices[1:]:
            iou = calculate_iou(pred_boxes[current], pred_boxes[index])  # Use the IoU function from above
            ious.append(iou)

        # Select indices that have IoU less than the threshold
        indices = indices[np.where(np.array(ious) < iou_threshold)[0] + 1]

    return selected_boxes

def convert_bbox(center_bbox):
    # Unpack the input bounding box
    x_center, y_center, width, height = center_bbox
    
    # Calculate x1, y1, x2, y2
    x1 = x_center - (width / 2)
    y1 = y_center - (height / 2)
    x2 = x_center + (width / 2)
    y2 = y_center + (height / 2)
    
    return (int(x1), int(y1), int(x2), int(y2))


def get_pred(model, image_bgr, conf_thresh = 0.2, iou_thresh = 0.4):
    t0 = time.time()
    # Step 1: Resize the image to (640, 640)
    image_cv2 = cv2.resize(image_bgr, (640, 640))

    # Step 2: Normalize the image to [0, 1]
    image_normalized = image_cv2/ 255.0
    
    # # Step 3: Convert to PyTorch tensor and rearrange dimensions
    image_tensor = torch.tensor(image_normalized, dtype=torch.float32).permute(2, 0, 1)  # Shape: (3, 640, 640)

    # # Step 4: Add a batch dimension to the tensor (if necessary)
    img = image_tensor.unsqueeze(0)  # Shape: (1, 3, 640, 640)

    print(f"This part of the preprocessing took: {round(time.time() - t0, 3)} sec.")
    t1 = time.time()

    # # Step 5: Inference
    results, _ = model(img)
    
    print(f"Inference took {round(time.time() - t1, 3)} seconds")
    t2 = time.time()
    
    # # Step 6: Postprocessing
    res = results[0].numpy()
    
    # # Step 7: Only detections over the confidence threshold
    res_f = res[res[:, 4]>conf_thresh]
    
    # Get the boxes
    boxes = res_f[:, :4]
    # Get the class confidences
    conf = res_f[:, 4]
    
    # Non maximumal confidence boxes get removes
    boxe = non_max_suppression(boxes, conf, iou_threshold= iou_thresh)
    
    
    # # Step 8: Plot the boxes 
    for j, box in enumerate(boxe):
    # Draw the bounding box
        x1, y1, x2, y2 = convert_bbox(box)
        cv2.rectangle(image_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Display class name and confidence
        # label = f"{class_name}: {confidence:.2f}"
        #cv2.putText(image_cv2, "Test", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        
    print(f"Postprocessing took {round(time.time() - t2, 2)} seconds")

    return image_cv2



    

##########################################
# Start of the program
##########################################

# Initialise the classes
classes = {
  0: "Pot",
  1: "AllCrack",
  2: "LongCrack",
  3: "LatCrack"}

conf_thresh = 0.25 # Higher confidence threshold, means the model has to be more sure to make the detection.
iou_thresh = 0.2 # Lower threshold, means with lower overlap they will be filtered out. 


# Set paths
#weights_path = '/home/vincent/Documents/DetectPot/yolov5/best.pt'  # Path to your trained weights
weights_path = '/home/vincent/Documents/DetectPot/yolov5/SingleClassYolov5Weights.pt'  # Path to your trained weights

# save pictures
pic_path = '/home/vincent/Documents/DetectPot/Fotos_Detection'

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model
print("Loading model...")
model = DetectMultiBackend(weights_path)

# Initialize Picamera2
picam2 = Picamera2()

# Set resolution and framerate in the preview configuration
config = picam2.create_preview_configuration(main={"size": (640, 640), "format": "RGB888"})
picam2.configure(config)

# Start the camera
picam2.start()
print("Video stream started with resolution (640x640) and 32 FPS (if supported)")

# Capture frames continuously from the video stream
while True:
    
    t0 = time.time()
    
    # Capture the current frame as a NumPy array
    image_bgr = picam2.capture_array()

    
    frame = get_pred(model, image_bgr, conf_thresh = 0.2, iou_thresh = 0.4)
    
    # Display the frame in a window (optional)
    cv2.imshow("Video Stream", frame)
    print(f"Took {round(time.time() - t0, 3)} seconds")
    
    # Save the image
    timestamp = int(time.time())
    filename_og = os.path.join(pic_path, f"{timestamp}_captured.jpg")
    filename_pred = os.path.join(pic_path, f"{timestamp}_predicted.jpg")
    cv2.imwrite(filename_pred, frame)
    cv2.imwrite(filename_og, fimage_bgr)
    
    # Save the image on 's' key press
    if cv2.waitKey(1) & 0xFF == ord('s'):
        timestamp = int(time.time())
        filename = f"captured_image_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Image saved as {filename}")
    
    # Exit the stream on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the camera and close the window
picam2.stop()
cv2.destroyAllWindows()
print("Video stream stopped.")

