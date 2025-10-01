import cv2
from ultralytics import YOLO

breed_model = YOLO("models/breed_model.pt")
keypoint_model = YOLO("models/keypoint_model.pt")

def detect_breed(img_path):
    results = breed_model(img_path)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            breed_name = r.names[cls_id]
            confidence = float(box.conf[0])
            return breed_name, confidence
    return None, None

def detect_keypoints(img_path):
    results = keypoint_model(img_path)
    keypoints = None
    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.xy[0].cpu().numpy()
    return keypoints
