import cv2
import matplotlib.pyplot as plt

def draw_explainable_overlay(img_path, keypoints, atc_scores):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    kps = keypoints.astype(int)

    skeleton_pairs = [(0,2),(2,1),(0,4),(3,1),(1,7)]
    for (p1, p2) in skeleton_pairs:
        cv2.line(img, tuple(kps[p1]), tuple(kps[p2]), (0,255,0), 2)

    for i, (x, y) in enumerate(kps):
        cv2.circle(img, (x,y), 5, (255,0,0), -1)
        cv2.putText(img, str(i), (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0),1)

    cv2.putText(img, f"Body Length: {atc_scores['Body Length']}", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,0,0),2)
    cv2.putText(img, f"Height: {atc_scores['Height']}", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,0,0),2)
    cv2.putText(img, f"Chest Width: {atc_scores['Chest Width']}", (30,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,0,0),2)
    cv2.putText(img, f"Rump Angle: {atc_scores['Rump Angle']}", (30,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,0,0),2)

    plt.figure(figsize=(10,8))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Explainable AI: ATC Trait Overlay")
    plt.show()
