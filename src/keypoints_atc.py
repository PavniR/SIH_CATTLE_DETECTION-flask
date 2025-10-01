import numpy as np

def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)

def angle(p1, p2, p3):
    a = euclidean(p2, p3)
    b = euclidean(p1, p3)
    c = euclidean(p1, p2)
    return np.degrees(np.arccos((a**2 + c**2 - b**2) / (2*a*c + 1e-6)))

def extract_traits(keypoints):
    traits = {
        "Body Length": euclidean(keypoints[2], keypoints[1]),
        "Height": euclidean(keypoints[0], keypoints[4]),
        "Chest Width": euclidean(keypoints[2], keypoints[6]),
        "Rump Angle": angle(keypoints[3], keypoints[1], keypoints[7])
    }
    return traits

def score_trait(value, min_val, max_val):
    scaled = 1 + 8 * (value - min_val) / (max_val - min_val + 1e-6)
    return int(np.clip(round(scaled), 1, 9))

def calculate_atc_scores(traits):
    return {
        "Body Length": score_trait(traits["Body Length"], 150, 500),
        "Height": score_trait(traits["Height"], 100, 400),
        "Chest Width": score_trait(traits["Chest Width"], 50, 200),
        "Rump Angle": score_trait(traits["Rump Angle"], 10, 40)
    }
