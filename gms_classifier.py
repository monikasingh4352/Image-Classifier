import cv2
import numpy as np
import os
import csv
from scipy.signal import find_peaks
from collections import Counter

# =====================================================
# HELPERS
# =====================================================

def clamp(x, a=0, b=1):
    return max(a, min(b, float(x)))

def nz(x):
    return 0.0 if np.isnan(x) or np.isinf(x) else float(x)

def compute_entropy(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0,256])
    hist = hist / (hist.sum() + 1e-9)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))


# =====================================================
# FEATURE EXTRACTION
# =====================================================

def extract_features(img):

    img = cv2.resize(img, (800, 1000))
    h, w = img.shape[:2]
    total_area = h * w
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 80, 160)
    edge_density = np.mean(edges > 0)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    vertical_strength = np.mean(np.abs(sobelx))
    horizontal_strength = np.mean(np.abs(sobely))
    directional_edge_ratio = vertical_strength / (horizontal_strength + 1e-6)

    angles = np.arctan2(sobely, sobelx)
    hist, _ = np.histogram(angles, bins=18, range=(-np.pi, np.pi))
    hist = hist / (hist.sum() + 1e-9)
    hist = hist[hist > 0]
    edge_orientation_entropy = -np.sum(hist * np.log2(hist))

    vproj = np.mean(edges, axis=0)
    hproj = np.mean(edges, axis=1)

    col_peaks, _ = find_peaks(vproj, distance=max(10, w//60))
    row_peaks, _ = find_peaks(hproj, distance=max(10, h//60))

    col_density = len(col_peaks) / w
    row_density = len(row_peaks) / h

    col_spacing_cv = (
        np.std(np.diff(col_peaks)) /
        (np.mean(np.diff(col_peaks)) + 1e-6)
        if len(col_peaks) > 3 else 1.5
    )

    row_spacing_cv = (
        np.std(np.diff(row_peaks)) /
        (np.mean(np.diff(row_peaks)) + 1e-6)
        if len(row_peaks) > 3 else 1.5
    )

    periodicity_strength = np.std(vproj) / (np.mean(vproj) + 1e-6)
    row_periodicity_strength = np.std(hproj) / (np.mean(hproj) + 1e-6)

    valleys = vproj < (0.2 * np.max(vproj))
    gap_lengths = []
    count = 0
    for val in valleys:
        if val:
            count += 1
        else:
            if count > 0:
                gap_lengths.append(count)
            count = 0

    if gap_lengths:
        vertical_whitespace_ratio = max(gap_lengths) / w
    else:
        vertical_whitespace_ratio = 0

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 10
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    text_density = np.mean(binary > 0)

    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    heights = []

    for c in cnts:
        area = cv2.contourArea(c)
        if area > 30:
            areas.append(area)
            x,y,wc,hc = cv2.boundingRect(c)
            heights.append(hc)

    if areas:
        cc_cv = np.std(areas)/(np.mean(areas)+1e-6)
    else:
        cc_cv = 2.0

    if heights:
        text_height_cv = np.std(heights)/(np.mean(heights)+1e-6)
    else:
        text_height_cv = 2.0

    non_text = cv2.bitwise_not(binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(non_text)

    large_blocks = 0
    largest_block_ratio = 0

    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        area_ratio = area / total_area

        if area_ratio > 0.08:
            aspect_ratio = bw / (bh + 1e-6)
            if 0.4 < aspect_ratio < 3.5:
                large_blocks += 1
                largest_block_ratio = max(largest_block_ratio, area_ratio)

    entropy = compute_entropy(gray)
    color_std = np.mean(np.std(img.reshape(-1,3), axis=0))
    white_ratio = np.mean(gray > 235)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    center_x, center_y = magnitude.shape[0]//2, magnitude.shape[1]//2
    horizontal_energy = np.sum(magnitude[center_x-5:center_x+5, :])
    vertical_energy = np.sum(magnitude[:, center_y-5:center_y+5])
    fft_directional_ratio = horizontal_energy / (vertical_energy + 1e-6)

    return {
        "edge_density": edge_density,
        "col_density": col_density,
        "col_spacing_cv": nz(col_spacing_cv),
        "periodicity_strength": nz(periodicity_strength),
        "row_periodicity_strength": nz(row_periodicity_strength),
        "row_spacing_cv": nz(row_spacing_cv),
        "text_density": text_density,
        "large_blocks": large_blocks,
        "largest_block_ratio": largest_block_ratio,
        "entropy": entropy,
        "color_std": color_std,
        "white_ratio": white_ratio,
        "cc_cv": nz(cc_cv),
        "directional_edge_ratio": directional_edge_ratio,
        "edge_orientation_entropy": edge_orientation_entropy,
        "text_height_cv": nz(text_height_cv),
        "vertical_whitespace_ratio": vertical_whitespace_ratio,
        "fft_directional_ratio": fft_directional_ratio
    }


def score_general(f):
    return np.mean([
        clamp(f["entropy"]/8),
        clamp(f["edge_orientation_entropy"]/4),
        clamp(1 - f["col_density"]*80),
        clamp(1 - f["text_density"] * 6),     
        clamp(1 - f["col_density"] * 80),      
        clamp(1 - f["vertical_whitespace_ratio"] * 6),
        clamp(f["color_std"] / 60)
    ])

def score_multicol(f):
    return np.mean([
        clamp(f["text_density"] * 4),          
        clamp(f["col_density"] * 80),          
        clamp(f["vertical_whitespace_ratio"] * 6),  
        clamp(1 - f["col_spacing_cv"]),
        clamp(f["periodicity_strength"] * 2),
        clamp(f["row_periodicity_strength"] * 2)
    ])

def score_magazine(f):
    if f["text_density"] < 0.08 or f["large_blocks"] <=0:
        return 0.0

    return np.mean([
        clamp(f["text_density"]*2),
        clamp(f["large_blocks"]/2),
        clamp(f["largest_block_ratio"]*4),
        clamp(f["color_std"]/60),
        clamp(f["entropy"]/8),
        clamp(f["text_height_cv"])
    ])


def classify(features):
    g = score_general(features)
    mcol = score_multicol(features)
    mag = score_magazine(features)

    total = g + mcol + mag + 1e-9

    norm_g = g / total
    norm_mcol = mcol / total
    norm_mag = mag / total

  
    if features["largest_block_ratio"] > 0.1 or norm_mag > 0.4:
        label = "MAGAZINE"

    elif (
        features["text_density"] > 0.08 and
        features["col_density"] > 0.02 and
        features["vertical_whitespace_ratio"] > 0.05 and
        norm_mcol > 0.45
    ):
        label = "MULTICOLUMN"

    else:
        label = "GENERAL"

    norm_scores = {
        "GENERAL": round(norm_g, 3),
        "MULTICOLUMN": round(norm_mcol, 3),
        "MAGAZINE": round(norm_mag, 3)
    }

    return label, norm_scores


# =====================================================
# DATASET EVALUATION PIPELINE
# =====================================================
def evaluate_dataset(dataset_dir):
    # Maps your folder names to the expected label from your classify function
    folder_to_label = {
        "general": "GENERAL",
        "multicolumn": "MULTICOLUMN",
        "magazine": "MAGAZINE"
    }

    total_images = 0
    correct_predictions = 0

    class_stats = {
        "GENERAL": {"total": 0, "correct": 0},
        "MULTICOLUMN": {"total": 0, "correct": 0},
        "MAGAZINE": {"total": 0, "correct": 0}
    }

    print(f"Starting evaluation on dataset: {dataset_dir}\n")

    for folder_name, expected_label in folder_to_label.items():
        folder_path = os.path.join(dataset_dir, folder_name)

        if not os.path.exists(folder_path):
            print(f"Skipping {folder_name}, folder not found at {folder_path}.")
            continue

        print(f"Processing folder: {folder_name}...")

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                continue

            image_path = os.path.join(folder_path, filename)
            
            img = cv2.imread(image_path)
            if img is None:
                continue
                
            # Use your exact original functions
            feats = extract_features(img)
            pred_label, scores = classify(feats)

            total_images += 1
            class_stats[expected_label]["total"] += 1

            if pred_label == expected_label:
                correct_predictions += 1
                class_stats[expected_label]["correct"] += 1

    print("\n==================================================")
    print("FINAL RESULTS")
    print("==================================================")

    if total_images == 0:
        print("No images found. Please check your dataset directory path.")
        return

    overall_accuracy = (correct_predictions / total_images) * 100

    print(f"Total Images Processed : {total_images}")
    print(f"Total Correct          : {correct_predictions}")
    print(f"OVERALL ACCURACY       : {overall_accuracy:.2f}%\n")

    print("--- Per-Class Accuracy ---")
    for label, stats in class_stats.items():
        if stats["total"] > 0:
            acc = (stats["correct"] / stats["total"]) * 100
            print(f"{label:<12}: {acc:>6.2f}% ({stats['correct']}/{stats['total']})")


if __name__ == "__main__":
    # Change this to your actual dataset directory containing the 3 class folders
    DATASET_DIR = r"C:\Users\monika\Desktop\dataset_GMS"
    evaluate_dataset(DATASET_DIR)
