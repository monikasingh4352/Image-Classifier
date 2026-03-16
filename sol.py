import cv2
import numpy as np
import mahotas
import json
import os
import time
import csv
from collections import defaultdict

# ===============================
# FEATURE EXTRACTION
# ===============================

def extract_features(image_path):
    
    # 1. READ THE IMAGE FIRST
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    
    # 2. THEN RESIZE IT
    target_width = 800
    h, w = img.shape[:2]
    if w != target_width:
        ratio = target_width / float(w)
        target_height = int(float(h) * ratio)
        img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # 3. THEN CONVERT TO GRAYSCALE
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    total_pixels = h * w

    features = {}

    # Haralick (13 features)
    glcm = mahotas.features.haralick(gray).mean(axis=0)

    names = [
        "har_contrast","har_correlation","har_energy",
        "har_homogeneity","har_variance","har_sum_avg",
        "har_sum_var","har_sum_entropy","har_entropy",
        "har_diff_var","har_diff_entropy","har_imc1","har_imc2"
    ]

    for i, name in enumerate(names):
        features[name] = float(glcm[i])

    # Edge Density
    edges = cv2.Canny(gray, 50, 150)
    features["edge_density"] = np.sum(edges > 0) / total_pixels

    # White Ratio
    thresh = mahotas.thresholding.otsu(gray)
    binary = (gray > thresh).astype(np.uint8)
    features["white_ratio"] = np.mean(binary)

    # Column Detection
    col_projection = np.sum(edges, axis=0)
    smooth = np.convolve(col_projection, np.ones(51)/51, mode="same")

    peaks = np.sum(
        (smooth[1:-1] > smooth[:-2]) &
        (smooth[1:-1] > smooth[2:])
    )

    peak_indices = np.where(
        (smooth[1:-1] > smooth[:-2]) &
        (smooth[1:-1] > smooth[2:])
    )[0]

    features["vertical_peaks"] = float(len(peak_indices))
    features["column_strength"] = float(np.max(smooth))

    # Hough Lines
    lines = cv2.HoughLinesP(edges,1,np.pi/180,80,50,10)
    if lines is not None:
        features["hough_lines"] = float(len(lines))
    else:
        features["hough_lines"] = 0.0

    # Color entropy
    hist = cv2.calcHist([img],[0],None,[32],[0,256])
    p = hist/np.sum(hist)
    p = p[p>0]
    features["color_entropy"] = float(-np.sum(p*np.log2(p)))
    
    # Text Density (Connected Components)
    num_labels, labels = cv2.connectedComponents(binary)
    features["text_component_count"] = float(num_labels / total_pixels * 10000)

    # Horizontal Projection
    row_projection = np.sum(edges, axis=1)
    smooth_row = np.convolve(row_projection, np.ones(51)/51, mode="same")

    row_peaks = np.sum(
        (smooth_row[1:-1] > smooth_row[:-2]) &
        (smooth_row[1:-1] > smooth_row[2:])
    )
    features["horizontal_peaks"] = float(row_peaks)
    
    # Stroke Width Estimate
    distance = cv2.distanceTransform(255 - binary*255, cv2.DIST_L2, 5)
    features["avg_stroke_width"] = float(np.mean(distance))
  
    # Colorfulness
    (B, G, R) = cv2.split(img.astype("float"))
    rg = np.abs(R - G)
    yb = np.abs(0.5*(R + G) - B)
    features["colorfulness"] = float(np.sqrt(np.mean(rg**2) + np.mean(yb**2)))
    
    # ===============================
    # NEW HEURISTICS
    # ===============================

    # 1. Face Detection
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if os.path.exists(cascade_path):
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray_eq = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=8, minSize=(30, 30))
        features["face_count"] = float(len(faces))
        face_area = sum([fw * fh for (fx, fy, fw, fh) in faces])
        features["face_area_ratio"] = float(face_area / total_pixels)
    else:
        features["face_count"] = 0.0
        features["face_area_ratio"] = 0.0

    # 2. Gutter Width
    binary_proj = (col_projection < (np.max(col_projection) * 0.15)).astype(int)
    max_gutter = 0
    current_gutter = 0
    for val in binary_proj:
        if val == 1:
            current_gutter += 1
            max_gutter = max(max_gutter, current_gutter)
        else:
            current_gutter = 0
    features["max_gutter_width"] = float(max_gutter / w)

    # 3. Text Block Density
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    features["text_block_density"] = float(np.sum(dilated_edges > 0) / total_pixels)
    
    return features


# ===============================
# TEMPLATE BUILDING
# ===============================

def build_template(dataset_path):
    print("\n--- BUILDING NEW TEMPLATE ---")
    print("Extracting features from all images... This will take a moment.")
    template = {}

    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue

        all_features = []
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                features = extract_features(img_path)
                all_features.append(features)
            except Exception as e:
                pass # Skip silently on error

        if not all_features:
            continue

        template[class_name] = {}
        for key in all_features[0].keys():
            values = [f[key] for f in all_features]
            mean = np.mean(values)
            std  = np.std(values) + 1e-6

            template[class_name][key] = {
                "mean": float(mean),
                "std": float(std),
                "weight": 1.0
            }

    template = assign_weights(template)

    with open("template.json","w") as f:
        json.dump(template,f,indent=4)

    print("--- TEMPLATE BUILD COMPLETE ---\n")
    return template


# ===============================
# WEIGHT ASSIGNMENT
# ===============================

def assign_weights(template):
    feature_means = {}
    feature_stds = {}

    for class_name in template:
        for feature in template[class_name]:
            if feature not in feature_means:
                feature_means[feature] = []
                feature_stds[feature] = []
            
            feature_means[feature].append(template[class_name][feature]["mean"])
            feature_stds[feature].append(template[class_name][feature]["std"])

    for class_name in template:
        for feature in template[class_name]:
            inter_var = np.var(feature_means[feature])
            mean_intra_var = np.mean(np.array(feature_stds[feature])**2) + 1e-6
            
            fisher_weight = inter_var / mean_intra_var
            bounded_weight = min(max(fisher_weight, 0.5), 5.0)
            template[class_name][feature]["weight"] = float(bounded_weight)

    return template


# ===============================
# CLASSIFICATION
# ===============================

def classify_image(image_path, template):
    features = extract_features(image_path)
    scores = {}

    for class_name in template:
        distance = 0
        for feature in template[class_name]:
            
            # If the feature exists in the image but not the template, skip it.
            if feature not in template[class_name]:
                continue
                
            mean = template[class_name][feature]["mean"]
            std  = template[class_name][feature]["std"]
            weight = template[class_name][feature]["weight"]

            z = (features[feature] - mean) / std
            z_clipped = max(min(z, 4.0), -4.0)
            distance += weight * (z_clipped ** 2)

        scores[class_name] = distance / len(template[class_name])

    predicted = min(scores, key=scores.get)
    
    inverse_distances = {cls: 1.0 / (dist + 1e-6) for cls, dist in scores.items()}
    total_weight = sum(inverse_distances.values())
    confidence = (inverse_distances[predicted] / total_weight) * 100.0

    return predicted, confidence, features

# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    
    test_path = r"C:\Users\monika\Desktop\dataset"
    
    # 1. FORCE TEMPLATE REBUILD IF OUTDATED
    needs_rebuild = False
    if not os.path.exists("template.json"):
        needs_rebuild = True
    else:
        with open("template.json", "r") as f:
            existing_template = json.load(f)
            # Check if the new features are in the JSON. If not, rebuild.
            first_class = list(existing_template.keys())[0]
            if "face_count" not in existing_template[first_class]:
                print("Old template detected. Rebuilding...")
                needs_rebuild = True
                
    if needs_rebuild:
        template = build_template(test_path)
    else:
        with open("template.json", "r") as f:
            template = json.load(f)

    # 2. GENERATE CSV FILENAME
    timestamp = int(time.time())
    output_csv_path = f"classification_results_{timestamp}.csv"

    total = 0
    correct = 0
    confusion = defaultdict(lambda: defaultdict(int))

    print(f"Starting classification... Saving to: {output_csv_path}")

    # 3. RUN CLASSIFICATION
    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        headers_written = False

        for class_name in os.listdir(test_path):
            class_folder = os.path.join(test_path, class_name)
            if not os.path.isdir(class_folder):
                continue

            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)

                try:
                    predicted, confidence, features = classify_image(img_path, template)

                    total += 1
                    confusion[class_name][predicted] += 1
                    if predicted == class_name:
                        correct += 1

                    if not headers_written:
                        feature_names = list(features.keys())
                        headers = ["Filename", "True_Class", "Predicted_Class", "Confidence_%"] + feature_names
                        csv_writer.writerow(headers)
                        headers_written = True

                    row = [img_name, class_name, predicted, round(confidence, 2)] + [features[f] for f in feature_names]
                    csv_writer.writerow(row)

                    print(f"{img_name} → Predicted: {predicted} | True: {class_name} | Conf: {round(confidence, 2)}%")

                except Exception as e:
                    print(f"Error reading {img_name}: {e}")

    # 4. FINAL OUTPUT
    accuracy = (correct / total) * 100 if total else 0

    print("\n==============================")
    print("Total Images:", total)
    print("Correct:", correct)
    print(f"Accuracy: {round(accuracy, 2)}%")
    print(f"Results saved to: {output_csv_path}")

    print("\nConfusion Matrix:")
    for true_class in confusion:
        print(true_class, dict(confusion[true_class]))