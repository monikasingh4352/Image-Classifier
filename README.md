# Intelligent Image Classification via Feature Extraction

## Project Overview
This project is a custom image classification system developed for integration with the Jio Cloud app. Unlike standard "black-box" deep learning models, this system utilizes a highly interpretable **Statistical Template Matching** and **Feature Extraction** approach. 

By extracting specific structural, textural, and color-based heuristics from images, the model builds a weighted statistical template to classify new images based on mathematical distance. This repository contains the core Python processing scripts, feature extraction logic, and automated CSV reporting tools.

## Methodology
The core engine processes images by extracting a robust set of features, including:
* **Textural Features:** Haralick textures computed via Mahotas.
* **Structural Features:** Edge density (Canny), vertical/horizontal projections, Hough lines, and stroke width estimation.
* **Color & Content Features:** Color entropy, colorfulness, and Haar-cascade face detection.
* **Dynamic Template Building:** Calculates the mean, standard deviation, and Fisher weights for each feature across classes to build and cache a `template.json` model.
* **Distance-Based Classification:** Uses weighted z-score distances to assign predictions and confidence scores.

## Dataset Setup
Due to the large file size of the training dataset, the images are hosted externally. 

1. Download the complete `dataset.zip` file here: https://www.jioaicloud.com/l/?u=KqcnzoVt6h00fxSJz5MAYhqsL9RFy5xDEwnhVmQHQvvzJcgkBI64VyM2L7nwmJZFo8v
2. Extract the downloaded `.zip` file.
3. Place the unzipped `dataset` folder directly into the same directory as the main Python script. 
4. *(Note: If you place the dataset elsewhere, you must update the `test_path` variable in the script to match your local directory).*

## Prerequisites & Installation
This project requires **Python 3.x**. To install the necessary dependencies, run the following command in your terminal:

```bash
pip install opencv-python numpy mahotas
