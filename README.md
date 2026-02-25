# Human Activity Recognition using MediaPipe and Machine Learning

This project implements an intelligent video surveillance system for recognizing human activities (Walking, Jogging, Cycling, Yoga) by analyzing biomechanical joint angles extracted using MediaPipe Pose.

## Project Structure
- `extract_features.py`: Processes the training dataset, extracts key joint angles per frame, and generates `training_master.csv`.
- `train_model.py`: Trains a RandomForestClassifier using the generated dataset, outputs evaluation metrics, and saves the trained model (`activity_model.pkl`).
- `inference.py`: Uses the trained model to predict activities on new, unseen videos frame-by-frame and applies majority voting for the final video-level prediction.
- `requirements.txt`: List of required Python packages.

## Dataset Structure
Ensure your dataset is organized exactly like this in the root directory before running feature extraction:
```
Dataset/
    walking/
        video1.mp4
        ...
    jogging/
        video1.mp4
        ...
    cycling/
        video1.mp4
        ...
    yoga/
        video1.mp4
        ...
```

## How to Run Locally

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Extract Features**
   Make sure you have placed the videos in the `Dataset/` folder.
   ```bash
   python extract_features.py
   ```
   This will create a CSV file for each video and combine them into `training_master.csv`.

3. **Train the Model**
   ```bash
   python train_model.py
   ```
   This trains the model, saves `activity_model.pkl`, and generates a `confusion_matrix.png`.

4. **Inference / Testing**
   To test a new video:
   ```bash
   python inference.py path/to/your/test_video.mp4
   ```
   This outputs `test_video_predictions.csv` and prints the final predicted label based on majority voting.

---

## How to Run in Google Colab

You can easily run this pipeline in Google Colab using your Google Drive.

1. Create a new notebook in Google Colab.
2. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Upload this project folder to your Google Drive (e.g., to `MyDrive/InfosysProject`).
4. Ensure your `Dataset/` folder is placed inside this project folder in Drive.
5. In Colab, change the directory to your project folder:
   ```python
   import os
   os.chdir('/content/drive/MyDrive/InfosysProject')
   ```
6. Install dependencies (if any are missing from Colab's default environment):
   ```python
   !pip install mediapipe opencv-python pandas scikit-learn matplotlib seaborn
   ```
7. Run the feature extraction step:
   ```python
   !python extract_features.py
   ```
8. Train the model:
   ```python
   !python train_model.py
   ```
9. Test a new video (upload a test video to your drive first):
   ```python
   !python inference.py path/to/test_video.mp4
   ```

## Technical Details

- **Feature Extraction**: Uses `mediapipe.solutions.pose` to find 33 body landmarks.
- **Angles Calculated**: Left/Right Knee, Left/Right Elbow, Left/Right Shoulder, and Left/Right Hip angles are computed using 3-point geometry.
- **Model**: `RandomForestClassifier` with 100 estimators.
- **Evaluation**: Split 80/20 for training and validation. Metrics include Accuracy, Precision, Recall, F1-Score, and a visual Confusion Matrix.
- **Voting Logic**: The final prediction for a test video is determined by identifying the most frequently predicted activity across all processed frames (majority voting).
