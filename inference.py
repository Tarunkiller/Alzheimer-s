import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import sys
import os
from collections import Counter

# We can reuse the angle calculation from extract_features
from extract_features import calculate_angle, get_landmark_coords

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def predict_video(video_path, model_path='activity_model.pkl'):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return

    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}. Please train the model first.")
        return

    print("Loading model...")
    model = joblib.load(model_path)
    
    print(f"Processing {video_path}...")
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    predictions = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Optimize processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        results = pose.process(image)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            try:
                # Coords
                shoulder_l = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
                elbow_l = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW)
                wrist_l = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_WRIST)
                
                shoulder_r = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
                elbow_r = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW)
                wrist_r = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST)
                
                hip_l = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
                knee_l = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
                ankle_l = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
                
                hip_r = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
                knee_r = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE)
                ankle_r = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)
                
                # Calculate angles
                angle_elbow_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
                angle_elbow_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
                angle_shoulder_l = calculate_angle(hip_l, shoulder_l, elbow_l)
                angle_shoulder_r = calculate_angle(hip_r, shoulder_r, elbow_r)
                angle_hip_l = calculate_angle(shoulder_l, hip_l, knee_l)
                angle_hip_r = calculate_angle(shoulder_r, hip_r, knee_r)
                angle_knee_l = calculate_angle(hip_l, knee_l, ankle_l)
                angle_knee_r = calculate_angle(hip_r, knee_r, ankle_r)
                
                # Format features for prediction
                features = [[
                    angle_elbow_l, angle_elbow_r, angle_shoulder_l, angle_shoulder_r,
                    angle_hip_l, angle_hip_r, angle_knee_l, angle_knee_r
                ]]
                
                # Predict frame activity
                pred_activity = model.predict(features)[0]
                
                predictions.append({
                    'frame': frame_count,
                    'predicted_activity': pred_activity
                })
                
            except Exception as e:
                # Landmark extraction failed for this frame
                pass
                
    cap.release()
    
    if not predictions:
        print("Could not extract any poses from the video.")
        return
        
    # Save frame-wise predictions to CSV
    output_csv = 'test_video_predictions.csv'
    df = pd.DataFrame(predictions)
    df.to_csv(output_csv, index=False)
    
    print(f"Processed {len(predictions)} frames. Frame-wise predictions saved to '{output_csv}'")
    
    # Majority Voting
    all_preds = [p['predicted_activity'] for p in predictions]
    voting_counts = Counter(all_preds)
    final_activity = voting_counts.most_common(1)[0][0]
    
    print("\n--- Final Results ---")
    print(f"Majority Voting Distribution: {dict(voting_counts)}")
    print(f"==========================================")
    print(f"FINAL PREDICTED ACTIVITY: {final_activity.upper()}")
    print(f"==========================================")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_test_video>")
    else:
        video_path = sys.argv[1]
        predict_video(video_path)
