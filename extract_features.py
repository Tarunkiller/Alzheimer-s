import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import glob
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points (a, b, c).
    b is the vertex.
    Each point is a tuple or list [x, y].
    """
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def get_landmark_coords(landmarks, landmark_type):
    """Helper to extract [x, y] coordinates for a given landmark."""
    return [landmarks[landmark_type.value].x, landmarks[landmark_type.value].y]

def process_video(video_path, activity_label):
    """
    Process a video, extract angles for each frame, and return a dictionary containing the frame data.
    """
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    video_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make detection
        results = pose.process(image)
        
        # Extract landmarks
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            try:
                # Required Joints: Knee, Elbow, Shoulder, Hip
                
                # Coords
                # Left Arm
                shoulder_l = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
                elbow_l = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW)
                wrist_l = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_WRIST)
                
                # Right Arm
                shoulder_r = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
                elbow_r = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW)
                wrist_r = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST)
                
                # Left Leg
                hip_l = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
                knee_l = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
                ankle_l = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
                
                # Right Leg
                hip_r = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
                knee_r = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE)
                ankle_r = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)
                
                # Calculate required angles
                # Elbows
                angle_elbow_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
                angle_elbow_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
                
                # Shoulders (Hip - Shoulder - Elbow)
                angle_shoulder_l = calculate_angle(hip_l, shoulder_l, elbow_l)
                angle_shoulder_r = calculate_angle(hip_r, shoulder_r, elbow_r)
                
                # Hips (Shoulder - Hip - Knee)
                angle_hip_l = calculate_angle(shoulder_l, hip_l, knee_l)
                angle_hip_r = calculate_angle(shoulder_r, hip_r, knee_r)
                
                # Knees (Hip - Knee - Ankle)
                angle_knee_l = calculate_angle(hip_l, knee_l, ankle_l)
                angle_knee_r = calculate_angle(hip_r, knee_r, ankle_r)
                
                # Append to video data
                video_data.append({
                    'video_name': os.path.basename(video_path),
                    'frame': frame_count,
                    'left_elbow': angle_elbow_l,
                    'right_elbow': angle_elbow_r,
                    'left_shoulder': angle_shoulder_l,
                    'right_shoulder': angle_shoulder_r,
                    'left_hip': angle_hip_l,
                    'right_hip': angle_hip_r,
                    'left_knee': angle_knee_l,
                    'right_knee': angle_knee_r,
                    'activity': activity_label
                })
                
            except Exception as e:
                # If a landmark is not visible enough to calculate, skip frame
                print(f"Error on frame {frame_count} of {video_path}: {e}")
                pass
                
    cap.release()
    return video_data

def process_dataset(dataset_dir="Dataset"):
    """
    Iterate over the dataset, process all videos, extract features, and combine into a master CSV.
    """
    activities = ['walking', 'jogging', 'cycling', 'yoga']
    all_data = []
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found.")
        print("Please ensure your dataset is located in the 'Dataset/' folder.")
        return

    # Check if activities exist
    for activity in activities:
        activity_path = os.path.join(dataset_dir, activity)
        if not os.path.exists(activity_path):
            print(f"Warning: Activity folder not found: {activity_path}")
            continue
            
        # Find all MP4 or AVI files
        video_files = glob.glob(os.path.join(activity_path, "*.mp4")) + \
                      glob.glob(os.path.join(activity_path, "*.avi")) + \
                      glob.glob(os.path.join(activity_path, "*.mov"))
                      
        print(f"Found {len(video_files)} videos for activity: {activity}")
        
        for video_path in video_files:
            print(f"Processing {video_path}...")
            video_data = process_video(video_path, activity)
            
            # Save per-video CSV
            if video_data:
                video_df = pd.DataFrame(video_data)
                csv_filename = os.path.splitext(video_path)[0] + "_angles.csv"
                video_df.to_csv(csv_filename, index=False)
                print(f"Saved {len(video_data)} frames to {csv_filename}")
                
                # Append to master list
                all_data.extend(video_data)
            else:
                print(f"No frames extracted for {video_path}")

    if all_data:
        master_df = pd.DataFrame(all_data)
        master_df.to_csv('training_master.csv', index=False)
        print(f"\nSuccessfully combined {len(master_df)} total frames into 'training_master.csv'")
    else:
        print("\nNo data extracted. Verify your dataset structure and video accessibility.")

if __name__ == "__main__":
    print("Starting Feature Extraction...")
    process_dataset()
