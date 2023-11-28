import numpy as np
import cv2
import os

def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    count = 0
    max_frames = 30 # generating maximum 30 frame images

    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()

        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rotation_matrix = cv2.getRotationMatrix2D((frame.shape[1] // 2, frame.shape[0] // 2), - 135, 1)
        rotated_frame = cv2.warpAffine(gray, rotation_matrix, (frame.shape[1], frame.shape[0]))
        center_x, center_y = rotated_frame.shape[1] // 2, rotated_frame.shape[0] // 2
        cropped_frame = rotated_frame[center_y - 400 :center_y + 100, center_x + 150:center_x + 400]

        cv2.imwrite(os.path.join(output_folder, f"frame_{count}.png"), cropped_frame)
        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Generate Original Data Frames
extract_frames('data/target_data/original.avi', 'data/demo_data/original_frames/')
# Generate Denoising Data Frames
extract_frames('data/target_data/denoising.avi', 'data/demo_data/denoising_frames/')