import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path

def analyze_jump_video(video_path, g=9.81, save_annotated=False, annotated_path=None):
    """
    Analyze a jump in a video and return:
    - takeoff_frame
    - landing_frame
    - flight_time (s)
    - jump_height_m (m)

    Logic:
    1) Use MediaPipe Pose, foot index landmarks (31, 32) to get foot_y per frame.
    2) frame_min = frame with minimum foot_y.
    3) Take-off:
       - start i = apex_idx - 1
       - loop backwards while foot_y[i] > foot_y[i+1]
       - takeoff_idx = i + 1
    4) Landing:
       - start j = apex_idx + 1
       - loop forward while foot_y[j] < foot_y[takeoff_idx]
       - landing_idx = j
    5) Flight time and jump height from time between take-off and landing.

    Also computes right ankle angle (knee–ankle–foot index) per frame and stores it
    as `right_ankle_angle_deg` in the DataFrame.

    If save_annotated=True, writes an annotated video and returns its path.
    """

    # ---------- Setup pose ----------
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # ---------- Open video ----------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_indices = []
    time_stamps = []
    foot_y_pixels = []
    right_ankle_angles = []

    frame_idx = 0

    # ---------- Extract foot_y and right ankle angle per frame ----------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        foot_y = np.nan
        ankle_angle_deg = np.nan

        if results.pose_landmarks:
            h, w, _ = frame.shape
            lm = results.pose_landmarks.landmark

            # --- foot_y from LEFT/RIGHT_FOOT_INDEX ---
            foot_idxs = [
                mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
                mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
            ]
            y_vals = [lm[i].y * h for i in foot_idxs]
            foot_y = max(y_vals)  # lowest foot in image coordinates

            # --- right ankle angle (knee–ankle–foot index) ---
            rk = lm[mp_pose.PoseLandmark.RIGHT_KNEE]
            ra = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
            rf = lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]

            kx, ky = rk.x * w, rk.y * h
            ax, ay = ra.x * w, ra.y * h
            fx, fy = rf.x * w, rf.y * h

            v1 = np.array([kx - ax, ky - ay])  # knee -> ankle
            v2 = np.array([fx - ax, fy - ay])  # foot -> ankle

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 > 1e-6 and norm2 > 1e-6:
                cosang = np.dot(v1, v2) / (norm1 * norm2)
                cosang = np.clip(cosang, -1.0, 1.0)
                ankle_angle_deg = float(np.degrees(np.arccos(cosang)))

        frame_indices.append(frame_idx)
        time_stamps.append(frame_idx / fps if fps > 0 else np.nan)
        foot_y_pixels.append(foot_y)
        right_ankle_angles.append(ankle_angle_deg)

        frame_idx += 1

    cap.release()

    # ---------- Build DataFrame ----------
    df = pd.DataFrame({
        "frame": frame_indices,
        "time_s": time_stamps,
        "foot_y_px": foot_y_pixels,
        "right_ankle_angle_deg": right_ankle_angles
    })

    if df["foot_y_px"].isna().all():
        raise RuntimeError("No valid foot_y_px detected in any frame.")

    # ---------- 1) frame_min (apex) ----------
    apex_idx = int(df["foot_y_px"].idxmin())
    frame_min = int(df.loc[apex_idx, "frame"])

    # ---------- 2) Take-off (your backward logic) ----------
    i = apex_idx - 1
    if i < 0:
        raise RuntimeError("Apex at first frame, cannot search backwards for take-off.")

    #while i > 0 and df.loc[i, "foot_y_px"] > df.loc[i + 1, "foot_y_px"]:
    while i > 0 and df.loc[i+1, "right_ankle_angle_deg"] - df.loc[i, "right_ankle_angle_deg"]<20:
        i -= 1

    takeoff_idx = i
    frame_takeoff = int(df.loc[takeoff_idx, "frame"])
    frame_takeoff_y = float(df.loc[takeoff_idx, "foot_y_px"])
    t_takeoff = float(df.loc[takeoff_idx, "time_s"])

    # ---------- 3) Landing (your forward logic) ----------
    j = apex_idx + 1
    if j >= len(df):
        raise RuntimeError("Apex at last frame, cannot search forwards for landing.")

    while j < len(df) - 1 and df.loc[j, "foot_y_px"] < frame_takeoff_y:
        j += 1

    landing_idx = j
    frame_landing = int(df.loc[landing_idx, "frame"])
    frame_landing_y = float(df.loc[landing_idx, "foot_y_px"])
    t_landing = float(df.loc[landing_idx, "time_s"])

    # ---------- 4) Flight time and jump height ----------
    flight_time = t_landing - t_takeoff  # seconds
    t_up = flight_time / 2.0
    jump_height_m = 0.5 * g * (t_up ** 2)

    annotated_video_path = None

    # ---------- 5) Save annotated video (optional) ----------
    if save_annotated:
        if annotated_path is None:
            stem = Path(video_path).stem
            annotated_video_path = f"{stem}_annotated.mp4"
        else:
            annotated_video_path = annotated_path

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video for annotation: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(annotated_video_path, fourcc, fps, (width, height))

        df_idx = df.set_index("frame")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            # draw pose
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

            # lookup values from df
            if frame_idx in df_idx.index:
                fy = df_idx.loc[frame_idx, "foot_y_px"]
                ankle_angle = df_idx.loc[frame_idx, "right_ankle_angle_deg"]
            else:
                fy = np.nan
                ankle_angle = np.nan

            # Lines of text
            if np.isnan(fy):
                text1 = f"frame {frame_idx} | foot_y = NaN"
            else:
                text1 = f"frame {frame_idx} | foot_y = {fy:.1f}px"

            if np.isnan(ankle_angle):
                text2 = "right_ankle_angle = NaN"
            else:
                text2 = f"right_ankle_angle = {ankle_angle:.1f} deg"

            cv2.putText(
                frame,
                text1,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                text2,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

            # mark takeoff / landing
            if frame_idx == frame_takeoff:
                cv2.putText(
                    frame,
                    "TAKEOFF",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA
                )

            if frame_idx == frame_landing:
                cv2.putText(
                    frame,
                    "LANDING",
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()

    # ---------- Return results ----------
    results = {
        "takeoff_frame": frame_takeoff,
        "takeoff_y": frame_takeoff_y,
        "landing_frame": frame_landing,
        "landing_y": frame_landing_y,
        "frame_min": frame_min,
        "flight_time_s": flight_time,
        "jump_height_m": jump_height_m,
        "jump_height_cm": jump_height_m * 100.0,
        "fps": fps,
        "total_frames": total_frames,
        "dataframe": df,
        "i": i,
        "annotated_video_path": annotated_video_path
    }

    return results
