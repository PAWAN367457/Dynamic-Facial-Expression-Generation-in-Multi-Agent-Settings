import os
import cv2
import torch
import pandas as pd
from tqdm import tqdm

def extract_clips_gpu(recording_dir, annotation_csv, split_csv, output_dir, clip_len=10, fps=25, batch_size=32):
    os.makedirs(output_dir, exist_ok=True)

    # Check GPU availability
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    print(f"🚀 Using {'GPU' if use_gpu else 'CPU'} for clip extraction")

    # Load annotations and splits
    annotations = pd.read_csv(annotation_csv)
    splits = pd.read_csv(split_csv)
    splits = dict(zip(splits["Recording"], splits["split"]))

    recording_col = "recording"
    recordings = annotations[recording_col].unique()

    for rec_id in tqdm(recordings, desc="Processing Recordings", unit="rec"):
        rec_split = splits.get(rec_id, "train")
        rec_path = os.path.join(recording_dir, rec_id)

        if not os.path.exists(rec_path):
            print(f"⚠ Missing recording folder: {rec_id}")
            continue

        rec_df = annotations[annotations[recording_col] == rec_id]

        for _, row in tqdm(rec_df.iterrows(), total=len(rec_df), desc=f"{rec_id} Segments", leave=False):
            speaker = int(row["speaker"])
            start = int(row["start"] * fps)
            end = int(row["end"] * fps)

            for view in [1, 2]:
                video_path = os.path.join(rec_path, f"subjectPos{speaker}.video{view}.avi")
                if not os.path.exists(video_path):
                    continue

                cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                start_frame = min(start, total_frames)
                end_frame = min(end, total_frames)

                num_clips = max(1, (end_frame - start_frame) // (clip_len * fps))
                clip_idx = 0

                for frame_start in tqdm(
                    range(start_frame, end_frame, clip_len * fps),
                    total=num_clips,
                    desc=f"Speaker {speaker} View {view}",
                    leave=False
                ):
                    frame_end = min(frame_start + clip_len * fps, end_frame)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

                    frames = []
                    for _ in range(frame_end - frame_start):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame_tensor = torch.from_numpy(frame).to(device)
                        frames.append(frame_tensor)

                        if len(frames) >= batch_size:
                            frames_np = torch.stack(frames).cpu().numpy()
                            frames.clear()

                    if len(frames) > 0:
                        frames_np = torch.stack(frames).cpu().numpy()
                        frames.clear()

                    out_dir = os.path.join(output_dir, rec_split, rec_id, f"subjectPos{speaker}")
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, f"clip_{view}_{clip_idx}.avi")

                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter(out_path, fourcc, fps, (
                        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    ))

                    for f in frames_np:
                        out.write(f)
                    out.release()

                    clip_idx += 1

                cap.release()

    print("✅ GPU-accelerated clip extraction completed!")

# ------------------------------
# Example usage
# ------------------------------
extract_clips_gpu(
    recording_dir="/home/mudasir/MPII/all_views",
    annotation_csv="/home/mudasir/MPII/voice_outputs/all_recordings_voiceactivity.csv",
    split_csv="/home/mudasir/MPII/voice_outputs/train_val_test_split.csv",
    output_dir="/home/mudasir/MPII/10s_clips_gpu",  # ✅ SEPARATE OUTPUT FOLDER
    clip_len=10,
    fps=25,
    batch_size=64  # Bigger batch size → Faster processing
)
