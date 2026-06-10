import cv2
import os
import re
import shutil
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# LOCAL PC SETTINGS
# Change these paths for your computer
# ============================================================

MODEL_PATH = r"C:\Users\Nothospan Vision\Desktop\fish_project\fish_model.pt"

INPUT_FOLDER = r"C:\Users\Nothospan Vision\Desktop\backup1\compressed108"
RESULTS_FOLDER = r"C:\fish_project\results108"
INPUT_BACKUP_FOLDER = r"C:\fish_project\input_backup"
OUTPUT_BACKUP_FOLDER = r"C:\fish_project\output_backup"

CONFIDENCE = 0.5

# CPU only
DEVICE = "cpu"

# If you want every frame, keep this as 1.
# For faster but less detailed analysis, use 2, 3, 5, etc.
VID_STRIDE = 1


# ============================================================
# CREATE FOLDERS
# ============================================================

Path(RESULTS_FOLDER).mkdir(parents=True, exist_ok=True)
Path(INPUT_BACKUP_FOLDER).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_BACKUP_FOLDER).mkdir(parents=True, exist_ok=True)


# ============================================================
# FILENAME FORMAT FILTER
# Expected example:
# 192.168.20.101__2026-04-30__11-47-54-11-57-54_192.168.20.101.mp4
# ============================================================

VALID_FILENAME_PATTERN = re.compile(
    r'^\d{1,3}(\.\d{1,3}){3}'
    r'__'
    r'\d{4}-\d{2}-\d{2}'
    r'__'
    r'\d{2}-\d{2}-\d{2}'
    r'-'
    r'\d{2}-\d{2}-\d{2}'
    r'_'
    r'\d{1,3}(\.\d{1,3}){3}'
    r'(\.\w+)?$'
)


def is_valid_camera_filename(filename):
    name_without_ext = os.path.splitext(filename)[0]
    return bool(VALID_FILENAME_PATTERN.match(name_without_ext))


# ============================================================
# CHECK VIDEO VALIDITY
# Opens one frame to make sure the file is readable.
# ============================================================

def is_video_valid(video_path):
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False

        ret, frame = cap.read()
        cap.release()

        return ret and frame is not None

    except Exception:
        return False


def get_total_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames


# ============================================================
# DISCOVER VIDEOS
# Looks inside INPUT_FOLDER and subfolders.
# Skips wrong filenames and avoids duplicate filenames.
# ============================================================

def discover_videos():
    print("Searching for videos...")

    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    found_filenames = set()
    all_video_paths = []

    backed_up_count = 0
    already_backed_up_count = 0
    skipped_format_count = 0

    input_path = Path(INPUT_FOLDER)

    if not input_path.exists():
        print(f"Input folder does not exist: {INPUT_FOLDER}")
        return [], 0, 0, 0

    for video_path in input_path.rglob("*"):
        if not video_path.is_file():
            continue

        if video_path.suffix.lower() not in video_extensions:
            continue

        filename = video_path.name

        if not is_valid_camera_filename(filename):
            skipped_format_count += 1
            print(f"Skipped wrong filename format: {filename}")
            continue

        if filename in found_filenames:
            continue

        found_filenames.add(filename)
        all_video_paths.append(video_path)

        backup_dest = Path(INPUT_BACKUP_FOLDER) / filename

        if not backup_dest.exists():
            try:
                shutil.copy2(video_path, backup_dest)
                backed_up_count += 1
                print(f"Backed up input video: {filename}")
            except Exception as e:
                print(f"Could not back up {filename}: {e}")
        else:
            already_backed_up_count += 1

    all_video_paths = sorted(all_video_paths)

    print()
    print("Backup summary:")
    print(f"Newly backed up: {backed_up_count}")
    print(f"Already backed up: {already_backed_up_count}")
    print(f"Skipped wrong format: {skipped_format_count}")
    print(f"Valid unique videos found: {len(all_video_paths)}")

    return all_video_paths, backed_up_count, already_backed_up_count, skipped_format_count


# ============================================================
# PROCESS ONE VIDEO
# Writes one row per frame:
# frame, center_x, center_y, width_px, height_px, fish_detected, confidence
# ============================================================

def process_video_with_detailed_info(model, video_path, output_file):
    try:
        video_path = Path(video_path)
        output_file = Path(output_file)

        print()
        print(f"Analyzing: {video_path.name}")

        total_frames = get_total_frames(video_path)

        if total_frames <= 0:
            print(f"Invalid total frame count, skipping: {video_path.name}")
            return False

        results = model.predict(
            source=str(video_path),
            stream=True,
            conf=CONFIDENCE,
            device=DEVICE,
            vid_stride=VID_STRIDE,
            verbose=False
        )

        detection_count = 0
        processed_frames = set()

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("frame,center_x,center_y,width_px,height_px,fish_detected,confidence\n")

            for frame_idx, r in enumerate(results):
                processed_frames.add(frame_idx)

                boxes = r.boxes

                if boxes is not None and len(boxes) > 0:
                    # Choose the highest-confidence fish box.
                    best_box = max(boxes, key=lambda b: float(b.conf[0]))

                    x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()

                    width_px = x2 - x1
                    height_px = y2 - y1
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    conf = float(best_box.conf[0])

                    f.write(
                        f"{frame_idx},"
                        f"{center_x:.2f},"
                        f"{center_y:.2f},"
                        f"{width_px:.2f},"
                        f"{height_px:.2f},"
                        f"yes,"
                        f"{conf:.4f}\n"
                    )

                    detection_count += 1

                else:
                    f.write(f"{frame_idx},,,,,no,0.0000\n")

            # Safety: fill missing frames if prediction skips any.
            for frame_idx in range(total_frames):
                if frame_idx not in processed_frames:
                    f.write(f"{frame_idx},,,,,no,0.0000\n")

            no_detection_count = total_frames - detection_count
            detection_percent = detection_count / total_frames * 100
            no_detection_percent = no_detection_count / total_frames * 100

            f.write("\n")
            f.write("# ========== DETECTION SUMMARY ==========\n")
            f.write(f"# Total frames          : {total_frames}\n")
            f.write(f"# Frames with fish      : {detection_count} ({detection_percent:.1f}%)\n")
            f.write(f"# Frames without fish   : {no_detection_count} ({no_detection_percent:.1f}%)\n")
            f.write("# ========================================\n")

        backup_path = Path(OUTPUT_BACKUP_FOLDER) / output_file.name
        shutil.copy2(output_file, backup_path)

        print(f"Complete: {detection_count}/{total_frames} frames with fish")
        print(f"Saved result: {output_file}")
        print(f"Backed up result: {backup_path}")

        return True

    except Exception as e:
        print(f"ERROR while processing {Path(video_path).name}: {e}")
        return False


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    print("Loading YOLO model on CPU...")

    if not Path(MODEL_PATH).exists():
        print(f"Model file not found: {MODEL_PATH}")
        return

    model = YOLO(MODEL_PATH)

    print("Model loaded successfully.")
    print(f"Model classes: {model.names}")
    print(f"Using device: {DEVICE}")

    all_video_paths, backed_up_count, already_backed_up_count, skipped_format_count = discover_videos()

    if not all_video_paths:
        print("No valid videos found.")
        return

    existing_results = set()

    for result_file in Path(RESULTS_FOLDER).glob("*_fish_detailed_sizes.txt"):
        existing_results.add(result_file.name)

    for result_file in Path(OUTPUT_BACKUP_FOLDER).glob("*_fish_detailed_sizes.txt"):
        existing_results.add(result_file.name)

    videos_without_results = []

    for video_path in all_video_paths:
        base_name = video_path.stem
        expected_result_file = f"{base_name}_fish_detailed_sizes.txt"

        if expected_result_file not in existing_results:
            videos_without_results.append(video_path)

    print()
    print("Processing analysis:")
    print(f"Unique valid videos found: {len(all_video_paths)}")
    print(f"Existing result files: {len(existing_results)}")
    print(f"Videos needing processing: {len(videos_without_results)}")

    if not videos_without_results:
        print("No videos need processing. Everything is already done.")
        return

    print()
    print("Validating videos...")

    valid_videos_to_process = []
    invalid_videos = []

    for video_path in videos_without_results:
        if is_video_valid(video_path):
            valid_videos_to_process.append(video_path)
        else:
            invalid_videos.append(video_path)

    print(f"Valid videos: {len(valid_videos_to_process)}")
    print(f"Invalid videos: {len(invalid_videos)}")

    if invalid_videos:
        print()
        print("Invalid videos skipped:")
        for video_path in invalid_videos:
            print(f"  {video_path.name}")

    successful_count = 0
    failed_count = 0

    print()
    print(f"Starting processing of {len(valid_videos_to_process)} videos...")

    for i, video_path in enumerate(valid_videos_to_process, start=1):
        print()
        print("=" * 70)
        print(f"Processing [{i}/{len(valid_videos_to_process)}]: {video_path.name}")
        print("=" * 70)

        output_file = Path(RESULTS_FOLDER) / f"{video_path.stem}_fish_detailed_sizes.txt"

        success = process_video_with_detailed_info(model, video_path, output_file)

        if success:
            successful_count += 1
        else:
            failed_count += 1

        print()
        print(f"Progress: {i}/{len(valid_videos_to_process)}")
        print(f"Successful: {successful_count}")
        print(f"Failed: {failed_count}")

    total_all_results = len(existing_results) + successful_count
    completion_pct = total_all_results / len(all_video_paths) * 100 if all_video_paths else 0

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Unique videos discovered: {len(all_video_paths)}")
    print(f"Total input backups: {backed_up_count + already_backed_up_count}")
    print(f"Total result files: {total_all_results}")
    print(f"Processing completion: {total_all_results}/{len(all_video_paths)} ({completion_pct:.1f}%)")
    print(f"Successful this run: {successful_count}")
    print(f"Failed this run: {failed_count}")
    print(f"Results folder: {RESULTS_FOLDER}")
    print(f"Output backup folder: {OUTPUT_BACKUP_FOLDER}")
    print("Script completed.")


if __name__ == "__main__":
    main()
