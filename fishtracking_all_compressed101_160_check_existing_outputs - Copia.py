"""
Fish position analysis with YOLO on an NVIDIA GPU.
Samples approximately 5 frames per second from each source video.

IMPORTANT:
1. Edit MODEL_PATH and INPUT_ROOT_FOLDER before running.
2. Use a PyTorch CUDA build that supports your GPU.
3. Videos should be locally available (mirrored or marked "Available offline"
   in Google Drive for Desktop) before processing.
"""

import os
import re
import shutil
from pathlib import Path
from datetime import datetime, timedelta

import cv2
import torch
from ultralytics import YOLO


# ============================================================
# SETTINGS - EDIT THESE PATHS FOR YOUR PC
# ============================================================

MODEL_PATH = r"D:\fish_model.pt"

# Process compressed101, compressed102, ... compressed160.
INPUT_ROOT_FOLDER = r"C:\Users\Utente\Desktop\secondexp06"
FIRST_FOLDER_NUMBER = 101
LAST_FOLDER_NUMBER = 160

# Separate result and backup subfolders will be created for each camera IP.
RESULTS_ROOT_FOLDER = r"C:\fish_project\results_101_to_160"
OUTPUT_BACKUP_ROOT_FOLDER = r"C:\fish_project\output_backup_101_to_160"

# Existing folders from earlier runs (for example results_102 and
# output_backup102) are checked before processing anything.
LEGACY_RESULTS_FOLDER_PATTERN = r"C:\\fish_project\\results_{number}"
LEGACY_BACKUP_FOLDER_PATTERN = r"C:\\fish_project\\output_backup{number}"

# Detection settings
CONFIDENCE = 0.50
TARGET_FPS = 5.0

# GPU 0 means the first NVIDIA GPU.
DEVICE = 0
USE_HALF_PRECISION = True  # FP16 can improve speed on supported NVIDIA GPUs.


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


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


def get_video_start_end_datetimes(video_path):
    """
    Extract start and end date/time from filenames such as:
    192.168.20.101__2026-04-30__11-47-54-11-57-54_192.168.20.101.mp4
    """
    match = re.search(
        r'__(\d{4}-\d{2}-\d{2})__(\d{2}-\d{2}-\d{2})-(\d{2}-\d{2}-\d{2})_',
        video_path.name
    )
    if not match:
        return None, None

    date_text, start_text, end_text = match.groups()
    try:
        start_dt = datetime.strptime(
            f"{date_text} {start_text}", "%Y-%m-%d %H-%M-%S"
        )
        end_dt = datetime.strptime(
            f"{date_text} {end_text}", "%Y-%m-%d %H-%M-%S"
        )
        # Handle recordings that cross midnight.
        if end_dt < start_dt:
            end_dt += timedelta(days=1)
        return start_dt, end_dt
    except ValueError:
        return None, None


# ============================================================
# GPU CHECK
# ============================================================

def check_gpu():
    print("Checking NVIDIA GPU and PyTorch CUDA support...")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. The script will NOT silently switch to CPU. "
            "Check the NVIDIA driver and install a CUDA-enabled PyTorch build."
        )

    gpu_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)

    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    print(f"GPU: {gpu_name}")
    print(f"GPU compute capability: sm_{capability[0]}{capability[1]}")

    # Force a small CUDA operation so incompatible installations fail early.
    test_tensor = torch.ones((2, 2), device="cuda")
    _ = test_tensor @ test_tensor
    torch.cuda.synchronize()

    print("CUDA test passed. GPU is ready.\n")


# ============================================================
# VIDEO HELPERS
# ============================================================

def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return 0.0, 0

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return fps, total_frames


def is_video_valid(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return False

    ret, frame = cap.read()
    cap.release()
    return bool(ret and frame is not None)


def calculate_frame_stride(source_fps):
    """
    Ultralytics vid_stride is a frame interval, not an FPS setting.
    For example, 30 FPS source / stride 6 = approximately 5 analyzed FPS.
    """
    if source_fps <= 0:
        raise ValueError("The video's FPS could not be read.")

    if source_fps < TARGET_FPS:
        print(
            f"WARNING: source video is only {source_fps:.2f} FPS. "
            f"It cannot provide {TARGET_FPS:.1f} unique frames per second."
        )
        return 1

    return max(1, round(source_fps / TARGET_FPS))


# ============================================================
# DISCOVER VIDEOS
# ============================================================

def get_camera_ip(video_path):
    """Get the first camera IP from a camera video filename."""
    match = re.match(r"^(\d{1,3}(?:\.\d{1,3}){3})__", video_path.name)
    return match.group(1) if match else "unknown_camera"


def discover_videos():
    """Find valid videos in compressed101 through compressed160."""
    root_path = Path(INPUT_ROOT_FOLDER)
    print(f"Searching for compressed folders in: {root_path}")

    found_filenames = set()
    all_video_paths = []
    skipped_format_count = 0
    missing_folders = []

    for folder_number in range(FIRST_FOLDER_NUMBER, LAST_FOLDER_NUMBER + 1):
        input_path = root_path / f"compressed{folder_number}"
        if not input_path.is_dir():
            missing_folders.append(input_path.name)
            print(f"Folder not found; skipping: {input_path}")
            continue

        print(f"Scanning: {input_path}")
        for video_path in input_path.rglob("*"):
            if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue

            filename = video_path.name
            if not is_valid_camera_filename(filename):
                skipped_format_count += 1
                print(f"Skipped wrong filename format: {filename}")
                continue

            if filename in found_filenames:
                print(f"Skipped duplicate filename: {filename}")
                continue

            found_filenames.add(filename)
            all_video_paths.append(video_path)

    all_video_paths.sort()
    print("\nDiscovery summary:")
    print(f"Folders requested: compressed{FIRST_FOLDER_NUMBER} to compressed{LAST_FOLDER_NUMBER}")
    print(f"Valid unique videos: {len(all_video_paths)}")
    print(f"Skipped wrong filename format: {skipped_format_count}")
    print(f"Missing folders: {len(missing_folders)}")
    return all_video_paths


# ============================================================
# PROCESS ONE VIDEO
# One row is written for each analyzed frame (approximately 5 FPS).
# Position is the center of the highest-confidence detected fish.
# ============================================================

def process_video(model, video_path, output_file):
    video_path = Path(video_path)
    output_file = Path(output_file)

    print(f"\nAnalyzing: {video_path.name}")

    source_fps, total_source_frames = get_video_info(video_path)

    if total_source_frames <= 0 or source_fps <= 0:
        print("Invalid video metadata; skipping.")
        return False

    stride = calculate_frame_stride(source_fps)
    estimated_sample_fps = source_fps / stride

    print(f"Original video FPS: {source_fps:.3f}")
    print(f"Original frame count: {total_source_frames}")
    print(f"YOLO frame stride: {stride}")
    print(f"Expected analysis rate: approximately {estimated_sample_fps:.3f} FPS")

    sampled_frames = 0
    detected_frames = 0

    try:
        results = model.predict(
            source=str(video_path),
            stream=True,
            conf=CONFIDENCE,
            device=DEVICE,
            half=USE_HALF_PRECISION,
            vid_stride=stride,
            verbose=False
        )

        video_start_datetime, video_end_datetime = get_video_start_end_datetimes(video_path)
        video_duration_seconds = (
            total_source_frames / source_fps if source_fps > 0 else 0.0
        )

        with open(output_file, "w", encoding="utf-8") as txtfile:
            # Each row describes the time interval covered by that sampled frame.
            txtfile.write("start_time\tend_time\tframe\tx\ty\tconf\tstate\n")

            for sample_index, result in enumerate(results):
                source_frame_index = min(
                    sample_index * stride,
                    max(0, total_source_frames - 1)
                )
                start_seconds = source_frame_index / source_fps
                end_seconds = min(
                    (source_frame_index + stride) / source_fps,
                    video_duration_seconds
                )
                sampled_frames += 1

                if video_start_datetime is not None:
                    frame_start = video_start_datetime + timedelta(seconds=start_seconds)
                    frame_end = video_start_datetime + timedelta(seconds=end_seconds)
                    if video_end_datetime is not None:
                        frame_end = min(frame_end, video_end_datetime)
                    start_text = frame_start.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    end_text = frame_end.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                else:
                    # If the filename has no recording date/time, use elapsed
                    # start/end times measured from the beginning of the video.
                    start_text = f"{start_seconds:.3f}"
                    end_text = f"{end_seconds:.3f}"

                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    # Keep the highest-confidence detection in each frame.
                    best_box = max(boxes, key=lambda box: float(box.conf[0]))

                    x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
                    image_height, image_width = result.orig_shape
                    center_x = float((x1 + x2) / 2) / image_width
                    center_y = float((y1 + y2) / 2) / image_height
                    confidence = float(best_box.conf[0])

                    txtfile.write(
                        f"{start_text}\t{end_text}\t{source_frame_index}\t"
                        f"{center_x:.4f}\t{center_y:.4f}\t"
                        f"{confidence:.3f}\tdetection\n"
                    )
                    detected_frames += 1
                else:
                    txtfile.write(
                        f"{start_text}\t{end_text}\t{source_frame_index}\t"
                        f"\t\t0.000\tno_detection\n"
                    )

        if sampled_frames == 0:
            print("No frames were returned by YOLO.")
            if output_file.exists():
                output_file.unlink()
            return False

        no_detection_frames = sampled_frames - detected_frames
        detection_percent = detected_frames / sampled_frames * 100

        # Separate summary file records basic video analysis statistics.
        summary_file = output_file.with_name(output_file.stem + "_summary.txt")
        with open(summary_file, "w", encoding="utf-8") as summary:
            summary.write(f"Video: {video_path.name}\n")
            summary.write(f"Original FPS: {source_fps:.3f}\n")
            summary.write(f"Original frames: {total_source_frames}\n")
            summary.write(f"Target analysis FPS: {TARGET_FPS:.1f}\n")
            summary.write(f"Actual approximate analysis FPS: {estimated_sample_fps:.3f}\n")
            summary.write(f"Frame stride: {stride}\n")
            summary.write(f"Analyzed frames: {sampled_frames}\n")
            summary.write(f"Frames with fish: {detected_frames}\n")
            summary.write(f"Frames without fish: {no_detection_frames}\n")
            summary.write(f"Detection percentage: {detection_percent:.1f}%\n")

        print(
            f"Complete: {detected_frames}/{sampled_frames} analyzed frames "
            f"contained a fish ({detection_percent:.1f}%)."
        )
        print(f"Detection TXT saved: {output_file}")
        print(f"Summary saved: {summary_file}")
        print(f"Output backup folder: {OUTPUT_BACKUP_FOLDER}")

        return True

    except Exception as exc:
        print(f"ERROR while processing {video_path.name}: {exc}")
        return False


# ============================================================
# MAIN
# ============================================================


def find_existing_output(output_name):
    """
    Search the new combined output roots and all legacy results_N / output_backupN
    folders (101..160). Search recursively so both flat and per-camera layouts work.
    A matching result TXT OR backup TXT means that video has already been processed.
    """
    candidate_roots = [
        Path(RESULTS_ROOT_FOLDER),
        Path(OUTPUT_BACKUP_ROOT_FOLDER),
    ]

    for number in range(FIRST_FOLDER_NUMBER, LAST_FOLDER_NUMBER + 1):
        candidate_roots.append(
            Path(LEGACY_RESULTS_FOLDER_PATTERN.format(number=number))
        )
        candidate_roots.append(
            Path(LEGACY_BACKUP_FOLDER_PATTERN.format(number=number))
        )

    checked_roots = set()
    for root in candidate_roots:
        root_key = str(root).lower()
        if root_key in checked_roots:
            continue
        checked_roots.add(root_key)

        if not root.is_dir():
            continue

        # Exact filename search supports both old flat folders and nested IP folders.
        try:
            if any(path.is_file() for path in root.rglob(output_name)):
                return root
        except (PermissionError, OSError) as exc:
            print(f"WARNING: Could not fully search {root}: {exc}")

    return None


def main():
    Path(RESULTS_ROOT_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_BACKUP_ROOT_FOLDER).mkdir(parents=True, exist_ok=True)

    check_gpu()

    if not Path(MODEL_PATH).is_file():
        print(f"Model file not found: {MODEL_PATH}")
        return

    print(f"Loading YOLO model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully.")
    print(f"Model classes: {model.names}")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    all_videos = discover_videos()
    if not all_videos:
        print("No valid videos found.")
        return

    videos_to_process = []
    for video_path in all_videos:
        camera_ip = get_camera_ip(video_path)
        output_name = f"{video_path.stem}_fish_positions_5fps.txt"
        existing_location = find_existing_output(output_name)

        if existing_location is not None:
            print(
                f"Already processed; skipping: {video_path.name} "
                f"(found in {existing_location})"
            )
            continue

        videos_to_process.append(video_path)

    print("\nProcessing overview:")
    print(f"Videos discovered: {len(all_videos)}")
    print(f"Videos already processed: {len(all_videos) - len(videos_to_process)}")
    print(f"Videos to process: {len(videos_to_process)}")

    valid_videos = []
    for video_path in videos_to_process:
        if is_video_valid(video_path):
            valid_videos.append(video_path)
        else:
            print(f"Invalid/unreadable video skipped: {video_path.name}")

    successful = 0
    failed = 0

    for index, video_path in enumerate(valid_videos, start=1):
        camera_ip = get_camera_ip(video_path)
        results_folder = Path(RESULTS_ROOT_FOLDER) / camera_ip
        backup_folder = Path(OUTPUT_BACKUP_ROOT_FOLDER) / camera_ip
        results_folder.mkdir(parents=True, exist_ok=True)
        backup_folder.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print(f"VIDEO {index}/{len(valid_videos)}: {video_path.name}")
        print(f"Source folder: {video_path.parent}")
        print(f"Camera IP: {camera_ip}")
        print("=" * 70)

        output_file = results_folder / f"{video_path.stem}_fish_positions_5fps.txt"

        if process_video(model, video_path, output_file):
            shutil.copy2(output_file, backup_folder / output_file.name)
            summary_file = output_file.with_name(output_file.stem + "_summary.txt")
            if summary_file.exists():
                shutil.copy2(summary_file, backup_folder / summary_file.name)
            successful += 1
        else:
            failed += 1

        print(f"Progress: {index}/{len(valid_videos)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Videos discovered: {len(all_videos)}")
    print(f"Videos processed this run: {len(valid_videos)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Results root: {RESULTS_ROOT_FOLDER}")
    print(f"Output backup root: {OUTPUT_BACKUP_ROOT_FOLDER}")
    print("Each root contains one subfolder per camera IP.")
    print("Analysis completed.")


if __name__ == "__main__":
    main()
