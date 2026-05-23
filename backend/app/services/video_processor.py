import cv2
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips

KEYFRAME_DIR = "output/keyframes"

os.makedirs(KEYFRAME_DIR, exist_ok=True)


def calculate_frame_difference(frame1, frame2):

    hist1 = cv2.calcHist(
        [frame1],
        [0, 1, 2],
        None,
        [8, 8, 8],
        [0, 256, 0, 256, 0, 256]
    )

    hist2 = cv2.calcHist(
        [frame2],
        [0, 1, 2],
        None,
        [8, 8, 8],
        [0, 256, 0, 256, 0, 256]
    )

    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)

    score = cv2.compareHist(
        hist1,
        hist2,
        cv2.HISTCMP_BHATTACHARYYA
    )

    return score


def detect_scenes(video_path, threshold=0.015):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    success, prev_frame = cap.read()

    if not success:
        return []

    timestamps = []

    frame_index = 0

    while True:

        success, curr_frame = cap.read()

        if not success:
            break

        diff = calculate_frame_difference(
            prev_frame,
            curr_frame
        )

        print(f"Frame {frame_index} Diff: {diff}")

        if diff > threshold:

            timestamp = frame_index / fps

            # avoid duplicate nearby detections
            if len(timestamps) == 0 or timestamp - timestamps[-1] > 2:

                timestamps.append(timestamp)

                frame_path = os.path.join(
                    KEYFRAME_DIR,
                    f"frame_{len(timestamps)}.jpg"
                )

                cv2.imwrite(frame_path, curr_frame)

        prev_frame = curr_frame

        frame_index += 1

    cap.release()

    return timestamps


def create_summary_video(
    video_path,
    timestamps,
    clip_duration=1
):

    output_path = "output/summary.mp4"

    video = VideoFileClip(video_path)

    clips = []

    used_ranges = []

    for ts in timestamps:

        start = max(ts - 0.5, 0)

        end = min(
            ts + clip_duration,
            video.duration - 0.1
        )

        # skip invalid clips
        if end <= start:
            continue

        # avoid overlapping clips
        overlap = False

        for s, e in used_ranges:
            if start < e and end > s:
                overlap = True
                break

        if overlap:
            continue

        used_ranges.append((start, end))

        try:
            clip = video.subclip(start, end)

            # ensure clip has duration
            if clip.duration > 0:
                clips.append(clip)

        except Exception as e:
            print(f"Clip error: {e}")

    if len(clips) == 0:
        return None

    final_clip = concatenate_videoclips(
        clips,
        method="compose"
    )

    final_clip.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=24
    )

    video.close()

    return output_path