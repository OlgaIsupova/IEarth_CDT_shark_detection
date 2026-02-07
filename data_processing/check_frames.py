import cv2
import os
import sys
from pathlib import Path

# Increase ffmpeg read attempts for videos with multiple streams
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "10000"

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import data_path


class FrameViewer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame_idx = 0

        print(f"Video loaded: {video_path}")
        print(f"Total frames: {self.total_frames}")
        print(f"FPS: {self.fps}")
        print("\nControls:")
        print("  SPACE or LEFT CLICK: Next frame")
        print("  A or RIGHT CLICK: Previous frame")
        print("  D: Forward 10 frames")
        print("  S: Backward 10 frames")
        print("  Q or ESC: Quit")

    def read_frame(self, frame_idx):
        """Read a specific frame, skipping unreadable frames"""
        if frame_idx < 0 or frame_idx >= self.total_frames:
            return None

        # Try to read the requested frame, and skip forward if it fails
        attempts = 0
        max_attempts = 5
        current_idx = frame_idx

        while attempts < max_attempts and current_idx < self.total_frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
            ret, frame = self.cap.read()
            if ret:
                self.current_frame_idx = current_idx
                return frame
            # If frame failed, try the next one
            current_idx += 1
            attempts += 1

        return None

    def draw_frame_info(self, frame):
        """Add frame number text to the frame"""
        frame_copy = frame.copy()
        height, width = frame.shape[:2]

        # Add frame number at top left
        text = f"Frame: {self.current_frame_idx + 1}/{self.total_frames}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        color = (0, 255, 0)  # Green

        # Get text size to create background
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        # Draw background rectangle
        cv2.rectangle(
            frame_copy, (5, 5), (text_size[0] + 15, text_size[1] + 15), (0, 0, 0), -1
        )

        # Draw text
        cv2.putText(frame_copy, text, (10, 30), font, font_scale, color, thickness)

        return frame_copy

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click: next frame
            self.go_next()
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click: previous frame
            self.go_previous()

    def go_next(self):
        """Go to next frame"""
        if self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
        else:
            print("Already at last frame")

    def go_previous(self):
        """Go to previous frame"""
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
        else:
            print("Already at first frame")

    def run(self):
        """Main loop"""
        window_name = "Frame Viewer"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        while True:
            frame = self.read_frame(self.current_frame_idx)

            if frame is None:
                print(
                    f"Could not read frame {self.current_frame_idx}, skipping to next readable frame..."
                )
                # Try to find next readable frame
                found = False
                for next_idx in range(
                    self.current_frame_idx + 1,
                    min(self.current_frame_idx + 100, self.total_frames),
                ):
                    test_frame = self.read_frame(next_idx)
                    if test_frame is not None:
                        found = True
                        break

                if not found:
                    print("Could not find any more readable frames. End of video.")
                    break
                continue

            # Draw frame info
            frame_with_info = self.draw_frame_info(frame)

            # Display frame
            cv2.imshow(window_name, frame_with_info)

            # Wait for key press (1 ms to keep display responsive)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:  # 'q' or ESC
                print("Exiting...")
                break
            elif key == ord(" "):  # Space bar - next frame
                self.go_next()
            elif key == ord("a"):  # 'a' - previous frame
                self.go_previous()
            elif key == ord("d"):  # 'd' - forward 10 frames
                self.current_frame_idx = min(
                    self.current_frame_idx + 10, self.total_frames - 1
                )
            elif key == ord("s"):  # 's' - backward 10 frames
                self.current_frame_idx = max(self.current_frame_idx - 10, 0)

        cv2.destroyAllWindows()
        self.cap.release()


def main():
    # Build video path from config
    video_path = os.path.join(data_path, "input_videos", "A few larger fish.MP4")

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        sys.exit(1)

    viewer = FrameViewer(video_path)
    viewer.run()


if __name__ == "__main__":
    main()
