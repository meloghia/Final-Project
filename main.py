import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

class BaseballDetectorApp:
    def __init__(self, master):
        self.master = master
        master.title("Baseball Pitch Detector")
        master.geometry("800x600")
        master.configure(bg="#f0f0f0")

        self.video_source = None
        self.cap = None
        self.playback_delay = 50  # Delay in milliseconds (adjust to control speed)
        self.is_playing = False
        self.is_paused = False
        self.current_frame = 0

        self.setup_ui()

    def setup_ui(self):
        # Main frame
        self.main_frame = tk.Frame(self.master, bg="#f0f0f0")
        self.main_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Video display
        self.video_label = tk.Label(self.main_frame)
        self.video_label.pack(pady=10)

        # Control frame
        self.control_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.control_frame.pack(pady=10)

        # Control buttons
        self.play_button = tk.Button(self.control_frame, text="Play", command=self.play_video, font=("Arial", 12), bg="#28a745", fg="white", padx=10, pady=5)
        self.play_button.pack(side=tk.LEFT, padx=5)

        self.pause_button = tk.Button(self.control_frame, text="Pause", command=self.pause_video, font=("Arial", 12), bg="#ffc107", fg="white", padx=10, pady=5)
        self.pause_button.pack(side=tk.LEFT, padx=5)

        self.rewind_button = tk.Button(self.control_frame, text="Rewind", command=self.rewind_video, font=("Arial", 12), bg="#dc3545", fg="white", padx=10, pady=5)
        self.rewind_button.pack(side=tk.LEFT, padx=5)

        # Result label
        self.result_label = tk.Label(self.main_frame, text="Upload a video to start analysis", font=("Arial", 12), bg="#f0f0f0")
        self.result_label.pack(pady=10)

        # Upload button
        self.upload_button = tk.Button(self.master, text="Upload Video", command=self.upload_video, font=("Arial", 12), bg="#007bff", fg="white", padx=10, pady=5)
        self.upload_button.pack(pady=20)

    def upload_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if file_path:
            self.video_source = file_path
            self.cap = cv2.VideoCapture(file_path)
            self.result_label.config(text="Analyzing video...")
            self.prev_frame = None
            self.current_frame = 0
            self.is_playing = False
            self.is_paused = False
            self.detect_baseball()

    def play_video(self):
        if self.cap is not None and self.cap.isOpened():
            self.is_playing = True
            self.is_paused = False
            self.detect_baseball()

    def pause_video(self):
        self.is_playing = False
        self.is_paused = True

    def rewind_video(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0
            self.is_playing = False
            self.is_paused = False
            self.detect_baseball()

    def detect_baseball(self):
        if self.cap is not None and self.cap.isOpened():
            if self.is_playing:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = self.cap.read()
                
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.GaussianBlur(gray, (21, 21), 0)

                    if self.prev_frame is None:
                        self.prev_frame = gray

                    # Compute the absolute difference between the current frame and the previous frame
                    frame_delta = cv2.absdiff(self.prev_frame, gray)
                    thresh = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)[1]
                    thresh = cv2.dilate(thresh, None, iterations=2)
                    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    detected = False
                    for contour in contours:
                        if cv2.contourArea(contour) < 100:  # Adjust minimum contour area
                            continue

                        (x, y, w, h) = cv2.boundingRect(contour)
                        if self.is_valid_baseball(gray, x, y, w, h):
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            detected = True

                    if detected:
                        self.result_label.config(text="Baseball detected!")
                    else:
                        self.result_label.config(text="No baseball detected.")

                    # Update the previous frame
                    self.prev_frame = gray

                    # Convert the frame to ImageTk format
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.video_label.imgtk = imgtk
                    self.video_label.config(image=imgtk)

                    # Increment the current frame
                    self.current_frame += 1

                    # Schedule the next frame
                    self.master.after(self.playback_delay, self.detect_baseball)
                else:
                    self.cap.release()
                    self.result_label.config(text="Finished analyzing the video.")
            elif self.is_paused:
                # If paused, do nothing
                self.master.after(100, self.detect_baseball)  # Polling interval

    def is_valid_baseball(self, gray_frame, x, y, w, h):
        # Example: Check if the detected object falls within expected size and shape ranges
        aspect_ratio = w / float(h)
        if aspect_ratio < 0.8 or aspect_ratio > 1.2:
            return False
        
        area = w * h
        if area < 200 or area > 1500:  # Adjust contour area ranges for smaller objects
            return False
        
        return True

if __name__ == "__main__":
    root = tk.Tk()
    app = BaseballDetectorApp(root)
    root.mainloop()
