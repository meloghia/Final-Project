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
       master.configure(bg="#1e3d59")  # Background color


       self.video_source = None
       self.cap = None
       self.playback_delay = 1  # Delay in milliseconds (adjust to control speed)
       self.is_playing = False
       self.is_paused = False
       self.current_frame = 0


       # Load YOLO
       self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
       self.layer_names = self.net.getLayerNames()
       try:
           self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
       except IndexError:
           self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
       self.classes = []
       with open("coco.names", "r") as f:
           self.classes = [line.strip() for line in f.readlines()]


       self.setup_ui()


   def setup_ui(self):
       # Main frame
       self.main_frame = tk.Frame(self.master, bg="#a83236")
       self.main_frame.pack(expand=True, fill=tk.BOTH, padx=15, pady=10)


       # Video display frame
       self.video_frame = tk.Frame(self.main_frame, bg="#a83236")
       self.video_frame.pack(expand=True, fill=tk.BOTH, pady=10)


       # Video display
       self.video_label = tk.Label(self.video_frame)
       self.video_label.pack()


       # Control frame
       self.control_frame = tk.Frame(self.main_frame, bg="#a83236")
       self.control_frame.pack(pady=10)


       # Control buttons
       self.pause_button = tk.Button(self.control_frame, text="Pause", command=self.pause_video, font=("Arial", 12), bg="#ffc107", fg="red", padx=10, pady=5)
       self.pause_button.pack(side=tk.LEFT, padx=5)


       self.rewind_button = tk.Button(self.control_frame, text="Rewind", command=self.rewind_video, font=("Arial", 12), bg="#dc3545", fg="red", padx=10, pady=5)
       self.rewind_button.pack(side=tk.LEFT, padx=5)


       # Result label
       self.result_label = tk.Label(self.main_frame, text="Upload a video to start analysis", font=("Arial", 12), bg="#f0f0f0", fg="#1e3d59")
       self.result_label.pack(pady=10)


       # Upload button
       self.upload_button = tk.Button(self.master, text="Upload Video", command=self.upload_video, font=("Arial", 12), bg="#007bff", fg="red", padx=10, pady=5)
       self.upload_button.pack(pady=20)


   def upload_video(self):
       file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
       if file_path:
           self.video_source = file_path
           self.cap = cv2.VideoCapture(file_path)
           self.result_label.config(text="Analyzing video...")
           self.prev_frame = None
           self.current_frame = 0
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
           self.is_playing = True
           self.is_paused = False
           self.detect_baseball()


   def detect_baseball(self):
       if self.cap is not None and self.cap.isOpened():
           if self.is_playing:
               self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
               ret, frame = self.cap.read()


               if ret:
                   height, width, channels = frame.shape


                   # Detecting objects
                   blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                   self.net.setInput(blob)
                   outs = self.net.forward(self.output_layers)


                   # Showing information on the screen
                   class_ids = []
                   confidences = []
                   boxes = []
                   for out in outs:
                       for detection in out:
                           scores = detection[5:]
                           class_id = np.argmax(scores)
                           confidence = scores[class_id]
                           if confidence > 0.5:
                               # Object detected
                               center_x = int(detection[0] * width)
                               center_y = int(detection[1] * height)
                               w = int(detection[2] * width)
                               h = int(detection[3] * height)


                               # Rectangle coordinates
                               x = int(center_x - w / 2)
                               y = int(center_y - h / 2)
                               boxes.append([x, y, w, h])
                               confidences.append(float(confidence))
                               class_ids.append(class_id)


                   indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                   if len(indexes) > 0:
                       for i in range(len(boxes)):
                           if i in indexes:
                               x, y, w, h = boxes[i]
                               label = str(self.classes[class_ids[i]])
                               if label == "sports ball":
                                   cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                   cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                                   self.result_label.config(text="Baseball detected!")
                                   break
                       else:
                           self.result_label.config(text="No baseball detected.")
                   else:
                       self.result_label.config(text="No baseball detected.")


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


if __name__ == "__main__":
   root = tk.Tk()
   app = BaseballDetectorApp(root)
   root.mainloop()



