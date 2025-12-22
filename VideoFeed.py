import os
import time
import cv2
from queue import Queue
from dataclasses import dataclass, field
from typing import Any
from DoomCounter_and_auxiliaries import SleepTime

@dataclass
class VideoFeed:
    video_source             : Any
    queues_from_source       : Queue = field(default_factory = Queue)
    SLEEP_TIME               : float = 0.000001
    MAX_SOURCE_FRAMES_IN_QUEUE : int = 100  ###A WAY TO AVOID MEMORY OVERLOAD

def __call__(self):
    sleep_time = SleepTime(self.SLEEP_TIME)
    
    if not os.path.exists(self.video_source):
        print(f"Error: Video file '{self.video_source}' not found")
        return
    
    cap = cv2.VideoCapture(self.video_source)
    if not cap.isOpened():
        print(f"Error: Unable to open video '{self.video_source}'")
        return
    
    try:
        while True:
            if self.queues_from_source.qsize() < self.MAX_SOURCE_FRAMES_IN_QUEUE:
                ret, frame = cap.read()
                if not ret:
                    if cap.get(cv2.CAP_PROP_POS_FRAMES) >= cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
                        print("Video ended normally")
                    else:
                        print("Error: Failed to read frame from video")
                    break
                
                self.queues_from_source.put(frame)
                sleep_time.decrease()
            else:
                sleep_time.increase()
            
            time.sleep(sleep_time())
    except Exception as e:
        print(f"Unexpected error in video capture: {e}")
    finally:
        cap.release()