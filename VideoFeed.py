import threading
import cv2
from queue import Queue
from dataclasses import dataclass, field
from typing import Any

@dataclass
class VideoFeed:
    source : Any
    video_queue : Queue = field(default_factory = Queue)
    MAX_QUEUE_FRAMES : int = 100  ###A WAY TO AVOID MEMORY OVERLOAD
    
    def __call__(self):
        cap = cv2.VideoCapture(self.source)
        while True:
            ret,frame = cap.read()
            if not ret:
                break
            # Adicionar resultado na fila
            if not (self.video_queue.qsize() > self.MAX_QUEUE_FRAMES):
                self.video_queue.put(frame)
            ###print("frames_in_queue:",self.video_queue.qsize(),"\n")
        cap.release()