import threading
import cv2
from queue import Queue
from dataclasses import dataclass, field
from typing import Any

@dataclass
class VideoFeed(source,queue):
    source=source
    queue=queue
    
    def __call__(self,source=self.source,queue=self.queue):
        cap = cv2.VideoCapture(source)
        while True:
            ret,frame = cap.read()
            if not ret:
                break
            # Adicionar resultado na fila
            queue.put({
                frame
            })
        cap.release()