import time
import cv2
from queue import Queue
from dataclasses import dataclass, field
from typing import Any

@dataclass
class VideoFeed:
    source           : Any
    video_queue      : Queue = field(default_factory = Queue)
    SLEEP_TIME       : float = 0.000001
    MAX_QUEUE_FRAMES : int = 100  ###A WAY TO AVOID MEMORY OVERLOAD

    def __call__(self):
        sleep_time = 0
        cap = cv2.VideoCapture(self.source)
        while True:
            ret,frame = cap.read()
            if not ret:
                break
            # Adicionar resultado na fila
            self.video_queue.put(frame) ### DON'T DISCART FRAMES
            if not (self.video_queue.qsize() > self.MAX_QUEUE_FRAMES): ###JUST WAIT MORE
                sleep_time += self.SLEEP_TIME ###Each time it recurrently pass through here, it waits more
            ###print("frames_in_queue:",self.video_queue.qsize(),"\n")
            else:
                sleep_time = 0
            time.sleep(sleep_time)
        cap.release()