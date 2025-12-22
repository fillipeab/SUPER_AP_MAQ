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
        cap = cv2.VideoCapture(self.video_source)
        try:
            while True:
                if not (self.queues_from_source.qsize() >= self.MAX_SOURCE_FRAMES_IN_QUEUE): ###JUST WAIT MORE
                    ret,frame = cap.read()
                    if not ret:
                        break
                    # Adicionar resultado na fila
                    self.queues_from_source.put(frame) ### DON'T DISCART FRAMES
                    ###print("frames_in_queue:",self.video_queue.qsize(),"\n")
                    sleep_time.decrease()
                else:
                    sleep_time.increase() ###Each time it recurrently pass through here, it waits more
                time.sleep(sleep_time())
        finally:
            cap.release()