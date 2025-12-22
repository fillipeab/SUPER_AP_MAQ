import threading
import cv2
import os
from queue import Queue
from typing import Any
from dataclasses import dataclass, field
from VideoFeed import VideoFeed


@dataclass
class VideoFeedManager:
    video_sources       : list = field(default_factory=list)
    queues_from_sources : list = field(default_factory=list)
    SLEEP_TIME          : float = 0.000001
    MAX_SOURCE_FRAMES_IN_QUEUE    : int = 100  ###A WAY TO AVOID MEMORY OVERLOAD

    @property
    def number_of_queues(self):
        return len(self.video_sources)
    
    @staticmethod
    def create_queues(number_of_queues : int = 1):
        queues_from_sources = []
        for i in range(number_of_queues): ###One queue for each source
            queues_from_sources.append(Queue())
        return queues_from_sources
    
    def __post_init__(self):
        self.queues_from_sources = self.create_queues(self.number_of_queues)
    
    def start(self):
        # Iniciar threads
        for i in range(self.number_of_queues): ###for each camera in the list
            video_feed = VideoFeed(self.video_sources[i], self.queues_from_sources[i], SLEEP_TIME = self.SLEEP_TIME, MAX_SOURCE_FRAMES_IN_QUEUE = self.MAX_SOURCE_FRAMES_IN_QUEUE)
            thread = threading.Thread(
                target=video_feed, ###args are the source, and the queue
                args=()
            )
            thread.daemon = True ###Doesn't stop the program from ending
            thread.start() ###Create the thread
    
    def __call__(self):
        self.start()
        return self.number_of_queues, self.queues_from_sources




### TESTING ###
@dataclass
class VideoWriter:
    output_file: str = 'output.mp4'
    fps: int = 30
    width: int = 1920
    height: int = 1080
    
    def __post_init__(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.output_file, 
            fourcc, 
            self.fps, 
            (self.width, self.height)
        )
    
    def __call__(self, frame):
        self.writer.write(frame)
    
    def __del__(self):
        if hasattr(self, 'writer'):
            self.writer.release()






if __name__ == "__main__":
    videowriter=VideoWriter(output_file='output.mp4')
    video_sources=["auxiliares/People_in_line_2.mp4"]
    videofeedmanager = VideoFeedManager(video_sources)
    _ , queue_from_sources = videofeedmanager()
    queue=queue_from_sources[0]
    try:
        counter = 0
        while True:
            if not queue.empty():
                element = queue.get_nowait()
                videowriter(element) ###just writes
                counter+=1
                if counter % 30 == 0:
                    print(counter)
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        del videowriter
        cv2.destroyAllWindows()
        os._exit(1)