import threading
import cv2
from queue import Queue
from typing import Any
from dataclasses import dataclass, field
from VideoFeed import VideoFeed


@dataclass
class VideoFeedManager:
    video_sources       : list = field(default_factory=list)
    queues_from_sources : list = field(default_factory=list)
    SLEEP_TIME          : float = 0.000001
    MAX_QUEUE_FRAMES    : int = 100  ###A WAY TO AVOID MEMORY OVERLOAD

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
            video_feed = VideoFeed(self.video_sources[i], self.queues_from_sources[i], SLEEP_TIME = self.SLEEP_TIME, MAX_QUEUE_FRAMES = self.MAX_QUEUE_FRAMES)
            thread = threading.Thread(
                target=video_feed ###args are the source, and the queue
            )
            thread.daemon = True ###Doesn't stop the program from ending
            thread.start() ###Create the thread
    
    def __call__(self):
        self.start()
        return self.number_of_queues, self.queues_from_sources

###just testing the atributting of sources

if __name__ == "__main__":
    print("Testing fuctioning")
    video_sources_2=[1,2]
    Test_manager = VideoFeedManager()
    Test_manager2 = VideoFeedManager(video_sources_2)
    print(Test_manager.video_sources[0])
    print(Test_manager2.video_sources[0],Test_manager2.video_sources[1])
