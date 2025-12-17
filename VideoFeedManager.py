import threading
import cv2
from queue import Queue
from typing import Any
from dataclasses import dataclass, field
from VideoFeed import VideoFeed

### Video sources
video_sources = [] #(Source)

def create_queues(number_of_queues : int = 1):
    queues_from_sources = []
    for i in range(number_of_queues): ###One queue for each source
        queues_from_sources.append(Queue())
    return queues_from_sources

@dataclass
class VideoFeedManager:
    video_sources = list = video_sources
    number_of_queues = self.video_sources.len()
    queues_from_sources = create_queues(number_of_queues)
        
    def start(self):
        # Iniciar threads
        for i in range(self.video_sources.len()): ###for each camera in the list
            thread = threading.Thread(
                target=VideoFeed,
                args=(self.video_sources[i], self.queues_from_sources[i]) ###args are the source, and the queue
            )
            thread.daemon = True ###Doesn't stop the program from ending
            thread.start() ###Create the thread
    
    __call__(self):
        self.start()
        return number_of_queues, queues_from_sources

###just testing

if __name__ == "__main__":
    video_sources_2=[1,2]
    Test_manager = VideoFeedManager
    Test_manager2 = VideoFeedManager(video_sources_2)
