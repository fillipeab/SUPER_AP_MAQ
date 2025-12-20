import threading
from queue import Queue
from typing import Any
from dataclasses import dataclass, field
from PersonDB import PersonDB
from MemorySystem import MemorySystem
from VideoFeedManager import VideoFeedManager
from ProcessManager import ProcessManager




### In the state that the program is now, it's not needed to have a memory_system. However, it's useful to have one central entity that might alocate more DBs. This configuration makes it easier to deal with it, and even create extra DBs.

@dataclass
class FirstPhase:
	sources             : list = field(default_factory=list)
    queues_from_sources : list = field(default_factory=list)
    output_queues       : list = field(default_factory=list)
    video_feed_manager  : VideoFeedManager = None
    process_manager     : ProcessManager   = None
    
    def __call__(self):
        video_feed_manager=VideoFeedManager(sources)
        _, self.queues_from_sources = video_feed_manager() ###Starts video_feed_manager
        process_manager = ProcessManager(self.queues_from_sources)
        _, _, _, _, self.output_queues = process_manager() #number_output_queues, queues_from_sources, ID_processed_queues, REID_processed_queues, output_queues ###Start process_manager
        
    
    
    







if __name__ == "__main__":
	