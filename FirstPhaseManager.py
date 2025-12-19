import threading
from queue import Queue
from typing import Any
from dataclasses import dataclass, field
from PersonDB import PersonDB
from MemorySystem import MemorySystem
from VideoFeedManager import VideoFeedManager




### External attributes
ext_sources = []

### In the state that the program is now, it's not needed to have a memory_system. However, it's useful to have one central entity that might alocate more DBs. This configuration makes it easier to deal with it, and even create extra DBs.
memory_system = MemorySystem() ###Creating the memory_system
memory_system() ### creating 1 PersonDB


@dataclass
class FirstPhase:
	sources : list = ext_sources

### A little fluxogram of FirstPhase
"""
VideoFeedManager(vfmanager) ----queues_from_sources[frames]---->ProcessingManager-----queues_from_processing[temp_person(s) by frame]------> queue_out_FirstPhase
Threads: 
- vfmanager(1 for each video)
- processing manager (2 for each video(1 for identifier_system, 1 for REID))
- this phase itself might benefit from using 1 specific thread

"""    
### Creating management entities
    ###VIDEO FEED
    vfmanager = VideoFeedManager(sources) ### Start reading the videosources, using multithreading. Also creates queues to export the data
    number_of_queues, queues_from_sources = videofeedmanager() ###  Each queue holds the frames that are there to be read
    ###PROCESSING
    
    







if __name__ == "__main__":
	