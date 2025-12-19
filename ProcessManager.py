import threading
import os
from queue import Queue
from typing import Any
from dataclasses import dataclass, field


### The objective os this class is to implement a system that does all the processing of the ID part, and the REID part. We will leave the writing part to the MemoryManager(do not confuse with the MemorySystem).
import threading
import time
import cv2
from queue import Queue
from typing import Any, ClassVar
from dataclasses import dataclass, field
from PersonDB import PersonDB
from IDSystem import IDSystem
from REIDSystem import REIDSystem
from VideoFeedManager import VideoFeedManager



@dataclass
class ProcessManager:
    ###Queues
    queues_from_sources : list = field(default_factory=list) ###Receives from VideoFeedManager
    ID_processed_queues : list = field(default_factory=list)
    Output_queues       : list = field(default_factory=list) ### element should have the following format {"frame" : frame, "model_analysis" : model_analysis, "reid_result" : list_of_temporary_person}
    
    ### pos_init variables
    number_ID_queues : int = 0
    number_output_queues : int = 0
    
    ###ID AND REID
    ID_SYSTEM = "YoloID8n"
    REID_SYSTEM = "mobileCLIP"
    
    ### WILL REID BE CENTRAL?
    SKIP_REID    : Bool = True ### True or False
    CENTRAL_REID : Bool = True
    
    ###SLEEP TIME
    SLEEP_TIME=0.000001
    
    ###Person DB
    person_db : PersonDB = field(default_factory=PersonDB)
    
    
    
    ### Methods ###
    def number_of_queues(self,queues):
        return len(queues)
    
    def create_queues(self,number : int = 1):
        queues = []
        for i in range(number): ###One queue for each source
            queues.append(Queue())
        return queues
    
    def __post_init__(self):
        self.number_ID_queues = self.number_of_queues(self.queues_from_sources)
        self.number_output_queues = self.number_of_queues(self.queues_from_sources)
        self.ID_processed_queues = self.create_queues(self.number_ID_queues)
        self.output_queues = self.create_queues(self.number_output_queues)
    
    def process_source_to_id(self,id_system,pos): ###to the correct position
        local_source_queue = self.queues_from_sources[pos]
        local_id_queue = self.ID_processed_queues[pos]
        while True: ###infinite loop
            time.sleep(self.SLEEP_TIME)
            if not local_source_queue.empty():
                frame = local_source_queue.get_nowait()
                model_analysis = id_system(frame)
                local_id_queue.put({"frame": frame, "model_analysis" : model_analysis})
    
    
    def process_ID_to_REID_central(self,reid_system): ###Central because it is one to each source. In the future, other architecture might be implemented 
        while True:
            time.sleep(self.SLEEP_TIME)
            for i in range(self.number_ID_queues): ###Check for each one of them
                local_id_queue = self.ID_processed_queues[i]
                local_output_queue = self.output_queues[i]
                if not local_id_queue.empty():
                    element = local_id_queue.get_nowait()
                    frame, model_analysis = element["frame"], element["model_analysis"]
                    list_of_temporary_person = reid_system(frame,model_analysis["temporary_persons"])
                    element["reid_result"] = list_of_temporary_person
                    local_output_queue.put(element) ### One REID queue for id queue
    
    def skip_REID_central(self):
        while True:
            time.sleep(self.SLEEP_TIME)
            for i in range(self.number_ID_queues): ###Check for each one of them
                local_id_queue = self.ID_processed_queues[i]
                local_output_queue = self.output_queues[i]
                if not local_id_queue.empty(): 
                    element = local_id_queue.get_nowait()
                    element["reid_result"] = element["model_analysis"]["temporary_persons"]
                    local_output_queue.put(element) ### Just repeats ID output
                    
    def start(self):
        # Iniciar threads
        
        ###ID threads
        for i in range(self.number_ID_queues):
            id_system = IDSystem(self.ID_SYSTEM) ###creates id system
            thread = threading.Thread(
                target=self.process_source_to_id,
                args=(id_system, i) ###args are the source, and the queue
            )
            thread.daemon = True ###Doesn't stop the program from ending
            thread.start() ###Create the thread
        
        if self.SKIP_REID == False:
            ###REID threads        
            if self.CENTRAL_REID == True:
                reid_system = REIDSystem(self.person_db,self.REID_SYSTEM)
                thread = threading.Thread(
                    target=self.process_ID_to_REID_central,
                    args=(reid_system,) ###args are the source, and the queue
                )
                thread.daemon = True ###Doesn't stop the program from ending
                thread.start() ###Create the thread
            else:
                pass ###not implemented yet
        
        
        ### Useful for testing ID without REID
        else:
            thread = threading.Thread(
            target=self.skip_REID_central,
            args=() ###args are the source, and the queue
            )
            thread.daemon = True ###Doesn't stop the program from ending
            thread.start() ###Create the thread
            
        
    def __call__(self):
        self.start()
        return self.number_output_queues, self.output_queues

###just testing the atributting of sources

if __name__ == "__main__":
    queue_index = 0
    video_sources=["auxiliares/People_in_line.mp4"]
    video_feed_manager = VideoFeedManager(video_sources=video_sources)
    n_of_sources, queues = video_feed_manager()
    process_manager = ProcessManager(queues_from_sources=queues)
    video_feed_manager.start()
    n_out_queues, output_queues = process_manager()
    
    
    ###for seeing output
    print("process manager + video feed testing")
    ###VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec MP4
    fps = 30
    width, height = 1920, 1080
    video_writer = cv2.VideoWriter(
        'output_test_video.mp4',
        fourcc,
        fps,
        (width, height)
    )
    
    doom_counter = 0
    listed_counter = 0
    try:
        while True:
            time.sleep(process_manager.SLEEP_TIME)
            if not output_queues[queue_index].empty():
                element = output_queues[queue_index].get_nowait()
                listed_counter+=1
                
                ###get processed image
                model_analysis = element["model_analysis"]
                result = model_analysis["result"] ### !!!!!!!!!!!!!!!!!!!!!!! This line is sensitive to the model type !!!!!!!!!!!!!!!!!!!!!!!!!!!
                video_writer.write(result[0].plot())
                print ("listed_counter: ",listed_counter)    
                ###get math result
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        video_writer.release()
        cv2.destroyAllWindows()
        os._exit(1)
    
