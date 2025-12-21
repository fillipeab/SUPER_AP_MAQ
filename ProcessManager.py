import os
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
    """
    queues_from_sources =>  process_source_to_id(ID_processor(s) (one for each)) => ID_processed_queues
    ID_processed_queues => process_ID_to_REID_central(1 TO all) => Output_queues ###Were previoulsy called REID_queues.
    
    """
    ###Queues
    queues_from_sources   : list = field(default_factory=list) ###Receives from VideoFeedManager
    ID_processed_queues   : list = field(default_factory=list) ###Internal
    REID_processed_queues : list = field(default_factory=list) ###Internal
    Output_queues         : list = field(default_factory=list) ###External
    ### element in output queue should have the following format {"frame" : frame, "model_analysis" : model_analysis, "reid_result" : list_of_temporary_person}
    
    ### pos_init variables
    number_ID_queues : int = 0
    number_output_queues : int = 0
    
    ###SKIP process?###
    ID_COUNTER      = 0
    REID_COUNTER    = 0
    ID_SKIP_FRAME   : int = 0 ### Number of frames that will be skipped - CANNOT BE NEGATIVE
    REID_SKIP_FRAME : int = 4 ### Number of frames that will be skipped - CANNOT BE NEGATIVE
    
    ###ID AND REID
    ID_SYSTEM = "YoloID8n"
    REID_SYSTEM = "mobileCLIP"
    
    ### WILL REID BE CENTRAL?
    SKIP_REID    : bool = True ### True or False
    CENTRAL_REID : bool = True
    
    ###SLEEP TIME
    SLEEP_TIME   : float = 0.000001
    
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
        self.number_REID_queues = self.number_of_queues(self.queues_from_sources)
        self.number_output_queues = self.number_of_queues(self.queues_from_sources)
        self.ID_processed_queues = self.create_queues(self.number_ID_queues)
        self.REID_processed_queues = self.create_queues(self.number_REID_queues)
        self.output_queues = self.create_queues(self.number_output_queues)
        if self.ID_SKIP_FRAME < 0:
            self.ID_SKIP_FRAME=0
        if self.REID_SKIP_FRAME < 0:
            self.REID_SKIP_FRAME=0
    
    def process_source_to_id(self,id_system,pos): ###to the correct position
        local_source_queue = self.queues_from_sources[pos]
        local_id_queue = self.ID_processed_queues[pos]
        while True: ###infinite loop
            time.sleep(self.SLEEP_TIME)
            if not local_source_queue.empty():
                frame = local_source_queue.get_nowait()
                if self.ID_COUNTER % (self.ID_SKIP_FRAME+1) == 0:
                    model_analysis = id_system(frame)
                    local_id_queue.put({"frame": frame, "model_analysis" : model_analysis})
                    self.ID_COUNTER=0
                self.ID_COUNTER+=1
    
    
    def process_ID_to_REID_central(self,reid_system): ###Central because it is one to each source. In the future, other architecture might be implemented 
        while True:
            time.sleep(self.SLEEP_TIME)
            for i in range(self.number_ID_queues): ###Check for each one of them
                local_id_queue = self.ID_processed_queues[i]
                local_reid_queue = self.REID_processed_queues[i]
                if not local_id_queue.empty():
                    element = local_id_queue.get_nowait()
                    if self.REID_COUNTER % (self.REID_SKIP_FRAME+1) == 0:
                        frame, model_analysis = element["frame"], element["model_analysis"]
                        list_of_temporary_person = reid_system(frame,model_analysis["temporary_people"])
                        element["reid_result"] = list_of_temporary_person
                        local_reid_queue.put(element) ### One REID queue for id queue
                        self.REID_COUNTER=0
                    self.REID_COUNTER+=1
    
    def skip_REID_central(self,reid_system):
        while True:
            time.sleep(self.SLEEP_TIME)
            for i in range(self.number_ID_queues): ###Check for each one of them
                local_id_queue = self.ID_processed_queues[i]
                local_reid_queue = self.REID_processed_queues[i]
                if not local_id_queue.empty(): 
                    element = local_id_queue.get_nowait()
                    if self.REID_COUNTER % (self.REID_SKIP_FRAME+1) == 0:
                        element["reid_result"] = element["model_analysis"]["temporary_people"]
                        local_reid_queue.put(element) ### Just repeats ID output
                        self.REID_COUNTER=0
                    self.REID_COUNTER+=1
                    
    def REID_to_Output(self):
        ###Initially, there is no need for any conversion, so it's completelly possible to just leave a simple atribution operation. In the future, more operations might be required
        self.output_queues=self.REID_processed_queues
                    
    def start(self):
        # Iniciar threads
        
        ### ID ###
        for i in range(self.number_ID_queues):
            id_system = IDSystem(self.ID_SYSTEM) ###creates id system
            thread = threading.Thread(
                target=self.process_source_to_id,
                args=(id_system, i) ###args are the source, and the queue
            )
            thread.daemon = True ###Doesn't stop the program from ending
            thread.start() ###Create the thread
        ### End of ID ###
        
        ### REID ###
        
        ###chosing reid_system
        reid_system = ""
        thread_target = 0
        if self.SKIP_REID == False:
            if self.CENTRAL_REID == True:
                thread_target = self.process_ID_to_REID_central
                reid_system = REIDSystem(self.person_db,self.REID_SYSTEM)
            else:
                thread_target=self.skip_REID_central
                pass ###not implemented yet
        else:
            thread_target=self.skip_REID_central
        ###end###   
        
        ###threading
        thread = threading.Thread(
        target=thread_target,
        args=(reid_system,) ###args are the source, and the queue
        )
        thread.daemon = True ###Doesn't stop the program from ending
        thread.start() ###Create the thread
        ### End of REID ###
        
        ### REID -> OUTPUT ###
        self.REID_to_Output()
        
        ###end###

    ###calling function
    """Returns : self.number_output_queues, self.queues_from_sources, self.ID_processed_queues, self.REID_processed_queues, self.output_queues"""
    def __call__(self):
        self.start()
        return self.number_output_queues, self.queues_from_sources, self.ID_processed_queues, self.REID_processed_queues, self.output_queues

###just testing the atributting of sources

if __name__ == "__main__":
    queue_index = 0
    ### video sources
    video_sources=["auxiliares/People_in_line.mp4"]
    video_feed_manager = VideoFeedManager(video_sources=video_sources)
    n_of_sources, queues = video_feed_manager()
    
    ### Process manager
    process_manager = ProcessManager(queues_from_sources=queues)
    
    ### START ###
    video_feed_manager()
    n_out_queues, queues_from_sources, id_processed_queues, reid_processed_queues, output_queues = process_manager()
    
    
    ### VIDEO SETTINGS ###
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
    ### VIDEO SETTINGS - END ###
    
    ### Main loop
    doom_counter = 0
    doom_flag = 0
    listed_counter = 0
    waiting_multiplier_normal = 30000 ###static
    waiting_multiplier = waiting_multiplier_normal
    try:
        while True:
            time.sleep(process_manager.SLEEP_TIME*waiting_multiplier)
            if not output_queues[queue_index].empty():
                element = output_queues[queue_index].get_nowait()
                listed_counter+=1
                ###get processed image
                model_analysis = element["model_analysis"]
                result = model_analysis["result"] ### !!!!!!!!!!!!!!!!!!!!!!! This line is sensitive to the model type !!!!!!!!!!!!!!!!!!!!!!!!!!!
                video_writer.write(result[0].plot())
                if listed_counter%50 == 0: ###printing takes a lot of time. Do it only for important values
                    print ("listed_counter: ",listed_counter) ### see if the process is getting to the end
                
            ### breaking mechanism ###
            if doom_counter%5 == 0: ###printing takes a lot of time. Do it only for important values
                    print (doom_counter) ### see if the process is getting to the end
            if doom_counter == 1000:
                try: ###checking for empty queues
                    if (queues_from_sources[queue_index].empty() and id_processed_queues[queue_index].empty() and
                    reid_processed_queues[queue_index].empty() and output_queues[queue_index].empty()):
                        doom_flag+=1
                        if doom_flag == 1:
                            print("\nDoom is aproaching\n")
                        elif doom_flag == 2:
                            print("\nDoom is INEVITABLE!\n")
                        elif doom_flag == 3:
                            print("\nDOOOOOOOOOOM!!!\n")
                        waiting_multiplier = 1
                    elif (queues_from_sources[queue_index].empty() and id_processed_queues[queue_index].empty() and
                    reid_processed_queues[queue_index].empty() and not output_queues[queue_index].empty()): ###which menas that there's only output to process
                        waiting_multiplier = 1
                    else:
                        doom_flag=0
                        waiting_multiplier = waiting_multiplier_normal
                except Exception as e:
                    print("Erro: ",e)
                    
                doom_counter = 0 ### reset doom counter
            doom_counter+=1
            
            if doom_flag>=3 :
                break
            
            ### breaking mechanism - end ###
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        video_writer.release()
        cv2.destroyAllWindows()
        os._exit(1)
    
