from dataclasses import dataclass, field
from PersonDB import PersonDB
from MemorySystem import MemorySystem
from VideoFeedManager import VideoFeedManager
from FirstProcessManager import FirstProcessManager
import os
import time
import cv2



### In the state that the program is now, it's not needed to have a memory_system. However, it's useful to have one central entity that might alocate more DBs. This configuration makes it easier to deal with it, and even create extra DBs.

@dataclass
class FirstPhaseManager:
    ### VIDEO _ CONFIGS ###
    video_sources                        : list = field(default_factory=list)
    list_passing_parameters_dicts        : list = field(default_factory=list)
    MAX_SOURCE_FRAMES_IN_QUEUE           : int = 100  ###A WAY TO AVOID MEMORY OVERLOAD
    ### VIDEO _ CONFIGS - END ###

    ### PROCESSING CONFIGS ###
    ID_SKIP_FRAME         : int = 0 ###SKIPPING THIS PROCESS WILL SHORTEN THE VIDEO: THE FRAMES FROM HERE ARE THE SOURCE OF THE OUTPUT VIDEO
    REID_SKIP_FRAME       : int = 10
    ### PROCESSING CONFIGS - END ###

    ### WILL REID BE CENTRAL? ###
    SKIP_REID    : bool = True ### True or False
    CENTRAL_REID : bool = True
    ### WILL REID BE CENTRAL? - END ###

    ### THREADING CONFIGS ###
    SLEEP_TIME            : float = 0.000001
    QUEUE_MAXIMUM_SIZE    : int = 25 ### FOR PROCESS
    ### THREADING CONFIGS - END ###


    ### NOT ADVISABLE TO MESS WITH THE VARIABLES BELOW ###
    ### Objects for operation ###
    queues_from_sources   : list = field(default_factory=list)
    output_queues         : list = field(default_factory=list)     
    ### element in output queue should have the following format {"frame" : frame, "model_analysis" : model_analysis, "reid_result" : list_of_temporary_person}
    video_feed_manager    : VideoFeedManager    = field(default_factory=VideoFeedManager)
    first_process_manager : FirstProcessManager = field(default_factory=FirstProcessManager)
    
    def __call__(self):
        ### VIDEO_FEED_MANAGER ###
        video_feed_manager=VideoFeedManager(video_sources = self.video_sources, list_passing_parameters_dicts = self.list_passing_parameters_dicts, MAX_SOURCE_FRAMES_IN_QUEUE = self.MAX_SOURCE_FRAMES_IN_QUEUE, SLEEP_TIME = self.SLEEP_TIME)
        _, self.queues_from_sources = video_feed_manager() ###Starts video_feed_manager
        ### VIDEO_FEED_MANAGER - END ###

        ### FIRST_PROCESS_MANAGER ###
        first_process_manager = FirstProcessManager(
        queues_from_sources = self.queues_from_sources,
        ID_SKIP_FRAME = self.ID_SKIP_FRAME,
        REID_SKIP_FRAME = self.REID_SKIP_FRAME,
        SLEEP_TIME=self.SLEEP_TIME,
        QUEUE_MAXIMUM_SIZE = self.QUEUE_MAXIMUM_SIZE,
        SKIP_REID = self.SKIP_REID,
        CENTRAL_REID = self.CENTRAL_REID
        )
        ###Start first_process_manager
        ### element in output queue should have the following format {"frame" : frame, "model_analysis" : model_analysis, "reid_result" : list_of_temporary_person}
        number_output_queues, queues_from_sources, ID_processed_queues, REID_processed_queues, self.output_queues = first_process_manager()  
        ### FIRST_PROCESS_MANAGER - END ###

        return number_output_queues, queues_from_sources, ID_processed_queues, REID_processed_queues, self.output_queues
        



### TESTING FirstPhaseManager ###
if __name__ == "__main__":
    queue_index = 0
    video_sources = ["auxiliares/People_in_line.mp4"]
    first_phase_manager=FirstPhaseManager(video_sources = video_sources)
    SLEEP_TIME = first_phase_manager.SLEEP_TIME
    number_output_queues, queues_from_sources, id_processed_queues, reid_processed_queues, output_queues = first_phase_manager()
    
    
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
            time.sleep(SLEEP_TIME*waiting_multiplier)
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
            if doom_counter%50 == 0: ###printing takes a lot of time. Do it only for important values
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
    except Exception as e:
        print(e)
    finally:
        video_writer.release()
        cv2.destroyAllWindows()
        os._exit(1)