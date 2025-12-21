from FirstPhaseManager import FirstPhaseManager
from SecondPhaseManager import SecondPhaseManager
from dataclasses import dataclass, field
import os
import time
import cv2


### começo
def main():
    ###video parameters
    videowriter=VideoWriter(output_file='output.mp4')
    video_sources=["auxiliares/People_in_line.mp4"]
    
    ###running parameters
    queue_index         = 0  ###queue that will be watched
    SLEEP_TIME          = 0.000001
    ID_SKIP_FRAME       = 0
    REID_SKIP_FRAME     = 4
    SKIP_PERMANENCE     = 0
    SKIP_MOVEMENT       = 0
    SKIP_LINE           = 0
    
    ###Program variables
    ### phase 1 ###
    first_phase=FirstPhaseManager(
    sources = video_sources,
    SLEEP_TIME = SLEEP_TIME,
    ID_SKIP_FRAME=ID_SKIP_FRAME,
    REID_SKIP_FRAME=REID_SKIP_FRAME
    )
    number_output_queues, queues_from_sources, id_processed_queues, reid_processed_queues, first_phase_output_queues = first_phase() ###starts phase 1
    print("phase 1 - running")
    ### phase 1 - ok ###
    
    ### phase 2 ###
    second_phase = SecondPhaseManager(
    SLEEP_TIME=SLEEP_TIME,
    SKIP_PERMANENCE=SKIP_PERMANENCE,
    SKIP_MOVEMENT=SKIP_MOVEMENT,
    SKIP_LINE=SKIP_LINE,
    queues_from_first_phase = first_phase_output_queues
    )
    output_queues = second_phase()
    print("phase 2 - running")
    ### phase 2 - ok ###
    
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
                ### get output from queue
                element = output_queues[queue_index].get_nowait()
                listed_counter+=1
                
                ###gets the results that we want to see
                model_analysis = element["model_analysis"]
                return_from_watch_permanence = element["return_from_watch_permanence"]
                return_from_watch_movement = element["return_from_watch_movement"]
                result = model_analysis["result"] ### !!!!!!!!!!!!!!!!!!!!!!! This line is sensitive to the model type !!!!!!!!!!!!!!!!!!!!!!!!!!!
                
                ### LOG so we can see who was classified as moving together
                for temp_person in return_from_watch_permanence:
                    write_log(str(temp_person.id))
                for temp_person in return_from_watch_movement:
                    write_log(str(temp_person.id),"log_2.txt")
                
                ###writes the frame, altered by YOLO, in a video
                videowriter(result[0].plot())
                
                ###prints how many outputs we already have
                if listed_counter%50 == 0: ###printing takes a lot of time. Do it only for important values
                    print ("listed_counter: ",listed_counter) ### see if the process is getting to the end
            

            
            ### breaking mechanism - stops the program once all queues are empty. Keep in mind that second_phase barely uses queues ###
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
            ### breaking mechanism - END ###
            
    except Exception as e:
        print("ERRO IN MAIN LOOP:" , e)
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        cv2.destroyAllWindows()
        os._exit(1)
    
###END OF MAIN###


###SUPPORT FUNCTIONS###

###class to write videos
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

def write_list_in_log(list):
    for e in list:
        write_log(str(e))
    write_log("\n")


def write_log(text, file="log.txt"):
    with open(file, 'a') as f:
        f.write(f"{text}\n")
###END OF SUPPORT FUNCTIONS###





### To run the script ###
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)