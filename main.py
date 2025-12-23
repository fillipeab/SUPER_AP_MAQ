from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
import cv2

import os
import gc
import threading
import time

from FirstPhaseManager  import FirstPhaseManager
from SecondPhaseManager import SecondPhaseManager
from ThirdPhaseManager  import ThirdPhaseManager
from DoomCounter_and_auxiliaries  import DoomCounter

### começo
def main():

    ### PROGRAM VARIABLES ###
    ###video parameters
    video_sources=["auxiliares/video1.mp4","auxiliares/People_in_line_2.mp4","auxiliares/video2.mp4"] ###
    list_passing_parameters_dicts = []
    MAX_SOURCE_FRAMES_IN_QUEUE = 100  ###A WAY TO AVOID MEMORY OVERLOAD

    ###THREADING PARAMETERS##
    SLEEP_TIME          = 0.000_001
    QUEUE_MAXIMUM_SIZE  = 45


    ID_SKIP_FRAME       = 0
    REID_SKIP_FRAME     = 0
    SKIP_PERMANENCE     = 0
    SKIP_MOVEMENT       = 0
    SKIP_LINE           = 0

    SKIP_REID           = False  ### True or False
    CENTRAL_REID        = True
    
    ###DEBUG###
    show_monitor_thread = True
    ### PROGRAM VARIABLES - END ###
    
    try:
        ### phase 1 ###
        first_phase=FirstPhaseManager(
        video_sources = video_sources,
        list_passing_parameters_dicts = list_passing_parameters_dicts,
        MAX_SOURCE_FRAMES_IN_QUEUE = MAX_SOURCE_FRAMES_IN_QUEUE,
        ID_SKIP_FRAME=ID_SKIP_FRAME,
        REID_SKIP_FRAME=REID_SKIP_FRAME,
        SKIP_REID    = SKIP_REID,
        CENTRAL_REID = CENTRAL_REID,
        SLEEP_TIME = SLEEP_TIME,
        QUEUE_MAXIMUM_SIZE=QUEUE_MAXIMUM_SIZE
        )
        _, queues_from_sources, id_processed_queues, reid_processed_queues, first_phase_output_queues = first_phase()
        print("phase 1 - running")
        ### phase 1 - ok ###
        
        ### phase 2 ###
        second_phase = SecondPhaseManager(
        SLEEP_TIME=SLEEP_TIME,
        SKIP_PERMANENCE=SKIP_PERMANENCE,
        SKIP_MOVEMENT=SKIP_MOVEMENT,
        SKIP_LINE=SKIP_LINE,
        queues_from_first_phase = first_phase_output_queues,
        QUEUE_MAXIMUM_SIZE=QUEUE_MAXIMUM_SIZE
        )
        output_queues = second_phase()
        print("phase 2 - running")
        ### phase 2 - ok ###

        ### phase 3 ###
        third_phase = ThirdPhaseManager(output_queues = output_queues, list_passing_parameters_dicts=list_passing_parameters_dicts)
        third_phase()
        print("phase 3 - running")
        ### phase 3 - ok ###

        ### doom counter ### Inside of try - makes sure everything goes smoothly
        ###DEBUGERS AND COUNTERS###
        doom_counter = DoomCounter(
        print_queue_stats = True,
        print_counting  = False,
        queues_to_check=[q[0] for q in [queues_from_sources,id_processed_queues,reid_processed_queues,first_phase_output_queues,output_queues]]
        )
       
        if show_monitor_thread == True:
            monitor_thread = threading.Thread(target=monitor_threads, daemon=True)
            monitor_thread.start()
        doom_counter()
        ###
        """while True:
            time.sleep(SLEEP_TIME*100_000_000)
            try:
                if not output_queues[0].empty():
                    print("NAO ESTA VAZIA")
            except:
                print("OH NO")   """
        ###
        ###DEBUGERS AND COUNTERS - END###
             
    except Exception as e:
        print("ERRO IN MAIN LOOP:" , e)
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        third_phase.end()
        cv2.destroyAllWindows()
        os._exit(1)
    

def monitor_threads(interval=12_000):
    """Run after 120s, with it follows the expected"""
    while True:
        print(f"\n\n||======= Threads ativas: {threading.active_count()} =======||")
        
        for thread in threading.enumerate():
            print(f"  {thread.name}: {'Alive' if thread.is_alive() else 'Dead'} - Daemon: {thread.daemon}")
        print(f"||=================================||")
        time.sleep(interval)



###END OF MAIN###

### To run the script ###
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
