from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
import os
import time
import cv2
import gc
from FirstPhaseManager  import FirstPhaseManager
from SecondPhaseManager import SecondPhaseManager
from ThirdPhaseManager  import ThirdPhaseManager
from DoomCounter        import DoomCounter

### começo
def main():
    ###video parameters
    video_sources=["auxiliares/People_in_line_2.mp4"]
    MAX_SOURCE_FRAMES_IN_QUEUE = 100  ###A WAY TO AVOID MEMORY OVERLOAD

    ###THREADING PARAMETERS##
    SLEEP_TIME          = 0.000001
    QUEUE_MAXIMUM_SIZE  = 25


    ID_SKIP_FRAME       = 0
    REID_SKIP_FRAME     = 5
    SKIP_PERMANENCE     = 20
    SKIP_MOVEMENT       = 0
    SKIP_LINE           = 0
    
    ###Program variables
    ### phase 1 ###
    first_phase=FirstPhaseManager(
    video_sources = video_sources,
    SLEEP_TIME = SLEEP_TIME,
    ID_SKIP_FRAME=ID_SKIP_FRAME,
    REID_SKIP_FRAME=REID_SKIP_FRAME,
    QUEUE_MAXIMUM_SIZE=QUEUE_MAXIMUM_SIZE,
    MAX_SOURCE_FRAMES_IN_QUEUE = MAX_SOURCE_FRAMES_IN_QUEUE
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
    queues_from_first_phase = first_phase_output_queues,
    QUEUE_MAXIMUM_SIZE=QUEUE_MAXIMUM_SIZE
    )
    output_queues = second_phase()
    print("phase 2 - running")
    ### phase 2 - ok ###

    ###phase 3 ###
    third_phase = ThirdPhaseManager(output_queues = output_queues)
    third_phase()
    print("phase 3 - running")
    ###phase 3 - ok ###

    ### doom counter ### Inside of try - makes sure everything goes smoothly
    try:
        doom_counter = DoomCounter(
        queues_to_check=[q[0] for q in [queues_from_sources,id_processed_queues,reid_processed_queues,first_phase_output_queues,output_queues]]
        )
        doom_counter()
    except Exception as e:
        print("ERRO IN MAIN LOOP:" , e)
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        cv2.destroyAllWindows()
        os._exit(1)
    
###END OF MAIN###

### To run the script ###
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)