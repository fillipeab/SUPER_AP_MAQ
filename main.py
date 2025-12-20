from FirstPhaseManager import FirstPhaseManager
from SecondPhaseManager import SecondPhaseManager
from dataclasses import dataclass, field


### começo
def main():
    videowriter=VideoWriter(output_file='output.mp4')
    video_sources=["auxiliares/People_in_line.mp4"]
    first_phase=FirstPhase(sources = video_sources)
    number_output_queues, queues_from_sources, ID_processed_queues, REID_processed_queues, output_queues = first_phase()









    except KeyboardInterrupt:
        print("interrupted")
    finally:
        cv2.destroyAllWindows()
        os._exit(1)


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

