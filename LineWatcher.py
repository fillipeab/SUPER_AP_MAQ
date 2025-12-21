###LINE WATCHER - THE END OF OUR JORNEY
from dataclasses import dataclass, field
from typing import Any
from scipy.spatial import cKDTree
import numpy as np

@dataclass
class LineWatcher():
    ### receives the list of people


def find_nearest_neighbors(list_of_temporary_people, n=2):
    """Returns the n nearest BBoxes for each BBox."""
    # Calculate BBox centers
    centers = np.array([
        [(x1 + x2) / 2, (y1 + y2) / 2]
        for person in list_of_temporary_people
        for x1, y1, x2, y2 in [person.bbox]
    ])
    
    # Build KDTree for fast nearest neighbor search
    tree = cKDTree(centers)
    _, indices = tree.query(centers, k=n+1)  # +1 to include self
    
    # Remove self from each list
    return {i: idxs[1:].tolist() for i, idxs in enumerate(indices)}