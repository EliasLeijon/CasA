import open3d
from typing import List

"""

|
y
|___x__
"""
ALPHABET = {
    "A":  {
            "points": [
                [0, 0, 0],
                [0.25, 0.5, 0],
                [0.5, 1, 0],
                [0.75, 0.5, 0],
                [1,0,0],
            ],
            "lines": [
                [0, 2],
                [2, 4],
                [1, 3],
            ],
        },
    "C":  {
            "points": [
                [0.7, 1, 0],
                [0, 1, 0],
                [0, 0, 0],
                [0.7, 0, 0],
            ],
            "lines": [
                [0, 1],
                [1, 2],
                [2, 3],
            ],
        },

}
def _get_letter_lines(letter: str, x: int, y: int, z: int,  scale: int = 1):
    return open3d.geometry.LineSet(
        points = open3d.utility.Vector3dVector(_scale_and_translate_points(ALPHABET[letter]["points"], scale=scale, translation=[x, y, z])),
        lines = open3d.utility.Vector2iVector(ALPHABET[letter]["lines"])
    )

def _scale_and_translate_points(points: List[List[int]], scale: int = 1, translation=List[int]):
    return [[coord*scale+translation[coord_index] for coord_index, coord in enumerate(point)] for point in points]