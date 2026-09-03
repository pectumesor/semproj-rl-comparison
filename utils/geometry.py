import torch
import numpy as np
import json
import os

def path_length(trajectory):
    """ Given a trajectory as a sequence of 2D points (list or array of shape (N, 2)),
     we compute the trajectory length by adding the distance of each edge between intermediate points"""

    trajectory = np.asarray(trajectory, dtype=np.float64)

    if trajectory.shape[0] < 2:
        return 0.0

    return float(np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum())
    
def cross2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """2-D cross product (scalar) on the last axis: a.x * b.y - a.y * b.x """
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def walls_json_to_numpy(json_path: str):
    walls = []
    with open(json_path) as f:
        for edge in json.load(f)["edges"]:
            walls.append(([edge["from"]["x"], edge["from"]["y"]],
                          [edge["to"]["x"],   edge["to"]["y"]]))
    return walls

def convert_hospital_json_format(input_path: str, output_path: str):

    """
     The JSON structure of hospital floor plans extracted from Arxitect are different than used 
     in this project. We convert here to the correct structure
    """

    nodes_dict = {}
    correct_format = {"edges": []}
    with open(input_path) as f:
        data = json.load(f)
        hospital_floorplan = data["wallGraph"]

        for dic in hospital_floorplan["nodes"]: # Rewrite to this dict for easy access of nodes
             nodes_dict[dic["v"]] = {
                  "x": dic["value"]["x"],
                  "y": dic["value"]["y"]
             }

        for edge in hospital_floorplan["edges"]:

            v = edge["v"]
            w = edge["w"]

            correct_format["edges"].append(

                {
                    "from": {"x": nodes_dict[v]["x"], "y": nodes_dict[v]["y"]},
                    "to": {"x": nodes_dict[w]["x"], "y": nodes_dict[w]["y"]}  
                }
            )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
         json.dump(correct_format, f, indent=2, ensure_ascii=False)


def compute_starts_and_ends(walls):
    wall_starts = np.array([p for p, _ in walls], dtype=np.float32)
    wall_ends   = np.array([q for _, q in walls], dtype=np.float32)
    return wall_starts, wall_ends


def compute_num_rays(fov, ray_density):
    num_rays = int(fov * ray_density)
    if num_rays % 2 == 0:
        num_rays += 1
    return num_rays

def w2s(pos, scale, screen_size, padding):
            """World (x, y) → pygame pixel (px, py) with Y-flip and padding."""
            return (int(float(pos[0]) * scale) + padding,
                    int(screen_size - padding - float(pos[1]) * scale))

def bounding_box(wall_ends, wall_starts):

    points = np.concatenate([wall_ends, wall_starts], axis=0)
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)

    return min_x, max_x, min_y, max_y

def diagonal_length(min_x, max_x, min_y, max_y):

    left_point = torch.tensor([min_x, min_y], dtype=torch.float32)
    right_point = torch.tensor([max_x, max_y], dtype=torch.float32)

    diagonal_vector = right_point - left_point

    return torch.linalg.norm(diagonal_vector).item()
