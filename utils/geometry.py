import torch
import numpy as np
import json
import os

class InitialPositionSampler:
    """
    Samples initial agent positions inside the room.

    Each spawn region in the room JSON is a list of edges describing an axis-aligned rectangle.
    Every sample picks a region uniformly at random per env and then a uniform point inside it.
    An empty list means the room is a single closed area, so the bounding box (shrunk by
    wall_offset) is used instead.

    Points closer than wall_offset to any wall are resampled: rooms without spawn regions
    (e.g. medium_maze, hard_maze) have interior walls inside their bounding box, and a spawn
    on top of a wall line lets the movement ray start past it.
    """
    MAX_RESAMPLES = 50

    def __init__(self, json_path, wall_offset=0.0, device="cpu"):
        self.device = device
        self.wall_offset = wall_offset
        self.spawn_regions = get_spawn_regions_json(json_path)
        self.N = len(self.spawn_regions)
        wall_starts, wall_ends = compute_starts_and_ends(walls_json_to_numpy(json_path))
        self.bounding_box = bounding_box(wall_starts, wall_ends)
        self.wall_starts = torch.tensor(wall_starts, device=device)
        self.wall_ends   = torch.tensor(wall_ends, device=device)

        if self.N == 0:
            min_x, max_x, min_y, max_y = self.bounding_box
            boxes = [[min_x + wall_offset, max_x - wall_offset, min_y + wall_offset, max_y - wall_offset]]
        else:
            boxes = [region_to_box(region) for region in self.spawn_regions]

        # (num_boxes, 4) as [min_x, max_x, min_y, max_y]
        self.boxes = torch.tensor(np.asarray(boxes, dtype=np.float32), device=device)

    def sample_spawn_region(self, num_envs):
        """Returns one box per env, shape (num_envs, 4)."""
        if self.N == 0:
            return self.boxes.expand(num_envs, -1)
        idx = torch.randint(0, self.N, (num_envs,), device=self.device)
        return self.boxes[idx]

    def _sample_once(self, num_envs):
        box = self.sample_spawn_region(num_envs)
        u = torch.rand((num_envs, 2), device=self.device)
        x = box[:, 0] + u[:, 0] * (box[:, 1] - box[:, 0])
        y = box[:, 2] + u[:, 1] * (box[:, 3] - box[:, 2])
        return torch.stack([x, y], dim=-1)

    def too_close_to_wall(self, pos):
        """(N, 2) positions -> (N,) bool, True if within wall_offset of a wall segment."""
        if self.wall_offset <= 0:
            return torch.zeros(pos.shape[0], dtype=torch.bool, device=pos.device)
        return point_segment_distance(pos, self.wall_starts, self.wall_ends).min(dim=-1).values < self.wall_offset

    def sample(self, num_envs):
        """Returns uniformly sampled positions inside freshly sampled spawn regions, shape (num_envs, 2)."""
        pos = self._sample_once(num_envs)
        for _ in range(self.MAX_RESAMPLES):
            bad = self.too_close_to_wall(pos)
            if not bad.any():
                break
            pos[bad] = self._sample_once(int(bad.sum()))
        return pos


def point_segment_distance(points: torch.Tensor, seg_starts: torch.Tensor, seg_ends: torch.Tensor):
    """Euclidean distance of every point (N, 2) to every segment (W, 2)->(W, 2). Returns (N, W)."""
    seg = seg_ends - seg_starts                                                     # (W, 2)
    rel = points[:, None, :] - seg_starts[None, :, :]                               # (N, W, 2)
    seg_len_sq = (seg ** 2).sum(dim=-1).clamp_min(1e-8)                             # (W,)
    s = ((rel * seg[None]).sum(dim=-1) / seg_len_sq).clamp(0.0, 1.0)               # (N, W)
    closest = seg_starts[None] + s[..., None] * seg[None]                           # (N, W, 2)
    return torch.linalg.norm(points[:, None, :] - closest, dim=-1)


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

def start_goal_pos_from_json(json_path: str):
    with open(json_path) as f:
        data = json.load(f)

    start_pos = [data["start_pos"]["x"], data["start_pos"]["y"]]
    end_pos = [data["goal_pos"]["x"], data["goal_pos"]["y"]]

    return start_pos, end_pos

def get_spawn_regions_json(json_path: str):
    with open(json_path) as f:
        data = json.load(f)

    return data.get("spawn_regions", [])

def wall_attributes_json(json_path: str, default_colour=(0.5, 0.5, 0.5)):
    """
    Optional per-edge attributes of the room JSON:
        "colour": [r, g, b] in [0, 1]  -> used by colour_mode=per_wall (default grey)
        "cue": true                    -> cue wall, coloured with the episode's cue colour in every colour mode
    Returns colours (W, 3) and cue flags (W,), in the same order as walls_json_to_numpy.
    """
    with open(json_path) as f:
        edges = json.load(f)["edges"]
    colours = np.array([edge.get("colour", default_colour) for edge in edges], dtype=np.float32)
    cue = np.array([bool(edge.get("cue", False)) for edge in edges], dtype=bool)
    return colours, cue

def goal_regions_json(json_path: str):
    """
    Goal layout of rooms with several candidate goals (e.g. the twin-room cue task):
        "goal_points":  [{"x", "y"}, ...]    one goal per candidate
        "goal_regions": [region, ...]        area belonging to each candidate (spawn-region format);
                                             entering a region that isn't the episode's goal ends the episode
        "cue_colours":  [[r, g, b], ...]     colour of the cue walls when that candidate holds the goal
    Returns (points (K, 2), boxes (K, 4) or None, cue colours (K, 3) or None), or None if the room has no goal_points.
    """
    with open(json_path) as f:
        data = json.load(f)
    if "goal_points" not in data:
        return None

    points = np.array([[p["x"], p["y"]] for p in data["goal_points"]], dtype=np.float32)
    boxes = (np.array([region_to_box(r) for r in data["goal_regions"]], dtype=np.float32)
             if "goal_regions" in data else None)
    cue_colours = np.array(data["cue_colours"], dtype=np.float32) if "cue_colours" in data else None
    return points, boxes, cue_colours

def junction_region_json(json_path: str):
    """Box [min_x, max_x, min_y, max_y] of the room's optional "junction_region" (twin rooms), or None."""
    with open(json_path) as f:
        data = json.load(f)
    return region_to_box(data["junction_region"]) if "junction_region" in data else None

def junction_path_json(json_path: str):
    """(P, 2) waypoints of the room's optional "junction_path" (twin rooms: alcove -> bend -> junction), or None."""
    with open(json_path) as f:
        data = json.load(f)
    if "junction_path" not in data:
        return None
    return np.array([[point["x"], point["y"]] for point in data["junction_path"]], dtype=np.float32)

def region_to_box(region):
    """Axis-aligned box [min_x, max_x, min_y, max_y] spanned by the edges of a spawn region."""
    points = np.array([[edge[k]["x"], edge[k]["y"]] for edge in region for k in ("from", "to")],
                      dtype=np.float32)
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    return [min_x, max_x, min_y, max_y]

    
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
