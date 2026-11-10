"""
Generates the twin-room cue task (experiment E2) as room JSONs: rooms/twin_rooms_L<L>.json

    cue alcove
        │ shaft
        └──── corridor (L steps long) ──── T-junction ─┬─ arm ── room A (goal if cue = colour 0)
                                                       └─ arm ── room B (goal if cue = colour 1)

- The alcove walls are cue walls: every episode they show the cue colour of the room holding the goal.
  The agent spawns in the alcove, so the cue is only visible at the start.
- The shaft bends 90° into the corridor, so no straight line runs from the corridor, the junction or the
  rooms to a cue wall. With a straight corridor the alcove's back wall faced down the corridor and the
  edge ray of a 180° FOV saw the cue again at the junction, which let a memoryless agent re-read it.
  Past CUE_HIDDEN_AFTER units beyond the bend no cue wall is in line of sight.
- Rooms A and B are geometrically identical (point symmetric about the junction) and every wall has
  the same colour, so the rooms can't be told apart by looking. Only the remembered cue says which is right.
- A baffle wall inside each door hides the goal from the corridor and the junction.
- Entering the wrong room ends the episode (goal_regions), otherwise a memoryless agent can just check both.

The corridor length is L * max_speed, i.e. L steps at full speed, so L sets the memory horizon.
Train with env.goal_mode=regions and env.colour_mode=off (or per_wall).

    python scripts/generate_twin_rooms.py                # L = 8 32 64 128 256
    python scripts/generate_twin_rooms.py --L 16 512
"""
import argparse
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

MAX_SPEED = 13.0       # configs/env/base_env.yaml max_speed: corridor length = L steps at full speed
CORRIDOR_WIDTH = 60.0  # also the door and arm width
ALCOVE = 100.0         # side of the square cue alcove
SHAFT = 120.0          # vertical shaft from the alcove down to the corridor's bend
# A line from the corridor up the shaft must pass the bend's inner corner, so it can only reach the
# alcove from within CORRIDOR_WIDTH**2 / SHAFT units past the bend (30 units, about 2 steps)
CUE_HIDDEN_AFTER = CORRIDOR_WIDTH ** 2 / SHAFT
ARM = 60.0             # length of the junction arms between the corridor and the rooms
ROOM = 200.0           # side of the square rooms
BAFFLE_DEPTH = 50.0    # distance of the baffle from the door, inside the room
BAFFLE_HALF = 50.0     # half width of the baffle (wider than the door, so the goal is hidden)
GOAL_DEPTH = 130.0     # distance of the goal from the door, behind the baffle
MARGIN = 10.0          # keep every coordinate positive (render's w2s has no offset)
CUE_COLOURS = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]   # cue colour 0 -> room A, cue colour 1 -> room B
WALL_COLOUR = [0.5, 0.5, 0.5]


def edge(x0, y0, x1, y1, cue=False):
    e = {"from": {"x": float(x0), "y": float(y0)}, "to": {"x": float(x1), "y": float(y1)}, "colour": WALL_COLOUR}
    if cue:
        e["cue"] = True
    return e

def box_region(min_x, max_x, min_y, max_y):
    """Axis-aligned rectangle in the spawn-region format (list of edges)."""
    return [edge(min_x, min_y, max_x, min_y), edge(max_x, min_y, max_x, max_y),
            edge(max_x, max_y, min_x, max_y), edge(min_x, max_y, min_x, min_y)]

def twin_rooms(L: int) -> dict:
    half_w = CORRIDOR_WIDTH / 2
    corridor_len = L * MAX_SPEED

    y0 = MARGIN + ROOM + ARM + half_w          # corridor centre line
    x_alcove = MARGIN                           # alcove left wall
    x_shaft = x_alcove + (ALCOVE - CORRIDOR_WIDTH) / 2   # shaft (and bend) left wall, alcove centred on it
    x_corridor = x_shaft + CORRIDOR_WIDTH       # corridor start = bend's inner corner
    x_junction = x_corridor + corridor_len      # corridor end = left wall of the arms
    xc = x_junction + half_w                    # junction centre (rooms are centred on it)
    y_alcove = y0 + half_w + SHAFT              # alcove floor, the shaft opens into it

    edges = []

    # Cue alcove above the shaft: left, right and top walls, plus the two stubs around the shaft opening
    edges += [edge(x_alcove, y_alcove, x_alcove, y_alcove + ALCOVE, cue=True),
              edge(x_alcove + ALCOVE, y_alcove, x_alcove + ALCOVE, y_alcove + ALCOVE, cue=True),
              edge(x_alcove, y_alcove + ALCOVE, x_alcove + ALCOVE, y_alcove + ALCOVE, cue=True),
              edge(x_alcove, y_alcove, x_shaft, y_alcove, cue=True),
              edge(x_corridor, y_alcove, x_alcove + ALCOVE, y_alcove, cue=True)]

    # Shaft and bend: the left wall runs from the corridor floor up to the alcove, the right wall
    # from the bend's inner corner up to the alcove
    edges += [edge(x_shaft, y0 - half_w, x_shaft, y_alcove),
              edge(x_corridor, y0 + half_w, x_corridor, y_alcove)]

    # Corridor (its floor also closes the bend)
    edges += [edge(x_corridor, y0 + half_w, x_junction, y0 + half_w),
              edge(x_shaft, y0 - half_w, x_junction, y0 - half_w)]

    # Junction: left walls of the two arms and one right wall through both arms
    y_door_a = y0 + half_w + ARM                # door of room A (above)
    y_door_b = y0 - half_w - ARM                # door of room B (below)
    edges += [edge(x_junction, y0 + half_w, x_junction, y_door_a),
              edge(x_junction, y_door_b, x_junction, y0 - half_w),
              edge(xc + half_w, y_door_b, xc + half_w, y_door_a)]

    # Rooms: A opens downwards at y_door_a, B is A rotated by 180° about the junction centre
    goal_points, goal_regions = [], []
    for sign, y_door in ((1.0, y_door_a), (-1.0, y_door_b)):
        y_back = y_door + sign * ROOM
        edges += [edge(xc - ROOM / 2, y_door, xc - half_w, y_door),           # door wall, left of the door
                  edge(xc + half_w, y_door, xc + ROOM / 2, y_door),           # door wall, right of the door
                  edge(xc - ROOM / 2, y_door, xc - ROOM / 2, y_back),
                  edge(xc + ROOM / 2, y_door, xc + ROOM / 2, y_back),
                  edge(xc - ROOM / 2, y_back, xc + ROOM / 2, y_back),
                  edge(xc - BAFFLE_HALF, y_door + sign * BAFFLE_DEPTH,
                       xc + BAFFLE_HALF, y_door + sign * BAFFLE_DEPTH)]       # baffle
        goal_points.append({"x": xc, "y": y_door + sign * GOAL_DEPTH})
        goal_regions.append(box_region(xc - ROOM / 2, xc + ROOM / 2, min(y_door, y_back), max(y_door, y_back)))

    inset = 15.0
    return {
        "edges": edges,
        "start_pos": {"x": x_alcove + ALCOVE / 2, "y": y_alcove + ALCOVE / 2},
        # Reference goal for the metrics' room plot; the episode goal comes from goal_points
        "goal_pos": goal_points[0],
        "spawn_regions": [box_region(x_alcove + inset, x_alcove + ALCOVE - inset,
                                     y_alcove + inset, y_alcove + ALCOVE - inset)],
        "goal_points": goal_points,
        "goal_regions": goal_regions,
        # Square where the corridor meets both arms: every route to a room crosses it (E2 reset diagnostic)
        "junction_region": box_region(x_junction, x_junction + CORRIDOR_WIDTH, y0 - half_w, y0 + half_w),
        # Centre line from the alcove's back wall down the shaft and along the corridor to the junction's
        # corridor-side edge: env.junction_shaping measures the path distance to the junction along it, the E2
        # diagnostics the progress along the corridor (last segment)
        "junction_path": [{"x": x_shaft + half_w, "y": y_alcove + ALCOVE}, {"x": x_shaft + half_w, "y": y0},
                          {"x": x_junction, "y": y0}],
        "cue_colours": CUE_COLOURS,
        "memory_horizon_steps": L,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, nargs="+", default=[8, 32, 64, 128, 256])
    args = parser.parse_args()

    for L in args.L:
        path = ROOT_DIR / "rooms" / f"twin_rooms_L{L}.json"
        with open(path, "w") as f:
            json.dump(twin_rooms(L), f, indent=2)
        print(f"Wrote {path}")
