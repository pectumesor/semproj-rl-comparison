"""
Rollout videos of whole evaluation episodes, drawn from the env's state (not from env.render): the room's walls, the
goal regions with the episode's own goal room shaded, the cue walls in the episode's cue colour, the rays as the agent
saw them (coloured where they hit a coloured wall), the trail, the heading, and a banner with the title, the episode,
the step ("TOLD" on the steps the cue is in the input) and the outcome. In the twin rooms one episode per cue,
elsewhere a few random episodes; the episodes are fixed by seed, so every config is filmed on the same spawns and cues.

Used by scripts/train.py (the "Rollout" video of every run) and scripts/report_to_wandb.py (the report's rollouts).
Not imported by utils/__init__: utils.geometry -> utils -> ... -> envs is circular, import it after utils.
"""
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from utils.geometry import (walls_json_to_numpy, wall_attributes_json, goal_regions_json, get_spawn_regions_json,
                            region_to_box, start_goal_pos_from_json)
from utils.metrics import initial_recurrent_state

CUE_NAMES = {(1, 0, 0): "red", (0, 0, 1): "blue", (0, 1, 0): "green", (1, 1, 0): "yellow"}
W, BANNER = 576, 64              # video frame width and the text banner on top of the map; the map's height follows
                                 # the room's aspect ratio, all sizes divisible by 16 for the encoder
MAX_EPISODE_FRAMES = 600         # longer episodes are subsampled, so a 4096-step maze episode stays watchable
FPS = 20


def font(size: int):
    import matplotlib.font_manager as fm
    return ImageFont.truetype(fm.findfont("DejaVu Sans"), size)

class RoomCanvas:
    """World -> pixel transform of one room and its static picture (walls, regions), drawn once."""

    def __init__(self, room_path: str, size: int = W, pad: int = 24):
        self.walls = np.array(walls_json_to_numpy(room_path), dtype=np.float32)              # (W, 2, 2)
        _, self.cue_walls = wall_attributes_json(room_path)                                    # (W,)
        regions = goal_regions_json(room_path)
        self.goal_points, self.goal_boxes, self.cue_colours = regions if regions else (None, None, None)
        self.spawn_boxes = [region_to_box(r) for r in get_spawn_regions_json(room_path)]
        lo, hi = self.walls.reshape(-1, 2).min(0), self.walls.reshape(-1, 2).max(0)
        self.scale = (size - 2 * pad) / float((hi - lo).max())
        self.size = (size, int(16 * np.ceil(((hi - lo)[1] * self.scale + 2 * pad) / 16)))            # (width, height)
        # Centre the room in the canvas
        self.offset = pad + ((np.array(self.size) - 2 * pad) - (hi - lo) * self.scale) / 2 - lo * self.scale

    def px(self, p):
        """World (x, y) -> pixel, y up."""
        x, y = np.asarray(p, dtype=np.float32) * self.scale + self.offset
        return float(x), float(self.size[1] - y)

    def box(self, b):
        (x0, y0), (x1, y1) = self.px((b[0], b[2])), self.px((b[1], b[3]))
        return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]

    def base(self, goal_idx: int = None, cue_rgb=None, spawn: bool = False) -> Image.Image:
        img = Image.new("RGB", self.size, "white")
        d = ImageDraw.Draw(img)
        if spawn:
            for b in self.spawn_boxes:
                d.rectangle(self.box(b), fill=(222, 235, 250))
        if self.goal_boxes is not None:
            for k, b in enumerate(self.goal_boxes):
                own = goal_idx is None or k == goal_idx
                d.rectangle(self.box(b), fill=(214, 240, 220) if own else (236, 236, 236))
        for (start, end), cue in zip(self.walls, self.cue_walls):
            colour = tuple(int(255 * c) for c in cue_rgb) if (cue and cue_rgb is not None) else \
                     ((200, 60, 160) if cue else (0, 0, 0))
            d.line([self.px(start), self.px(end)], fill=colour, width=5 if cue else 2)
        return img

def room_picture(room_path: str) -> Image.Image:
    """Room with spawn regions (blue), goal regions (green), cue walls (magenta) and the start."""
    canvas = RoomCanvas(room_path)
    img = canvas.base(spawn=True)
    d = ImageDraw.Draw(img)
    if canvas.goal_points is not None:
        for k, p in enumerate(canvas.goal_points):
            x, y = canvas.px(p)
            colour = tuple(int(255 * c) for c in canvas.cue_colours[k]) if canvas.cue_colours is not None else (0, 160, 60)
            d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=colour, outline="black")
    start, _ = start_goal_pos_from_json(room_path)
    x, y = canvas.px(start)
    d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(0, 100, 255))
    return img

def cue_name(rgb) -> str:
    return CUE_NAMES.get(tuple(int(round(c)) for c in rgb), str([round(float(c), 2) for c in rgb]))


# ---------- rollouts ----------

@torch.no_grad()
def episode_frames(agent, env, obs, canvas: RoomCanvas, recurrent: bool, title: str, header: str, small, big):
    """One deterministic episode from obs, the observation of a fresh reset, as frames; returns (frames, outcome)."""
    base = env.unwrapped
    regions = base.goal_mode == "regions"
    goal_idx = int(base.goal_idx[0]) if regions else None
    cue_rgb = base.cue_colour[0].tolist() if regions else None
    background = canvas.base(goal_idx, cue_rgb)
    told_cue = base.cue_dim > 0 and base.cue_info == "every"

    state = initial_recurrent_state(agent, 1, base.device) if recurrent else None
    episode_start = torch.zeros(1, dtype=torch.bool, device=base.device)
    trail, frames, outcome = [canvas.px(base.agent_pos[0].tolist())], [], "timeout"
    # Every stride-th step is drawn. Past 2 * MAX_EPISODE_FRAMES the kept frames are halved and the stride doubled,
    # so a 4096-step episode never holds more than ~1200 frames (GBs of RGB arrays otherwise)
    step, stride = 0, 1
    while True:
        if step % stride == 0:
            told = told_cue and bool(base.cue_features()[0].any())
            frames.append(draw_frame(background, canvas, base, obs, trail, title, header, f"step {int(base.steps[0])}"
                                     + ("   TOLD" if told else ""), small, big, regions))
            if len(frames) >= 2 * MAX_EPISODE_FRAMES:
                frames, stride = frames[::2], 2 * stride
        if recurrent:
            action, state = agent.predict_action(obs, state, episode_start)
        else:
            action = agent.predict_action(obs)
        obs, _, terminated, truncated, info = env.step(action)
        trail.append(canvas.px(base.agent_pos[0].tolist()))
        step += 1
        if terminated[0] or truncated[0]:
            if bool(info["success"][0]):
                outcome = "success"
            elif terminated[0]:
                outcome = "wrong room" if regions else "terminated"
            break
    # Last frame with the outcome, held for a second
    last = draw_frame(background, canvas, base, obs, trail, title, header, f"step {int(base.steps[0])} -> {outcome}",
                      small, big, regions, outcome=outcome)
    if len(frames) > MAX_EPISODE_FRAMES:
        frames = frames[::int(np.ceil(len(frames) / MAX_EPISODE_FRAMES))]
    return frames + [last] * FPS, outcome

def draw_frame(background, canvas, base, obs, trail, title, header, status, small, big, regions, outcome=None):
    img = Image.new("RGB", (W, canvas.size[1] + BANNER), "white")
    img.paste(background, (0, BANNER))
    d = ImageDraw.Draw(img)
    shift = lambda p: (p[0], p[1] + BANNER)

    # Rays as the agent saw them: coloured where they hit a coloured wall (cue, goal), light grey otherwise
    intersect, _, _, _ = base.ray_cast.scan(base.agent_pos, base.facing_direction)          # (1, R, 2)
    rays = obs["rays"][0]
    rgb = (rays[-1] if rays.dim() == 3 else rays)[4:7].T                                    # newest frame, (R, 3)
    agent = shift(canvas.px(base.agent_pos[0].tolist()))
    for hit, c in zip(intersect[0].tolist(), rgb.tolist()):
        colour = tuple(int(255 * v) for v in c) if sum(c) > 0.05 else (205, 205, 205)
        d.line([agent, shift(canvas.px(hit))], fill=colour, width=1)

    if len(trail) > 1:
        d.line([shift(p) for p in trail], fill=(0, 90, 220), width=2)
    if not regions:
        gx, gy = shift(canvas.px(base.goal_pos[0].tolist()))
        d.ellipse([gx - 8, gy - 8, gx + 8, gy + 8], fill=(0, 190, 70), outline="black")
    heading = float(base.facing_direction[0])
    tip = (agent[0] + 14 * np.cos(heading), agent[1] - 14 * np.sin(heading))
    d.line([agent, tip], fill="black", width=2)
    d.ellipse([agent[0] - 6, agent[1] - 6, agent[0] + 6, agent[1] + 6], fill=(0, 100, 255), outline="black")

    d.text((8, 6), title, fill="black", font=big)
    d.text((8, 28), header, fill=(70, 70, 70), font=small)
    colour = {"success": (0, 140, 60), "wrong room": (200, 40, 40), "timeout": (200, 120, 0)}.get(outcome, (70, 70, 70))
    d.text((8, 45), status, fill=colour, font=small)
    return np.asarray(img)

def record(agent, env, room_path: str, recurrent: bool, title: str, n_random: int = 3, max_tries: int = 60):
    """
    Frames of one episode per cue (goal regions) or of n_random episodes, and the outcome of each. env has a single
    env (num_envs=1). The global torch RNG is restored afterwards, so the fixed episode seeds here don't change
    whatever the caller samples next (train.py's metric evaluation).
    """
    base = env.unwrapped
    canvas = RoomCanvas(room_path)
    small, big = font(13), font(15)
    regions = base.goal_mode == "regions"
    wanted = list(range(len(base.goal_points))) if regions else list(range(n_random))
    frames, outcomes = [], []
    with torch.random.fork_rng(devices=[base.device] if torch.device(base.device).type == "cuda" else []):
        for attempt in range(max_tries):
            if not wanted:
                break
            torch.manual_seed(1000 + attempt)   # fixed episodes: the same spawns and cues for every config
            obs, _ = env.reset()
            key = int(base.goal_idx[0]) if regions else wanted[0]
            if key not in wanted:
                continue
            wanted.remove(key)
            cue = f"cue {cue_name(base.cue_colour[0].tolist())}" if regions else ""
            n = len(base.goal_points) if regions else n_random
            header = f"episode {len(outcomes) + 1} of {n}" + (f"  ·  {cue}, goal room shaded" if regions else "")
            episode, outcome = episode_frames(agent, env, obs, canvas, recurrent, title, header, small, big)
            frames += episode
            outcomes.append(f"{cue or f'episode {len(outcomes) + 1}'}: {outcome}")
    return frames, outcomes
