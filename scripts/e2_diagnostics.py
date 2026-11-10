"""
Memory diagnostics of the twin-room cue task (experiment E2), run on trained checkpoints.

For every seed of one config it rolls out n_episodes evaluation episodes (one per env, deterministic
policy) and reports:

    1. Reset at the junction (recurrent backbones): the same episodes again, with the recurrent state
       zeroed once, the first step the agent stands in the junction. An agent that carries the cue in
       memory falls to chance (1/K rooms); one that doesn't is unaffected.
    2. Confusion matrix: cue shown (rows) x room entered (columns, last column = timeout). A memoryless
       policy gives the same row for every cue.
    3. Linear probe (recurrent backbones): logistic regression decoding the cue from the last layer's
       hidden state, per bin of steps since the cue was last in view, trained and tested on disjoint
       episodes. Accuracy at chance (1/K) means the cue is gone from memory at that distance. Once the cue
       shapes the trajectory the current observation carries it too, so high accuracy alone isn't memory.
    4. Reset sweep (recurrent backbones): the reset of 1., at several fractions of the corridor instead of the
       junction. Where the reset stops hurting the choice is committed, to the recurrent state or to the body.
    5. Mirrored pose (memoryless and recurrent, no frame stack): halfway down the corridor, reflect position,
       heading and last turn about the centre line and keep the recurrent state (Déletang et al. 2021). An agent
       that keeps the cue in its lateral offset or heading falls below chance, one that keeps it in memory doesn't.
    6. Swap (recurrent backbones): at the same point, take the recurrent state, the pose or both from an episode
       of the other cue. The room then follows whichever carries the cue; a swap with an episode of the same cue
       is the control for the damage the patch itself does.

The memory and reset runs use the same RNG seed, so they start from the same spawns, cues and goals and
only differ from the reset onwards.

Run it with the overrides the config was trained with (room, backbone, bptt, goal_mode, colour_mode, ...),
it finds the seeds under <storage>/models/<timestamp>/<room>/<storage name>/ like train.py writes them:

    python scripts/e2_diagnostics.py env.room_path=rooms/twin_rooms_L32.json env.goal_mode=regions \
        env.colour_mode=off env.goal_radius=16 env.gamma=0.999 algorithm.max_steps=320 \
        backbone=lstm_backbone backbone.bptt_window_size=64 storage=<dir> timestamp=<E2 sweep timestamp>

Options (hydra overrides, all optional): +diagnostics.n_episodes=200 +diagnostics.seed=0
+diagnostics.probe=true +diagnostics.checkpoint=<path to one best.pt> (instead of the sweep folders)
+diagnostics.reset_fractions=[0.0,0.25,0.5,0.75] +diagnostics.intervention_fraction=0.5.
Per seed: e2_diagnostics.json and e2_episodes.npz next to the seed's eval metrics; one line per seed is
appended to <storage>/e2_diagnostics/results.jsonl.
"""
import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import hydra
from omegaconf import DictConfig, open_dict
import numpy as np
import torch
import torch.optim as optim

import utils  # noqa: F401  -- must come before envs: envs -> utils.geometry -> utils -> algorithms -> envs is circular
from utils import create_ppo_agent, create_environment, initial_recurrent_state
from envs import compute_num_rays, compute_proprio_dim
from scripts.train import run_names

# Bins of "steps since the cue was last in view" for the probe: [low, high] inclusive
PROBE_BINS = [(0, 0), (1, 8), (9, 32), (33, 64), (65, 128), (129, 256), (257, 10 ** 9)]
SUCCESS, WRONG_ROOM, TIMEOUT = 0, 1, 2


def get_pose(base):
    """(E, 5) position, heading and last action (speed, turn): everything of the agent's body the policy can read."""
    return torch.stack([base.agent_pos[:, 0], base.agent_pos[:, 1], base.facing_direction,
                        base.last_speed, base.last_turning], dim=-1)

def set_pose(base, pose, mask):
    base.agent_pos[mask]        = pose[:, :2]
    base.facing_direction[mask] = pose[:, 2]
    base.last_speed[mask]       = pose[:, 3]
    base.last_turning[mask]     = pose[:, 4]

def mirror_pose(base, mask):
    """
    Reflect the pose of the envs in mask about the corridor's centre line (last junction_path segment):
    position, heading and last turn. An agent that keeps the cue in its lateral offset or heading now heads
    for the other room; one that keeps it in its recurrent state corrects (Déletang et al. 2021, T-maze push).
    """
    a, b = base.junction_path[-2], base.junction_path[-1]
    u = (b - a) / (b - a).norm()                                         # (2,) corridor direction
    rel = base.agent_pos[mask] - a                                       # (M, 2)
    base.agent_pos[mask] = a + 2.0 * (rel @ u)[:, None] * u - rel
    base.facing_direction[mask] = 2.0 * torch.atan2(u[1], u[0]) - base.facing_direction[mask]
    base.last_turning[mask] = -base.last_turning[mask]

def partners(cue: torch.Tensor, same: bool) -> torch.Tensor:
    """(N,) for every episode the next one (cyclically) whose cue is the same / differs, -1 if there is none."""
    n = len(cue)
    partner = torch.full((n,), -1, dtype=torch.long)
    for i in range(n):
        for k in range(1, n):
            j = (i + k) % n
            if (cue[j] == cue[i]) == same:
                partner[i] = j
                break
    return partner

# Interventions applied once per episode at the intervention point, see diagnostic_rollouts
INTERVENTIONS = ("none", "reset", "mirror", "swap_hidden", "swap_pose", "swap_both")

@torch.no_grad()
def diagnostic_rollouts(agent, env, recurrent: bool, seed: int, intervention: str = "none",
                        corridor_fraction: float = None, partner: torch.Tensor = None, source: dict = None,
                        collect_hidden: bool = False):
    """
    One episode per env, until every env has terminated or been truncated.

    intervention, applied once per episode, at the first step the agent is past corridor_fraction of the
    corridor (0 = the bend, 1 = the junction edge; None = the first step inside the junction):
        none         nothing; records the recurrent state and pose there (the source of the swaps)
        reset        zero the recurrent state (h, and c for the LSTM)
        mirror       reflect the pose about the corridor's centre line, keep the recurrent state
        swap_hidden  take the partner episode's recurrent state at its intervention point, keep the pose
        swap_pose    take the partner episode's pose at its intervention point, keep the recurrent state
        swap_both    take both
    The swaps need the "none" run's rollouts as source, with the same seed: the policy is deterministic, so every
    episode runs exactly as in the source up to its intervention point. Episodes whose partner never got there
    are left alone, "intervened" says which ones were changed.

    Returns per-episode numpy arrays (cue, outcome, room, reached_junction, n_resets, intervened) and, if
    collect_hidden, the probe samples (hidden (N, H), cue label (N,), steps since cue (N,), episode id (N,)).
    """
    assert intervention in INTERVENTIONS, intervention
    base = env.unwrapped
    num_envs = base.num_envs

    torch.manual_seed(seed)   # same spawns, cues and goals for every run with this seed
    obs, _ = env.reset()
    cue = base.goal_idx.clone()

    state = initial_recurrent_state(agent, num_envs, base.device) if recurrent else None
    episode_start = torch.zeros(num_envs, dtype=torch.bool, device=base.device)
    active   = torch.ones(num_envs, dtype=torch.bool, device=base.device)
    outcome  = torch.full((num_envs,), TIMEOUT, dtype=torch.long, device=base.device)
    room     = torch.full((num_envs,), -1, dtype=torch.long, device=base.device)
    reached  = torch.zeros(num_envs, dtype=torch.bool, device=base.device)
    n_resets = torch.zeros(num_envs, dtype=torch.long, device=base.device)
    at_point = torch.zeros(num_envs, dtype=torch.bool, device=base.device)    # past the intervention point
    intervened = torch.zeros(num_envs, dtype=torch.bool, device=base.device)
    # What the "none" run records at the intervention point
    point_pose = torch.zeros(num_envs, 5, device=base.device)
    point_state = tuple(torch.zeros_like(s) for s in state) if recurrent else None
    if intervention.startswith("swap"):
        valid = (partner >= 0) & torch.as_tensor(source["intervened"], device=base.device)[partner.clamp(min=0)]
        source_pose = torch.as_tensor(source["point_pose"], device=base.device)
        source_state = tuple(torch.as_tensor(s, device=base.device) for s in source["point_state"])
    # Steps since the cue was last in view, -1 until it has been seen once
    since_cue = torch.where(base.cue_visible, 0, -1)

    hidden, labels, since, episode = [], [], [], []
    env_ids = torch.arange(num_envs, device=base.device)

    while active.any():
        if recurrent:
            action, state = agent.predict_action(obs, state, episode_start)
        else:
            action = agent.predict_action(obs)

        # Probe sample: the last layer's hidden state right after reading obs_t, labelled at obs_t
        if collect_hidden and recurrent:
            keep = active & (since_cue >= 0)
            hidden.append(state[0][-1][keep].cpu())
            labels.append(cue[keep].cpu())
            since.append(since_cue[keep].cpu())
            episode.append(env_ids[keep].cpu())

        obs, _, terminated, truncated, info = env.step(action)
        finished = (terminated | truncated) & active
        outcome[finished & info["success"]] = SUCCESS
        outcome[finished & terminated & ~info["success"]] = WRONG_ROOM
        room = torch.where(finished & terminated, info["room"], room)
        active &= ~(terminated | truncated)

        since_cue = torch.where(base.cue_visible, 0, torch.where(since_cue >= 0, since_cue + 1, since_cue))

        entered = base.in_junction() & active
        reached |= entered
        here = entered if corridor_fraction is None else (base.corridor_fraction() >= corridor_fraction) & active
        first_time = here & ~at_point
        at_point |= here

        # The intervention acts before the policy reads the next observation
        episode_start = torch.zeros_like(first_time)
        if intervention == "none":
            intervened |= first_time
            point_pose[first_time] = get_pose(base)[first_time]
            if recurrent:
                for saved, current in zip(point_state, state):
                    saved[:, first_time] = current[:, first_time]
        elif intervention == "reset":
            # Raise the episode-start flag once, so the backbone zeroes the recurrent state
            episode_start = first_time
            intervened |= first_time
        else:
            change = first_time & valid if intervention.startswith("swap") else first_time
            intervened |= change
            if intervention == "mirror":
                mirror_pose(base, change)
            if intervention in ("swap_pose", "swap_both"):
                set_pose(base, source_pose[partner[change]], change)
            if intervention in ("swap_hidden", "swap_both"):
                for current, saved in zip(state, source_state):
                    current[:, change] = saved[:, partner[change]]
            if intervention in ("mirror", "swap_pose", "swap_both") and change.any():
                fresh = base.observe()
                for key in obs:
                    obs[key][change] = fresh[key][change]
        n_resets += episode_start

    rollouts = {
        "cue": cue.cpu().numpy(), "outcome": outcome.cpu().numpy(), "room": room.cpu().numpy(),
        "reached_junction": reached.cpu().numpy(), "n_resets": n_resets.cpu().numpy(),
        "intervened": intervened.cpu().numpy(),
    }
    if intervention == "none":
        rollouts["point_pose"] = point_pose.cpu().numpy()
        if recurrent:
            rollouts["point_state"] = tuple(s.cpu().numpy() for s in point_state)
    if collect_hidden and recurrent and hidden:
        rollouts["probe"] = (torch.cat(hidden), torch.cat(labels), torch.cat(since), torch.cat(episode))
    return rollouts

def confusion_matrix(cue, room, num_rooms: int):
    """(K, K + 1) counts: rows = cue shown, columns = room entered, last column = no room (timeout)."""
    counts = np.zeros((num_rooms, num_rooms + 1), dtype=int)
    for c, r in zip(cue, room):
        counts[c, r if r >= 0 else num_rooms] += 1
    return counts

def summarise(rollouts, num_rooms: int, subset: np.ndarray = None):
    """Outcome rates and confusion matrix, over every episode or only the subset (e.g. the intervened ones)."""
    keep = np.ones(len(rollouts["outcome"]), dtype=bool) if subset is None else subset
    outcome, reached = rollouts["outcome"][keep], rollouts["reached_junction"][keep]
    if not keep.any():
        return {"n": 0}
    return {
        "n": int(keep.sum()),
        "success": float((outcome == SUCCESS).mean()),
        "wrong_room": float((outcome == WRONG_ROOM).mean()),
        "timeout": float((outcome == TIMEOUT).mean()),
        "reached_junction": float(reached.mean()),
        # Success among the episodes that reached the junction, the ones a reset can affect
        "success_given_junction": float((outcome[reached] == SUCCESS).mean()) if reached.any() else float("nan"),
        "confusion": confusion_matrix(rollouts["cue"][keep], rollouts["room"][keep], num_rooms).tolist(),
    }

def fit_logistic(x, y, num_classes: int, l2: float = 1e-3, steps: int = 200):
    """Multinomial logistic regression with LBFGS, returns a predict function."""
    weight = torch.zeros(x.shape[1], num_classes, requires_grad=True)
    bias = torch.zeros(num_classes, requires_grad=True)
    opt = optim.LBFGS([weight, bias], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(x @ weight + bias, y) + l2 * weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return lambda z: (z @ weight + bias).argmax(dim=-1)

def linear_probe(hidden, labels, since, episode, num_classes: int, seed: int, test_frac: float = 0.3,
                 min_per_split: int = 20):
    """
    Held-out accuracy of decoding the cue from the hidden state, per PROBE_BINS bin. Train and test
    samples come from disjoint episodes, so the probe can't memorise an episode's own trajectory.
    """
    rng = np.random.default_rng(seed)
    episodes = np.unique(episode.numpy())
    test_eps = torch.as_tensor(rng.choice(episodes, size=max(1, int(test_frac * len(episodes))), replace=False))
    is_test = torch.isin(episode, test_eps)

    results = []
    for low, high in PROBE_BINS:
        in_bin = (since >= low) & (since <= high)
        train, test = in_bin & ~is_test, in_bin & is_test
        row = {"bin": [low, high], "n_train": int(train.sum()), "n_test": int(test.sum()), "accuracy": None}
        if row["n_train"] >= min_per_split and row["n_test"] >= min_per_split and labels[train].unique().numel() > 1:
            x_train, x_test = hidden[train], hidden[test]
            mean, std = x_train.mean(0), x_train.std(0) + 1e-6        # standardise with train statistics only
            predict = fit_logistic((x_train - mean) / std, labels[train], num_classes)
            row["accuracy"] = float((predict((x_test - mean) / std) == labels[test]).float().mean())
        results.append(row)
    return results


@hydra.main( config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):

    diagnostics = cfg.get("diagnostics", {})
    n_episodes = int(diagnostics.get("n_episodes", 200))
    seed       = int(diagnostics.get("seed", 0))
    run_probe  = bool(diagnostics.get("probe", True))
    checkpoint = diagnostics.get("checkpoint", None)
    # Corridor fractions of the reset sweep (0 = the bend, 1 = the junction edge), and where the push and the
    # swaps happen: halfway, long after the cue went out of view and well before the junction
    reset_fractions = [float(f) for f in diagnostics.get("reset_fractions", [0.0, 0.25, 0.5, 0.75])]
    point = float(diagnostics.get("intervention_fraction", 0.5))
    frame_stack = cfg.observation.frame_stack.enabled

    if cfg.env.goal_mode != "regions":
        raise ValueError("The E2 diagnostics need the twin-room cue task: set env.goal_mode=regions")

    with open_dict(cfg):
        cfg.env.proprio_dim = compute_proprio_dim(cfg)
    recurrent = cfg.backbone.name != "mlp"
    _, storage_name = run_names(cfg)
    room_name = Path(cfg.env.room_path).stem

    if checkpoint is not None:
        checkpoints = [Path(checkpoint)]
    else:
        config_dir = Path(cfg.storage) / "models" / str(cfg.timestamp) / room_name / storage_name
        checkpoints = sorted(config_dir.glob("seed_*/best.pt"), key=lambda p: int(p.parent.name.split("_")[-1]))
        if not checkpoints:
            raise FileNotFoundError(f"No seed_*/best.pt under {config_dir}")

    num_rays = compute_num_rays(cfg.env.fov, cfg.env.ray_density)
    ray_dim = np.array([cfg.env.ray_encoding, num_rays])
    out_log = Path(cfg.storage) / "e2_diagnostics"
    out_log.mkdir(parents=True, exist_ok=True)

    for path in checkpoints:
        agent = create_ppo_agent(ray_dim=ray_dim, cfg=cfg)
        agent.load_model(path, "cpu", optim.Adam(agent.parameters()))
        agent.eval()
        env = create_environment(cfg=cfg, agent=agent, num_rays=num_rays, ray_dim=ray_dim,
                                 num_envs=n_episodes, device="cpu")
        num_rooms = len(env.unwrapped.goal_points)

        memory = diagnostic_rollouts(agent, env, recurrent, seed, collect_hidden=run_probe)
        result = {
            "room": room_name, "storage_name": storage_name, "checkpoint": str(path),
            "seed": path.parent.name, "n_episodes": n_episodes, "rng_seed": seed, "chance": 1.0 / num_rooms,
            "memory": summarise(memory, num_rooms),
        }
        arrays = {f"memory_{key}": value for key, value in memory.items() if key not in ("probe", "point_pose", "point_state")}

        if recurrent:
            reset = diagnostic_rollouts(agent, env, recurrent, seed, intervention="reset")
            result["reset_at_junction"] = summarise(reset, num_rooms)
            result["reset_drop"] = result["memory"]["success"] - result["reset_at_junction"]["success"]
            arrays.update({f"reset_{key}": value for key, value in reset.items()})
            if "probe" in memory:
                result["probe"] = linear_probe(*memory["probe"], num_classes=num_rooms, seed=seed)
            # Reset sweep along the corridor: past the point where the choice is committed (to the recurrent
            # state or to the body) a reset no longer hurts. Rates over the episodes that got there
            result["reset_sweep"] = []
            for fraction in reset_fractions:
                sweep = diagnostic_rollouts(agent, env, recurrent, seed, intervention="reset", corridor_fraction=fraction)
                result["reset_sweep"].append({"corridor_fraction": fraction,
                                              **summarise(sweep, num_rooms, sweep["intervened"])})

        # Body vs recurrent memory at the intervention point, rates over the episodes that got there. Not with a
        # frame stack: its older frames would still show the pose before the intervention
        if not frame_stack and env.unwrapped.junction_path is not None:
            source = diagnostic_rollouts(agent, env, recurrent, seed, corridor_fraction=point)
            mirror = diagnostic_rollouts(agent, env, recurrent, seed, intervention="mirror", corridor_fraction=point)
            # Mirrored pose: success stays if the cue is kept in the recurrent state, falls below chance if it is
            # kept in the lateral offset or heading. "baseline" is the same episodes without any intervention
            result["mirror"] = {"corridor_fraction": point,
                                "baseline": summarise(source, num_rooms, source["intervened"]),
                                "mirrored": summarise(mirror, num_rooms, mirror["intervened"])}
            if recurrent:
                # 2 x 2 swap with an episode of the other cue: the room follows the recurrent state (hidden_other
                # fails, pose_other succeeds) or the body (the reverse). hidden_same swaps with an episode of the
                # same cue, the control for the damage the patch itself does
                cue = torch.as_tensor(source["cue"])
                other, same = partners(cue, same=False), partners(cue, same=True)
                result["swap"] = {"corridor_fraction": point}
                for name, intervention, partner in (("hidden_other", "swap_hidden", other),
                                                    ("pose_other", "swap_pose", other),
                                                    ("both_other", "swap_both", other),
                                                    ("hidden_same", "swap_hidden", same)):
                    swap = diagnostic_rollouts(agent, env, recurrent, seed, intervention=intervention,
                                               corridor_fraction=point, partner=partner, source=source)
                    result["swap"][name] = summarise(swap, num_rooms, swap["intervened"])

        # Next to the seed's eval metrics for a sweep, next to the checkpoint for a single best.pt
        out_dir = (path.parent if checkpoint is not None else
                   Path(cfg.storage) / "eval_metrics" / str(cfg.timestamp) / room_name / storage_name / path.parent.name)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "e2_diagnostics.json", "w") as f:
            json.dump(result, f, indent=2)
        np.savez(out_dir / "e2_episodes.npz", **arrays)
        with open(out_log / "results.jsonl", "a") as f:
            f.write(json.dumps({key: value for key, value in result.items() if key != "probe"}) + "\n")

        reset_str = (f" reset_success={result['reset_at_junction']['success']:.2f} drop={result['reset_drop']:+.2f}"
                     if recurrent else "")
        probe_str = (" probe=" + ",".join(f"{row['bin'][0]}-{row['bin'][1]}:{row['accuracy']:.2f}"
                                          for row in result.get("probe", []) if row["accuracy"] is not None))
        print(f"[E2] {room_name} {path.parent.name} success={result['memory']['success']:.2f}"
              f"{reset_str}{probe_str if recurrent else ''}")
        print(f"[E2] confusion (cue x room, last column = timeout): {result['memory']['confusion']}")
        rate = lambda summary: f"{summary['success']:.2f} (n={summary['n']})" if summary["n"] else "n/a (n=0)"
        if "reset_sweep" in result:
            print("[E2] reset sweep success: " + " ".join(f"{row['corridor_fraction']:.2f}:{rate(row)}"
                                                          for row in result["reset_sweep"]))
        if "mirror" in result:
            print(f"[E2] mirror at {point:.2f}: baseline {rate(result['mirror']['baseline'])}"
                  f" mirrored {rate(result['mirror']['mirrored'])}")
        if "swap" in result:
            print("[E2] swap success: " + " ".join(f"{name}:{rate(summary)}" for name, summary in result["swap"].items()
                                                  if name != "corridor_fraction"))


if __name__ == "__main__":
    main()
