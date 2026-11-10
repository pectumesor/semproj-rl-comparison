"""
Checks for the experiment-plan features (colour modes, per-env goals, twin-room cue task,
information slots, GRU, discrete actions, padded AUC, SPL / nDTW metrics).

    python -m pytest tests/test_experiment_features.py -q
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

import utils  # noqa: F401  -- must come before envs (circular import)
from utils import create_ppo_agent, evaluate_model_on_metrics
from utils.geometry import InitialPositionSampler, point_segment_distance
from envs import NavigationEnv, GridCode, RandomFourierFeatures, compute_num_rays, compute_proprio_dim
from models import SimpleGRU, SimpleLSTM


def make_cfg(*overrides):
    with initialize_config_dir(config_dir=str(ROOT_DIR / "configs"), version_base=None):
        cfg = compose(config_name="base", overrides=["wandb.mode=disabled", *overrides])
    with open_dict(cfg):
        cfg.env.proprio_dim = compute_proprio_dim(cfg)
    return cfg

def make_env(cfg, num_envs=8):
    num_rays = compute_num_rays(cfg.env.fov, cfg.env.ray_density)
    return NavigationEnv(cfg, None, num_rays, np.array([cfg.env.ray_encoding, num_rays]), num_envs)


@pytest.fixture(autouse=True)
def run_from_repo_root(monkeypatch):
    monkeypatch.chdir(ROOT_DIR)  # room paths in the configs are relative to the repo
    torch.manual_seed(0)


def test_raycast_wall_index():
    env = make_env(make_cfg("env.room_path=rooms/one_room.json"), num_envs=1)
    env.agent_pos[:] = torch.tensor([[200.0, 200.0]])
    env.facing_direction[:] = 0.0                       # facing +x: the centre ray hits the right wall
    _, distances, _, wall_idx = env.ray_cast.scan(env.agent_pos, env.facing_direction)
    centre = env.num_rays // 2
    start, end = env.walls[wall_idx[0, centre]]
    assert start[0] == end[0] == max(env.bounding_box[:2])   # the vertical wall at max x
    assert torch.isfinite(distances[0, centre])


def test_random_goals_respect_min_distance_and_walls():
    cfg = make_cfg("env.room_path=rooms/medium_maze.json", "env.goal_mode=random", "env.min_goal_distance=200")
    env = make_env(cfg, num_envs=64)
    env.reset()
    assert (torch.norm(env.goal_pos - env.agent_pos, dim=-1) >= 200).all()
    walls = env.spawn_sampler
    for pos in (env.agent_pos, env.goal_pos):
        assert (point_segment_distance(pos, walls.wall_starts, walls.wall_ends).min(dim=-1).values >= walls.wall_offset).all()
    first_goals = env.goal_pos.clone()
    env.reset()
    assert not torch.equal(first_goals, env.goal_pos)


def test_spawn_sampler_rejects_points_on_walls():
    sampler = InitialPositionSampler("rooms/hard_maze.json", wall_offset=10, device="cpu")
    pos = sampler.sample(2000)
    assert not sampler.too_close_to_wall(pos).any()


def test_twin_rooms_cue_and_wrong_room():
    cfg = make_cfg("env.room_path=rooms/twin_rooms_L8.json", "env.goal_mode=regions", "env.colour_mode=off")
    env = make_env(cfg, num_envs=32)
    obs, _ = env.reset()
    # Cue colour matches the room holding the goal
    assert torch.equal(env.cue_colour, env.cue_colours[env.goal_idx])
    # The only coloured walls are the cue walls (colour off), so any coloured ray shows the cue
    rgb = obs["rays"][:, 4:, :].permute(0, 2, 1)
    coloured = rgb.abs().sum(-1) > 0
    assert coloured.any()
    assert torch.allclose(rgb[coloured], env.cue_colour[:, None, :].expand_as(rgb)[coloured])

    # Teleport into the wrong room -> terminated without success, right room's goal -> success
    wrong = 1 - env.goal_idx
    env.agent_pos[:] = env.goal_points[wrong]
    action = torch.zeros(env.num_envs, 2)
    _, reward, terminated, _, info = env.step(action)
    assert terminated.all() and not info["success"].any()
    env.reset()
    env.agent_pos[:] = env.goal_pos
    _, reward, terminated, _, info = env.step(action)
    assert info["success"].all() and terminated.all() and (reward > 0.9).all()


@pytest.mark.parametrize("L", [8, 32, 256])
def test_twin_rooms_cue_hidden_past_the_bend(L):
    from scripts.generate_twin_rooms import twin_rooms, CUE_HIDDEN_AFTER, CORRIDOR_WIDTH, MAX_SPEED
    import json

    room_path = f"rooms/twin_rooms_L{L}.json"
    # The committed rooms must match the generator, otherwise they predate the bend
    assert json.load(open(room_path)) == json.loads(json.dumps(twin_rooms(L)))

    # 360° view with dense rays, so no heading can sneak a look at the cue
    cfg = make_cfg(f"env.room_path={room_path}", "env.goal_mode=regions", "env.colour_mode=off",
                   "env.fov=360", "env.ray_density=2.0")
    env = make_env(cfg)
    goal_a, goal_b = env.goal_points.tolist()
    half_w = CORRIDOR_WIDTH / 2
    y0 = (goal_a[1] + goal_b[1]) / 2                    # corridor centre line
    x_junction = goal_a[0] - half_w
    x_corridor = x_junction - L * MAX_SPEED             # bend's inner corner

    # Corridor past the bend, the junction with both arms, and both rooms (wall offset kept)
    margin = 5.0
    boxes = torch.tensor([[x_corridor + CUE_HIDDEN_AFTER + margin, x_junction, y0 - half_w + margin, y0 + half_w - margin],
                          [x_junction + margin, x_junction + CORRIDOR_WIDTH - margin, goal_b[1], goal_a[1]]])
    pos = torch.cat([box[[0, 2]] + torch.rand(2000, 2) * (box[[1, 3]] - box[[0, 2]]) for box in boxes]
                    + [env.goal_points[i] + (torch.rand(2000, 2) - 0.5) * 120.0 for i in range(2)])
    pos = pos[~env.spawn_sampler.too_close_to_wall(pos)]
    facing = torch.rand(pos.shape[0]) * 2 * torch.pi

    _, _, _, wall_idx = env.ray_cast.scan(pos, facing)  # (N, R)
    sees_cue = env.cue_walls[wall_idx.clamp(min=0)] & (wall_idx >= 0)
    assert not sees_cue.any(), f"{int(sees_cue.any(-1).sum())} of {pos.shape[0]} positions see a cue wall"

    # ...while the cue is in view at the start
    start = env.spawn_sampler.sample(200)
    _, _, _, wall_idx = env.ray_cast.scan(start, torch.rand(200) * 2 * torch.pi)
    assert (env.cue_walls[wall_idx.clamp(min=0)] & (wall_idx >= 0)).any(-1).all()


def test_colour_modes():
    for mode in ("global", "off", "room_local", "per_wall"):
        env = make_env(make_cfg("env.room_path=rooms/four_room.json", f"env.colour_mode={mode}"))
        obs, _ = env.reset()
        rgb = obs["rays"][:, 4:, :]
        assert rgb.min() >= 0 and rgb.max() <= 1
        if mode == "off":
            assert rgb.abs().sum() == 0
        if mode == "per_wall":
            hit = obs["rays"][:, 2, :] > 0
            assert torch.allclose(rgb.permute(0, 2, 1)[hit], torch.full((int(hit.sum()), 3), 0.5))


def test_colour_seed_is_independent_of_rl_seed():
    a = make_env(make_cfg("env.room_path=rooms/four_room.json", "seed=0"))
    b = make_env(make_cfg("env.room_path=rooms/four_room.json", "seed=7"))
    pts = torch.rand(4, 5) * 500
    assert torch.equal(a.color_field(pts, pts), b.color_field(pts, pts))


def test_information_slot_shapes():
    reserve = make_cfg("env.goal_mode=random", "env.information.reserve_info_slots=true",
                       "env.information.position_info=xy", "env.room_path=rooms/four_room.json")
    plain = make_cfg("env.goal_mode=random", "env.information.goal_info=oracle",
                     "env.information.position_info=grid", "env.information.goal_grid=true",
                     "env.room_path=rooms/four_room.json")
    none = make_cfg("env.room_path=rooms/four_room.json")
    for cfg, expected in ((reserve, 2 + 5 + 64 + 64), (plain, 2 + 5 + 64 + 64), (none, 2)):
        env = make_env(cfg)
        obs, _ = env.reset()
        obs, *_ = env.step(torch.rand(env.num_envs, 2))
        assert cfg.env.proprio_dim == expected == obs["proprio"].shape[-1]
        assert env.observation_space["proprio"].shape == (expected,)
        assert torch.isfinite(obs["proprio"]).all()


def test_goal_information_modes():
    cfg = make_cfg("env.room_path=rooms/one_room.json", "env.goal_mode=random", "env.information.goal_info=oracle")
    env = make_env(cfg, num_envs=1)
    env.reset()
    env.agent_pos[:] = torch.tensor([[100.0, 100.0]])
    env.goal_pos[:] = torch.tensor([[100.0, 300.0]])
    env.facing_direction[:] = 0.0
    feat = env.goal_features()[0]
    assert torch.allclose(feat[1:3], torch.tensor([0.0, 1.0]), atol=1e-5)  # goal 90° to the left
    assert feat[3] == 1.0 and feat[4] == 0.0

    for mode in ("last_seen_updated", "last_seen_frozen"):
        env.goal_info = mode
        env._reset_information(torch.ones(1, dtype=torch.bool))
        assert env.goal_features().abs().sum() == 0            # never seen yet
        env.update_goal_memory(torch.ones(1, env.num_rays, dtype=torch.bool))
        seen = env.goal_features()[0].clone()
        env.agent_pos[:] = torch.tensor([[300.0, 100.0]])        # walk away without seeing the goal again
        moved = env.goal_features()[0]
        if mode == "last_seen_updated":
            assert not torch.allclose(seen[:3], moved[:3])        # re-expressed in the new frame
        else:
            assert torch.allclose(seen[:3], moved[:3])            # frozen at the sighting


def test_grid_code_and_rff_range():
    grid = GridCode(modules=4, phases=16)
    rff = RandomFourierFeatures(dim=64)
    pos = torch.rand(1000, 2) * 1200
    for code in (grid(pos), rff(pos)):
        assert code.shape == (1000, 64)
        assert code.min() >= -1e-6 and code.max() <= 1 + 1e-6
    # Periodic: moving by one wave period along a module's wave vector keeps that wave's phase
    assert grid(pos).std(dim=0).min() > 0.05


def test_gru_matches_lstm_interface():
    gru, lstm = SimpleGRU(8, 16, 2), SimpleLSTM(8, 16, 2)
    state = (torch.zeros(2, 4, 16), torch.zeros(2, 4, 16))
    x, done = torch.rand(3 * 4, 8), torch.zeros(3, 4)
    for model in (gru, lstm):
        out, new_state = model(x, state, done)
        assert out.shape == (12, 16) and new_state[0].shape == (2, 4, 16) and new_state[1].shape == (2, 4, 16)
        assert model.num_layers == 2 and model.hidden_size == 16


def test_discrete_agent_maps_to_env_actions():
    cfg = make_cfg("env.action_type=discrete", "env.discrete_actions=5", "env.room_path=rooms/one_room.json")
    num_rays = compute_num_rays(cfg.env.fov, cfg.env.ray_density)
    agent = create_ppo_agent(ray_dim=np.array([cfg.env.ray_encoding, num_rays]), cfg=cfg)
    env = make_env(cfg)
    obs, _ = env.reset()
    action, env_action, logp, mu, std, value = agent.select_action(obs)
    table = agent.actor.action_table
    assert torch.equal(env_action, table[action[:, 0].long()])
    assert torch.allclose(agent.actor.log_prob_action(action), logp)
    assert agent.predict_action(obs).shape == (env.num_envs, 2)


def test_masked_channels_hide_distance():
    cfg = make_cfg("observation.masked_channels=[3]", "env.room_path=rooms/one_room.json")
    num_rays = compute_num_rays(cfg.env.fov, cfg.env.ray_density)
    agent = create_ppo_agent(ray_dim=np.array([cfg.env.ray_encoding, num_rays]), cfg=cfg)
    obs = {"rays": torch.rand(2, 7, num_rays), "proprio": torch.rand(2, 2)}
    changed = {"rays": obs["rays"].clone(), "proprio": obs["proprio"]}
    changed["rays"][:, 3] = torch.rand(2, num_rays)
    assert torch.allclose(agent.embed_obs(obs), agent.embed_obs(changed))


def test_auc_is_padded_to_budget():
    from algorithms import MLPPPO

    class Stub(MLPPPO):
        def __init__(self):
            self.n_iterations, self.eval_interval, self.seed = 100, 5, 0
            self.auc, self.eval_success_curve = None, []

    early, late = Stub(), Stub()
    early.log_eval_auc(4, [0.5, 1.0, 1.0, 1.0])          # stopped after 4 of 20 evals at 1.0
    late.log_eval_auc(20, list(np.linspace(0, 1, 20)))  # crawled up over the whole budget
    assert early.budget_eval_steps() == 20
    assert early.auc > late.auc


class GoalSeekingAgent:
    """Turns towards the goal and walks: reaches it in an open room, exercising the metrics."""
    def __init__(self, env):
        self.env = env
    def predict_action(self, obs):
        rel = self.env.goal_pos - self.env.agent_pos
        bearing = torch.atan2(rel[:, 1], rel[:, 0]) - self.env.facing_direction
        bearing = torch.atan2(torch.sin(bearing), torch.cos(bearing))
        turn = (bearing / self.env._max_turn_rad).clamp(-1, 1)
        return torch.stack([turn, torch.ones_like(turn)], dim=-1)


def test_metrics_with_random_goals(tmp_path):
    cfg = make_cfg("env.room_path=rooms/one_room.json", "env.goal_mode=random", "env.goal_radius=16",
                   "algorithm.max_steps=200")
    env = make_env(cfg, num_envs=8)
    evaluate_model_on_metrics(agent=GoalSeekingAgent(env), env=env, episodes=8, nr_runs=3,
                              json_path=cfg.env.room_path, storage_path=tmp_path, backbone_type="mlp")
    cr, spl, ndtw = (np.load(tmp_path / f"{k}_values.npy") for k in ("cr", "spl", "ndtw"))
    assert (cr == 100).all()
    assert ((spl > 0.7) & (spl <= 1.0)).all()
    assert len(ndtw) == 3 and ((ndtw > 0) & (ndtw <= 1)).all()


# ---- E2 diagnostics (scripts/e2_diagnostics.py) ----

def test_junction_and_room_index():
    env = make_env(make_cfg("env.room_path=rooms/twin_rooms_L8.json", "env.goal_mode=regions"), num_envs=4)
    env.reset()
    jb = env.junction_box
    junction = torch.stack([(jb[0] + jb[1]) / 2, (jb[2] + jb[3]) / 2])
    env.agent_pos[:] = torch.stack([junction, env.goal_points[0], env.goal_points[1], env.initial_pos])
    assert env.in_junction().tolist() == [True, False, False, False]
    assert env.goal_region_index().tolist() == [-1, 0, 1, -1]


@pytest.mark.parametrize("backbone", ["lstm_backbone", "gru_backbone"])
def test_reset_at_junction_happens_once(backbone, monkeypatch):
    from scripts.e2_diagnostics import diagnostic_rollouts

    cfg = make_cfg("env.room_path=rooms/twin_rooms_L8.json", "env.goal_mode=regions", "env.colour_mode=off",
                   "algorithm.max_steps=30", f"backbone={backbone}")
    env = make_env(cfg, num_envs=6)
    num_rays = compute_num_rays(cfg.env.fov, cfg.env.ray_density)
    agent = create_ppo_agent(ray_dim=np.array([cfg.env.ray_encoding, num_rays]), cfg=cfg)
    # Spawn every episode in the junction, so it is entered on the first step
    jb = env.junction_box
    junction = torch.stack([(jb[0] + jb[1]) / 2, (jb[2] + jb[3]) / 2])
    monkeypatch.setattr(env.spawn_sampler, "sample", lambda n: junction.expand(n, 2).clone())

    reset = diagnostic_rollouts(agent, env, recurrent=True, seed=0, intervention="reset", collect_hidden=True)
    memory = diagnostic_rollouts(agent, env, recurrent=True, seed=0)
    assert reset["reached_junction"].all()
    assert (reset["n_resets"] == 1).all() and (memory["n_resets"] == 0).all()
    # Same seed -> same cues in both runs, so their outcomes are comparable episode by episode
    assert (reset["cue"] == memory["cue"]).all()
    hidden, labels, since, episode = reset["probe"]
    assert hidden.shape[1] == cfg.backbone.lstm_backbone_feature_dim and len(labels) == len(since) == len(episode)


def test_junction_distance_and_corridor_fraction():
    env = make_env(make_cfg("env.room_path=rooms/twin_rooms_L8.json", "env.goal_mode=regions"), num_envs=4)
    env.reset()
    bend, edge = env.junction_path[-2], env.junction_path[-1]
    mid = (bend + edge) / 2
    assert torch.isclose(env.junction_distance(env.initial_pos[None])[0], torch.tensor(env.junction_norm))
    # Straight corridor: the lateral offset doesn't count, 0 at the junction edge
    env.agent_pos[:] = torch.stack([mid + torch.tensor([0.0, 20.0]), mid - torch.tensor([0.0, 20.0]), edge, bend])
    d = env.junction_distance()
    assert torch.isclose(d[0], d[1]) and d[2].abs() < 1e-4
    assert torch.allclose(env.corridor_fraction(), torch.tensor([0.5, 0.5, 1.0, 0.0]), atol=1e-4)


def test_junction_shaping_telescopes_and_ignores_the_cue():
    cfg = make_cfg("env.room_path=rooms/twin_rooms_L8.json", "env.goal_mode=regions", "env.colour_mode=off",
                   "env.junction_shaping=1.0", "env.gamma=0.999", "algorithm.max_steps=1000")
    env = make_env(cfg, num_envs=16)
    env.reset()
    gamma = cfg.env.gamma
    start_potential = env.junction_potential.clone()
    assert torch.allclose(start_potential, -env.junction_distance() / env.junction_norm)

    # Discounted shaping terms add up to gamma^T Phi(s_T) - Phi(s_0), whatever the policy (no episode ends here)
    shaping_sum = torch.zeros(16)
    for t in range(40):
        action = torch.stack([torch.rand(16) * 2 - 1, torch.rand(16)], dim=-1)
        _, reward, terminated, _, _ = env.step(action)
        assert not terminated.any()
        shaping_sum += gamma ** t * (reward + 1.0 / cfg.algorithm.max_steps)
    assert torch.allclose(shaping_sum, gamma ** 40 * env.junction_potential - start_potential, atol=1e-4)

    # Same potential for both cues at the same positions
    potential = env.junction_potential_now()
    env.goal_idx = 1 - env.goal_idx
    assert torch.equal(env.junction_potential_now(), potential)

    # 0 from the first step in the junction on, also back in the corridor
    jb = env.junction_box
    corridor = env.agent_pos.clone()
    env.agent_pos[:] = torch.stack([(jb[0] + jb[1]) / 2, (jb[2] + jb[3]) / 2])
    env.step(torch.zeros(16, 2))
    env.agent_pos[:] = corridor
    assert (env.junction_potential_now() == 0).all()

    # Standing still before the junction costs more than the shaping pays, at the E2 episode length 2L + 256
    env.max_steps = 2 * 8 + 256
    env.reset()
    _, reward, _, _, _ = env.step(torch.zeros(16, 2))
    assert (reward < 0).all()


def test_mirror_pose_and_partners():
    from scripts.e2_diagnostics import mirror_pose, partners, get_pose

    env = make_env(make_cfg("env.room_path=rooms/twin_rooms_L8.json", "env.goal_mode=regions"), num_envs=3)
    env.reset()
    y0 = float(env.junction_path[-1, 1])
    mid = (env.junction_path[-2] + env.junction_path[-1]) / 2
    env.agent_pos[:] = mid + torch.tensor([[0.0, 15.0], [5.0, -10.0], [0.0, 0.0]])
    env.last_turning[:] = torch.tensor([0.3, -0.5, 0.0])
    before = get_pose(env)
    mask = torch.tensor([True, True, False])
    mirror_pose(env, mask)
    after = get_pose(env)
    assert torch.allclose(after[:2, 1], 2 * y0 - before[:2, 1]) and torch.allclose(after[:2, 0], before[:2, 0])
    assert torch.allclose(torch.sin(after[:2, 2]), -torch.sin(before[:2, 2]), atol=1e-5)
    assert torch.allclose(after[:2, 4], -before[:2, 4]) and torch.equal(after[2], before[2])
    mirror_pose(env, mask)
    assert torch.allclose(get_pose(env), before, atol=1e-4)

    cue = torch.tensor([0, 0, 1, 1])
    assert partners(cue, same=False).tolist() == [2, 2, 0, 0]
    assert partners(cue, same=True).tolist() == [1, 0, 3, 2]
    assert partners(torch.tensor([0, 0]), same=False).tolist() == [-1, -1]


@pytest.mark.parametrize("backbone", ["lstm_backbone", "gru_backbone"])
def test_swaps_copy_the_partners_state_and_pose(backbone):
    from scripts.e2_diagnostics import diagnostic_rollouts, partners

    cfg = make_cfg("env.room_path=rooms/twin_rooms_L8.json", "env.goal_mode=regions", "env.colour_mode=off",
                   "algorithm.max_steps=20", f"backbone={backbone}")
    env = make_env(cfg, num_envs=8)
    num_rays = compute_num_rays(cfg.env.fov, cfg.env.ray_density)
    agent = create_ppo_agent(ray_dim=np.array([cfg.env.ray_encoding, num_rays]), cfg=cfg)
    # A point every episode passes on its first step, so every source episode records it
    source = diagnostic_rollouts(agent, env, recurrent=True, seed=0, corridor_fraction=-100.0)
    assert source["intervened"].all() and source["point_state"][0].shape[1] == 8

    partner = partners(torch.as_tensor(source["cue"]), same=False)
    swap = diagnostic_rollouts(agent, env, recurrent=True, seed=0, intervention="swap_pose",
                               corridor_fraction=-100.0, partner=partner, source=source)
    assert np.array_equal(swap["intervened"], partner.numpy() >= 0)
    # The same seed gives the same cues, so the pairs are the ones the source saw
    assert np.array_equal(swap["cue"], source["cue"])
    for intervention in ("reset", "mirror", "swap_hidden", "swap_both"):
        out = diagnostic_rollouts(agent, env, recurrent=True, seed=0, intervention=intervention,
                                  corridor_fraction=-100.0, partner=partner, source=source)
        assert len(out["outcome"]) == 8


def test_linear_probe_and_confusion_matrix():
    from scripts.e2_diagnostics import linear_probe, confusion_matrix

    torch.manual_seed(0)
    n_eps, steps = 60, 80
    labels = torch.randint(0, 2, (n_eps,)).repeat_interleave(steps)
    since = torch.arange(steps).repeat(n_eps)
    episode = torch.arange(n_eps).repeat_interleave(steps)
    # The cue is linearly readable for 32 steps after it was seen, then forgotten
    signal = (2.0 * labels - 1.0)[:, None] * (since <= 32).float()[:, None]
    hidden = torch.randn(len(labels), 16) + 3.0 * signal * torch.ones(1, 16)
    acc = {tuple(row["bin"]): row["accuracy"] for row in linear_probe(hidden, labels, since, episode, 2, seed=0)}
    assert acc[(1, 8)] > 0.95 and acc[(9, 32)] > 0.95
    assert 0.3 < acc[(33, 64)] < 0.7

    counts = confusion_matrix(np.array([0, 0, 1, 1]), np.array([0, -1, 0, 1]), num_rooms=2)
    assert counts.tolist() == [[1, 0, 1], [1, 1, 0]]


# ---- Seed-level statistics (scripts/seed_stats.py) ----

def test_seed_stats_use_the_seed_as_unit(tmp_path):
    from scripts.seed_stats import load_per_seed, holm, probability_of_improvement, short_labels
    from scipy.stats import ttest_ind

    # 5 seeds x 10 eval runs per config: the seeds differ a lot, the runs within a seed barely
    rng = np.random.default_rng(0)
    seed_means = {"encoder:mlp_backbone:mlp_frame_stack:False": [78, 84, 91, 70, 95],
                  "encoder:mlp_backbone:lstm_frame_stack:False": [88, 92, 83, 97, 90]}
    for config, means in seed_means.items():
        for s, m in enumerate(means, start=100):
            seed_dir = tmp_path / "four_room" / config / f"seed_{s}"
            seed_dir.mkdir(parents=True)
            np.save(seed_dir / "cr_values.npy", rng.normal(m, 2.0, 10))

    per_room = load_per_seed(tmp_path, "success")
    mlp, lstm = (np.array(list(per_room["four_room"][c].values())) for c in seed_means)
    assert len(mlp) == len(lstm) == 5                       # one value per seed, not 50 runs
    assert ttest_ind(lstm, mlp, equal_var=False).pvalue > 0.05
    assert short_labels(list(seed_means)) == {c: f"backbone:{c.split('backbone:')[1].split('_')[0]}" for c in seed_means}

    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert probability_of_improvement([2, 3], [1, 3]) == pytest.approx(0.625)


def test_seed_stats_pairs_within_conditions():
    from scripts.seed_stats import comparison_pairs

    names = [f"encoder:mlp_backbone:{b}_frame_stack:False" + (f"_bptt:{w}" if b != "mlp" else "") + f"_ginfo:{g}"
             for g in ("none", "oracle") for b, w in (("mlp", None), ("lstm", 16), ("gru", 64))]
    pairs = comparison_pairs(names, reference="mlp", vary="backbone")
    assert len(pairs) == 4                                  # lstm and gru vs mlp, in each of the 2 conditions
    for a, b in pairs:
        assert "backbone:mlp" in b and a.split("ginfo:")[1] == b.split("ginfo:")[1]


def fake_seeds(sweep_dir, room, config, values, file_name="auc.npy"):
    for seed, value in enumerate(values, start=100):
        seed_dir = sweep_dir / room / config / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        np.save(seed_dir / file_name, np.array([value]))


def test_paper_figure_memory_gain_is_lstm_minus_mlp(tmp_path):
    from types import SimpleNamespace
    from scripts.paper_figures import figure_e3, paper_style
    from scripts.e4_promotion import iqm

    names = {b: f"encoder:mlp_backbone:{b}_frame_stack:False" + ("_bptt:16" if b != "mlp" else "") +
                "_ginfo:none_pinfo:none_ggrid:False" for b in ("mlp", "lstm", "gru")}
    mlp, lstm = [0.50, 0.52, 0.48, 0.51, 0.49], [0.80, 0.82, 0.78, 0.81, 0.79]
    fake_seeds(tmp_path, "four_room", names["mlp"], mlp)
    fake_seeds(tmp_path, "four_room", names["lstm"], lstm)
    fake_seeds(tmp_path, "four_room", names["gru"], [0.95] * 5)     # the trigger rule ran GRU here: it must not be the gain
    # Both at exactly 1.0: the t-test gives NaN, which must not poison the Holm correction of the other cell
    oracle = names["mlp"].replace("ginfo:none", "ginfo:oracle"), names["lstm"].replace("ginfo:none", "ginfo:oracle")
    fake_seeds(tmp_path, "four_room", oracle[0], [1.0] * 5)
    fake_seeds(tmp_path, "four_room", oracle[1], [1.0] * 5)
    fake_seeds(tmp_path, "four_room", names["mlp"].replace("ggrid:False", "ggrid:Maybe"), [0.1] * 5)  # not a condition

    paper_style()
    name, _, rows = figure_e3(tmp_path, tmp_path, {"n_boot": 50}, np.random.default_rng(0), SimpleNamespace(timestamp="t"))
    gains = {row["condition"]: row for row in rows if row["backbone"] == "lstm-mlp"}
    assert name == "fig4" and set(gains) == {"I0", "G1"}
    assert gains["I0"]["iqm"] == pytest.approx(iqm(lstm) - iqm(mlp))
    assert gains["I0"]["p_holm"] == pytest.approx(gains["I0"]["p_welch"] * 2)   # two comparisons in the family
    assert gains["I0"]["significant"] and not gains["G1"]["significant"]


def test_paper_figure_cells_refuse_two_configs_in_one_cell(tmp_path):
    from scripts.paper_figures import collect

    fake_seeds(tmp_path, "C_maze", "encoder:mlp_backbone:mlp_lr:0.1", [0.5, 0.6])
    fake_seeds(tmp_path, "C_maze", "encoder:mlp_backbone:mlp_lr:0.2", [0.7, 0.8])
    with pytest.raises(RuntimeError, match="cell"):
        collect(tmp_path, "auc", lambda fields: fields["backbone"])
    fake_seeds(tmp_path, "C_maze", "encoder:mlp_backbone:lstm_lr:0.1", [0.1])
    with pytest.warns(UserWarning, match="seeds"):
        collect(tmp_path, "auc", lambda fields: fields["lr"] + fields["backbone"])


def test_paper_figure_e2_names_come_from_the_real_storage_names(tmp_path):
    import json
    from scripts.train import run_names
    from scripts.e4_promotion import parse_storage_name
    from scripts.paper_figures import e2_key, e2_order, e2_table, memory_horizon

    expected = {"mlp": ("MLP", None), "fs": ("FS16", None), "lstm": ("LSTM", 256), "gru": ("GRU", 64)}
    overrides = {
        "mlp": ["backbone=mlp_backbone"],
        "fs": ["backbone=mlp_backbone", "observation.frame_stack.enabled=true", "observation.frame_stack.size=16"],
        "lstm": ["backbone=lstm_backbone", "backbone.bptt_window_size=256"],
        "gru": ["backbone=gru_backbone", "backbone.bptt_window_size=64"]}
    names = {key: run_names(make_cfg(*o))[1] for key, o in overrides.items()}
    assert {key: e2_key(parse_storage_name(name)) for key, name in names.items()} == expected
    assert memory_horizon("twin_rooms_L128") == 128
    assert sorted([(8, "GRU", 64), (8, "LSTM", 16), (8, "MLP", None)], key=lambda k: e2_order(*k))[0][1] == "MLP"

    # Row = cue, column = room, last column = timeout: every episode that chose a room chose the right one
    seed_dir = tmp_path / "twin_rooms_L8" / names["lstm"] / "seed_100"
    seed_dir.mkdir(parents=True)
    result = {"memory": {"success": 0.7, "wrong_room": 0.0, "timeout": 0.3, "confusion": [[3, 0, 2], [0, 4, 1]]},
              "reset_at_junction": {"success": 0.5}, "reset_drop": 0.2,
              "probe": [{"bin": [1, 8], "accuracy": 0.9}, {"bin": [9, 32], "accuracy": None}]}
    (seed_dir / "e2_diagnostics.json").write_text(json.dumps(result))
    e2_table(tmp_path, tmp_path, table_L=8)
    row = (tmp_path / "table_e2.tex").read_text().splitlines()[4]
    assert row.startswith("LSTM-256 & 0.70") and "1.00" in row and "0.90" in row    # correct room, probe at L

    # The body vs memory checks land in the csv, a check with no episode at its point (n = 0) stays empty
    result.update({"mirror": {"baseline": {"n": 10, "success": 1.0}, "mirrored": {"n": 10, "success": 0.9}},
                   "swap": {"corridor_fraction": 0.5, "hidden_other": {"n": 10, "success": 0.1},
                            "pose_other": {"n": 0}},
                   "reset_sweep": [{"corridor_fraction": 0.25, "n": 10, "success": 0.5}]})
    (seed_dir / "e2_diagnostics.json").write_text(json.dumps(result))
    e2_table(tmp_path, tmp_path, table_L=8)
    import csv
    csv_row = next(csv.DictReader(open(tmp_path / "table_e2.csv")))
    assert float(csv_row["mirror_success"]) == 0.9 and float(csv_row["swap_hidden_other"]) == 0.1
    assert float(csv_row["reset_at_0.25"]) == 0.5 and np.isnan(float(csv_row["swap_pose_other"]))


def test_paper_figure_kendall_tau():
    from scripts.paper_figures import rank_scores, rank_tau

    promotion = {"themes": {"a": {"levels": [{"config": "x", "mean_per_room": {"C_maze": 0.1, "hard_maze": 0.1}},
                                             {"config": "y", "mean_per_room": {"C_maze": 0.2, "hard_maze": 0.2}},
                                             {"config": "z", "mean_per_room": {"C_maze": 0.3, "hard_maze": 0.3}}]},
                            "b": {"levels": [{"config": "w", "mean_per_room": {"C_maze": 0.4}}]}}}
    a, b = rank_scores(promotion, "C_maze"), rank_scores(promotion, "hard_maze")
    assert rank_tau(a, b) == pytest.approx((1.0, 3))        # only the levels both rooms have
    assert rank_tau(a, {key: -value for key, value in b.items()}) == pytest.approx((-1.0, 3))
    assert np.isnan(rank_tau({"x": 1.0, "y": 2.0}, {"x": 1.0, "y": 2.0})[0])
