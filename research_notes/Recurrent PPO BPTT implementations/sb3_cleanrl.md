# SB3-contrib RecurrentPPO and CleanRL LSTM PPO: BPTT and sequence minibatching

Versions read, as raw source from git, not from docs or blogs:

- `Stable-Baselines-Team/stable-baselines3-contrib` at `a8af636` (tag v2.9.2, 2026-10-26)
- `vwxyzjn/cleanrl` at `fe8d8a0` (master, 2026-04-20; the last tag, v1.0.0, is much older)
- `DLR-RM/rl-baselines3-zoo` at `61956d9` (tag v2.9.2, 2026-10-26)

All line numbers below refer to these commits. Issue threads were read through `gh`.

Note: RecurrentPPO lives in sb3-contrib, not in SB3 core. "SB3 has LSTM PPO" is only true of the contrib package ([ppo_recurrent.py](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/a8af63682070df0bda9707400b2e971055600910/sb3_contrib/ppo_recurrent/ppo_recurrent.py)).

## In sb3_contrib, what determines sequence length, and can it be set?

### Takeaway
The user's claim is correct in the sense that matters: there is no BPTT or sequence-length parameter. It needs one correction. The length is not free: a sequence is at most `min(batch_size, n_steps, steps left in the episode)`, so `batch_size` below `n_steps` works as an undocumented hard cap. The maintainer's own answer names only `n_steps` and the episode as bounds.

### Cited Findings
- The `RecurrentPPO.__init__` signature has no sequence, BPTT, or truncation argument. Its batching arguments are `n_steps=128`, `batch_size=128`, `n_epochs=10`, and the docstring gives `batch_size` no meaning beyond minibatch size — [ppo_recurrent.py L74-L100, L36-L39](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/a8af63682070df0bda9707400b2e971055600910/sb3_contrib/ppo_recurrent/ppo_recurrent.py#L74-L100)
- `RecurrentRolloutBuffer.__init__` has no such argument either; it takes only buffer size, spaces, hidden-state shape, device, `gae_lambda`, `gamma`, `n_envs` — [buffers.py L116-L126](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/a8af63682070df0bda9707400b2e971055600910/sb3_contrib/common/recurrent/buffers.py#L116-L126)
- Sequence starts are computed inside each minibatch only, as the OR of `episode_starts` and `env_change`, with the minibatch's first index forced to be a start — [buffers.py L84-L88](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/a8af63682070df0bda9707400b2e971055600910/sb3_contrib/common/recurrent/buffers.py#L84-L88)
- A minibatch is `batch_size` consecutive positions of a rotated flat index array, flattened env by env, so a sequence cannot be longer than `batch_size`, and `env_change` cuts it at `n_steps` — [buffers.py L182-L199, L159-L161](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/a8af63682070df0bda9707400b2e971055600910/sb3_contrib/common/recurrent/buffers.py#L182-L199)
- Maintainer answer to a direct request for a 32-step BPTT option: the code uses as many steps as possible until a new episode, so at most `n_steps`, also bounded by episode length; it is not really settable, except that `n_steps` is an upper bound — [issue #110](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/issues/110). This omits `batch_size`, which the source shows is also a cap.
- Maintainer on the design: the minibatch of `batch_size` transitions is kept constant so the learning rate need not be retuned, and if an episode is long enough then `n_seq=1` — [issue #103](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/issues/103). Rationale repeated in [issue #113](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/issues/113).
- Gradient steps per `train()` call are `n_epochs * ceil(n_steps * n_envs / batch_size)` from the `while start_idx < ...` loop run once per epoch, with no dependence on sequence length. `target_kl` can stop this early — [buffers.py L195-L199](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/a8af63682070df0bda9707400b2e971055600910/sb3_contrib/common/recurrent/buffers.py#L195-L199); [ppo_recurrent.py L333-L336, L406-L410](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/a8af63682070df0bda9707400b2e971055600910/sb3_contrib/ppo_recurrent/ppo_recurrent.py#L333-L336)
- Logging trap: `self._n_updates += 1` runs once per epoch, so `train/n_updates` counts epochs, not gradient steps — [ppo_recurrent.py L419, L436](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/a8af63682070df0bda9707400b2e971055600910/sb3_contrib/ppo_recurrent/ppo_recurrent.py#L419)

### Inferences
Numbers below come from a re-typed numpy copy of `get()` and `create_sequencers()`, run for 40 passes. It is not the live package. Episode lengths are geometric with the stated mean, which is an assumption, not data from the user's env.

| n_steps × n_envs, batch_size, mean episode | minibatches per epoch | mean sequences per minibatch | max length | step-weighted median length | padded tensor / batch_size |
|---|---|---|---|---|---|
| 128×1, 128, no resets | 1 | 1.98 | 128 | 93 | 1.49 |
| 128×8, 256, no resets | 4 | 3.00 | 128 | 126 | 1.50 |
| 256×256, 8192, no resets | 8 | 33.0 | 256 | 256 | 1.03 |
| 256×256, 8192, ep 100 | 8 | 114 | 256 | 118 | 3.56 |
| 256×256, 8192, ep 30 | 8 | 305 | 251 | 44 | 6.04 |
| 256×256, 256, no resets | 256 | 1.98 | 256 | 192 | 1.51 |
| 256×256, 64, no resets | 1024 | 1.23 | 64 | 64 | 1.11 |
| 256×256, 32, no resets | 2048 | 1.11 | 32 | 32 | 1.06 |

- A window can be forced with `batch_size < n_steps`, but this couples it to the minibatch. Setting `batch_size=64` gives a 64-step cap with only about one trajectory per gradient step, from one env. There is no way to ask for "window w, k trajectories per minibatch". The user's separate window and envs-per-minibatch are more expressive here.
- Sequence length in SB3 is mostly a result of episode length, not a design choice: 118 steps at mean episode 100 and 44 at mean episode 30, in the 8192 row.
- Because `batch_size` counts transitions, longer sequences do not cost gradient steps in SB3. In the user's repo they do, because a minibatch's transition count grows with the window. Holding `envs_per_minibatch × window` constant would reproduce the SB3 property (see the CleanRL section).

### Gaps
- The sequence-length table was not cross-checked by running the installed sb3-contrib package.

## How does sb3_contrib split, pad, and mask sequences, and how far does the gradient flow?

### Takeaway
The "split and swap" is a single rotation of the flat index array, redone each epoch, which moves every minibatch boundary. Sequences are cut at episode starts, env changes and minibatch boundaries, then zero-padded on the right and masked in every loss term. Gradient flows through the whole of each sequence but never across a cut, so it spans at most one rollout segment.

### Cited Findings
- Before sampling, arrays are flattened from `(n_steps, n_envs, ...)` to `(n_envs * n_steps, ...)`, keeping each env's steps contiguous — [buffers.py L159-L175](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/a8af63682070df0bda9707400b2e971055600910/sb3_contrib/common/recurrent/buffers.py#L159-L175)
- Rotation: `split_index = np.random.randint(N)`, then `indices = concat(indices[split_index:], indices[:split_index])`, commented as a trick to shuffle a bit while keeping sequence order. There is no env permutation — [buffers.py L184-L188](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/a8af63682070df0bda9707400b2e971055600910/sb3_contrib/common/recurrent/buffers.py#L184-L188)
- `get()` is called once per epoch, so each epoch gets a new rotation — [ppo_recurrent.py L333-L336](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/a8af63682070df0bda9707400b2e971055600910/sb3_contrib/ppo_recurrent/ppo_recurrent.py#L333-L336)
- Maintainer describes the split as optional randomness when sampling over multiple epochs — [issue #284 comment](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/issues/284#issuecomment-2766526133)
- `env_change[0, :] = 1.0` flags the first timestep of each env as a sequence start — [buffers.py L190-L193](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/a8af63682070df0bda9707400b2e971055600910/sb3_contrib/common/recurrent/buffers.py#L190-L193)
- Padding uses `th.nn.utils.rnn.pad_sequence(..., batch_first=True, padding_value=0.0)` to `(n_seq, max_length, ...)`, then reshapes to `padded_batch_size = n_seq * max_length` — [buffers.py L36-L40, L213-L215, L235-L236](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/a8af63682070df0bda9707400b2e971055600910/sb3_contrib/common/recurrent/buffers.py#L36-L40)
- `mask` is the padded ones array. Policy, value, entropy, KL, and clip-fraction terms are all indexed by it — [buffers.py L243](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/a8af63682070df0bda9707400b2e971055600910/sb3_contrib/common/recurrent/buffers.py#L243); [ppo_recurrent.py L343-L403](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/a8af63682070df0bda9707400b2e971055600910/sb3_contrib/ppo_recurrent/ppo_recurrent.py#L343-L403)
- Confirmed in the thread: padded tensors exceed `batch_size`, while `observations[mask]` stays equal to it — [issue #113](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/issues/113)
- `_process_sequence` reshapes to `(max_length, n_seq, features)`. It passes the whole tensor to `nn.LSTM` if no `episode_starts` are set, otherwise it loops step by step, multiplying the state by `1 - episode_start` — [policies.py L185-L206](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/a8af63682070df0bda9707400b2e971055600910/sb3_contrib/common/recurrent/policies.py#L185-L206)
- Maintainer confirms backprop runs over the sampled sequences rather than one step, answering a question about whether only a single RNN step was backpropagated — [PR #53 comment](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/pull/53#issuecomment-1097297131); [issue #103](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/issues/103)
- Maintainer and code disagree on one detail. The #284 comment says the hidden state must be zeroed at episode starts or env-change events. In the source, `env_change` goes only to `create_sequencers`, and only `episode_starts` reaches the reset — [buffers.py L208-L210, L242](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/a8af63682070df0bda9707400b2e971055600910/sb3_contrib/common/recurrent/buffers.py#L208-L210); [issue #284 comment](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/issues/284#issuecomment-2766526133). I trust the code: an env-change sequence starts from its stored state, not zeros.
- PR #118 (draft, open since 2022-11-28, last updated 2024-01-10) adds an optional `whole_sequence` batching mode. Its author reported 2.6x to 9x speedups, but it is unmerged and does not add a window either — [PR #118](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/pull/118); [issue #113](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/issues/113)

### Inferences
- Rotation matters more for moving the cuts than for shuffling. Minibatch boundaries land at different offsets each epoch, so SB3 has no fixed windows, unlike the user's scheme. But the same neighbouring envs stay together: envs are never permuted, so at 8192 a minibatch always holds about 32 adjacent env indices.
- Episode resets never happen inside a sequence, since every episode start begins a new one. The step-by-step branch therefore acts only at position 0, and the loop exists because a sequence start may need zeroing.
- Gradient never crosses an episode boundary, a minibatch boundary, or a rollout boundary. An episode spanning two rollouts is truncated at the rollout edge.
- The padded forward pass costs more than `batch_size` suggests, because the feature extractor also runs on padded zeros. With short episodes this is several times the nominal batch (6x in the simulated ep-30 row). That is relevant if the user tries this scheme with a CNN encoder.

### Gaps
- Upstream's reason for rotating rather than permuting envs: the maintainer calls it added randomness, but I found no statement comparing it with env shuffling.

## Which hidden state starts each sequence, and how is advantage normalization done?

### Takeaway
SB3 stores the pre-step LSTM states at every timestep and starts each sequence from the stored value, with no recomputation and no burn-in. This is the same family as the user's repo. Advantage normalization is per minibatch in both SB3 and CleanRL. SB3 computes the statistics over real steps only.

### Cited Findings
- Actor and critic hidden and cell states are stored at every step in arrays of shape `(n_steps, num_layers, n_envs, hidden_size)` — [buffers.py L133-L145](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/a8af63682070df0bda9707400b2e971055600910/sb3_contrib/common/recurrent/buffers.py#L133-L145); [ppo_recurrent.py L174](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/a8af63682070df0bda9707400b2e971055600910/sb3_contrib/ppo_recurrent/ppo_recurrent.py#L174)
- The value stored at step t is `self._last_lstm_states`, the state before step t, which is updated only after `add()` — [ppo_recurrent.py L287-L299](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/a8af63682070df0bda9707400b2e971055600910/sb3_contrib/ppo_recurrent/ppo_recurrent.py#L287-L299)
- Each sequence takes the stored state at its first index, converted from numpy, so it is a constant for autograd — [buffers.py L216-L231](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/a8af63682070df0bda9707400b2e971055600910/sb3_contrib/common/recurrent/buffers.py#L216-L231)
- Collected states come under `th.no_grad()` from the policy at collection time, and there is no recomputation step in `train()` — [ppo_recurrent.py L238-L242, L333-L350](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/a8af63682070df0bda9707400b2e971055600910/sb3_contrib/ppo_recurrent/ppo_recurrent.py#L238-L242)
- Hidden state carries over between rollouts via `lstm_states = deepcopy(self._last_lstm_states)` — [ppo_recurrent.py L231](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/a8af63682070df0bda9707400b2e971055600910/sb3_contrib/ppo_recurrent/ppo_recurrent.py#L231)
- SB3 normalizes per minibatch with statistics on `advantages[mask]`, controlled by `normalize_advantage=True` — [ppo_recurrent.py L353-L356](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/a8af63682070df0bda9707400b2e971055600910/sb3_contrib/ppo_recurrent/ppo_recurrent.py#L353-L356)
- CleanRL normalizes per minibatch over `b_advantages[mb_inds]`, with `norm_adv=True` by default and no mask, because it has no padding — [ppo_atari_lstm.py L323-L325](https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl/ppo_atari_lstm.py#L323-L325)

### Inferences
- Staleness in SB3 grows with the epoch: later epochs replay states computed under older weights, and any sequence starting mid-rollout inherits one. CleanRL has this only at the rollout's first step, since it replays everything after that under current weights. Shorter windows therefore carry more stale starts, in both SB3 and the user's repo.
- The user's first-minibatch ratio check (`algorithms/ppo.py` L764) should hold in SB3's design too, as stored state and old weights match on the first update.

### Gaps
- I found no measurement in either project of how much stale initial states matter, and no burn-in option in either code base.

## How does CleanRL build recurrent minibatches, and how do gradient steps scale?

### Takeaway
CleanRL treats each env's whole rollout as one sequence, shuffles envs into minibatches, and has no truncation parameter. BPTT length equals `num_steps` (128), reduced only by episode ends inside the rollout. Gradient steps per iteration are `update_epochs × num_minibatches` (16), independent of `num_steps`. The user's `window=num_steps` case is this scheme.

### Cited Findings
- `ppo_atari_lstm.py` is the only recurrent PPO script. A grep for `LSTM|GRU|lstm` across `cleanrl/*.py` finds only it and `pqn_atari_envpool_lstm.py`, which is a Q-learning script. The file's `Args` has no sequence or BPTT field — [ppo_atari_lstm.py L44-L86](https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl/ppo_atari_lstm.py#L44-L86)
- Defaults: `num_envs=8`, `num_steps=128`, `num_minibatches=4`, `update_epochs=4`, `clip_coef=0.1`, `ent_coef=0.01`, LSTM hidden size 128 — [ppo_atari_lstm.py L51-L71, L131](https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl/ppo_atari_lstm.py#L51-L71)
- Minibatching requires `num_envs % num_minibatches == 0`, sets `envsperbatch = num_envs // num_minibatches`, reshuffles `envinds` every epoch, and takes all steps for those envs via `flatinds[:, mbenvinds].ravel()` — [ppo_atari_lstm.py L296-L306](https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl/ppo_atari_lstm.py#L296-L306)
- One state per env per iteration is saved, `initial_lstm_state`, cloned before the rollout, and indexed by `mbenvinds` at training — [ppo_atari_lstm.py L234, L308-L313](https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl/ppo_atari_lstm.py#L234)
- `get_states` steps through the sequence in a Python loop, multiplying the state by `1.0 - d` at every step. No splitting, padding, or mask is used — [ppo_atari_lstm.py L140-L158](https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl/ppo_atari_lstm.py#L140-L158)
- `dones[step]` holds the flag from the previous step, so `d=1` means "this step starts an episode" — [ppo_atari_lstm.py L243-L244, L255](https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl/ppo_atari_lstm.py#L243-L244)
- Changing `num_steps` also changes `batch_size` and `num_iterations` — [ppo_atari_lstm.py L175-L177](https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl/ppo_atari_lstm.py#L175-L177)
- `target_kl` is checked once per epoch, so it can stop at an epoch boundary only — [ppo_atari_lstm.py L355-L356](https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl/ppo_atari_lstm.py#L355-L356)
- `pqn_atari_envpool_lstm.py` uses the same env-index batching and `initial_lstm_state` replay — [pqn_atari_envpool_lstm.py L311-L321](https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl/pqn_atari_envpool_lstm.py#L311-L321)
- `ppo_trxl.py` is a transformer, not an RNN. It has a settable `trxl_memory_length=119`, and because it stores memory it can shuffle single transitions with `torch.randperm` — [ppo_trxl.py L89, L568](https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl/ppo_trxl/ppo_trxl.py#L89)
- CleanRL's docs base the script on five LSTM details from the ICLR blog, including sequential rollouts in minibatches and reconstructing LSTM states during training, both tied to OpenAI baselines `a2c/utils.py#L81` — [docs/rl-algorithms/ppo.md L443-L449](https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/docs/rl-algorithms/ppo.md)
- The ICLR blog's LSTM section also describes sequential minibatches per sub-environment and replay from a saved initial state, and states no separate sequence length. It reuses the Atari settings with frame stacking removed — [ICLR blog](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/). I read this page through a summarizing fetch, not line by line.

### Inferences
- CleanRL truncates at `num_steps` as an implicit fixed window with no choice of length apart from the rollout itself.
- At defaults, 2 envs × 128 steps per minibatch and 16 gradient steps per iteration. With `total_timesteps=1e7`, that is 9765 iterations × 16, or about 156k updates.
- Multiplying by `1.0 - d = 0` zeroes both state and gradient, so gradient does not cross episodes here either, and sequences are cut in effect at dones without being split.
- Because all steps after the first are recomputed under current weights, only the first step of the rollout relies on a stored state, unlike SB3 and the user's windowed mode.

Comparison, using a 256×256 rollout. `n_epochs=10` is illustrative, since the assignment did not state it and I did not find it in `configs/base.yaml`:

| | SB3-contrib | CleanRL | User's repo |
|---|---|---|---|
| Unit of `minibatch` | transitions (`batch_size`) | envs × full rollout | envs × window |
| Window parameter | none; `min(batch_size, n_steps, episode)` | none; `num_steps` | yes, `bptt_window` |
| Window placement | moves each epoch | fixed, whole rollout | fixed, consecutive |
| Stored states used | every step | rollout start only | every window start |
| Env shuffling | no, rotation only | yes, each epoch | yes |
| Burn-in | none | none | none |
| Grad steps/iter | `epochs × ceil(N/B)` | `epochs × num_minibatches` | `(E/e) × (T/w) × epochs` |

User's formula with 32 envs per minibatch (`/Users/emers/projects/semproj-rl-comparison/algorithms/ppo.py` L690-L691 confirms the windows):

| window | gradient steps/iter | transitions per step |
|---|---|---|
| 8 | 2560 | 256 |
| 32 | 640 | 1024 |
| 64 | 320 | 2048 |
| 128 | 160 | 4096 |
| 256 | 80 | 8192 |

- The `window=256` row is CleanRL's scheme with `num_envs=256, num_minibatches=8`, and gives the same 80 steps.
- Neither library has the user's 32:1 update spread across windows. In SB3 the steps are fixed by `batch_size`, in CleanRL by `num_minibatches`. Comparing windows at fixed envs-per-minibatch thus varies update count and gradient variance along with BPTT length. Holding `envs_per_minibatch × window` constant isolates the window, though with 256 envs it cannot go below `window=32` at 8192 transitions per minibatch unless windows, rather than envs, are shuffled.

### Gaps
- Whether `ppo_trxl.py` backpropagates through stored memories was not checked. Other CleanRL docs matching the LSTM grep (`ppg.md`, `overview.md`, etc.) were not read.
- The last 1074 characters of the ICLR blog page were not read. They are likely references, but this is unverified.

## What settings are recommended, and has a sequence-length parameter been discussed?

### Takeaway
Neither project documents a recommended window, because neither has one. The tuned zoo configs imply caps of 32 to 256 steps, and several are set by `batch_size`, not `n_steps`. A truncation parameter was asked about twice in sb3-contrib without being added, and I found no such proposal for CleanRL.

### Cited Findings
Values below are from [hyperparams/ppo_lstm.yml](https://github.com/DLR-RM/rl-baselines3-zoo/blob/61956d90d9f8ac126043ed4024efa12a462d97e1/hyperparams/ppo_lstm.yml). The cap and steps columns are my arithmetic. `PendulumNoVel-v1` sets no `batch_size`, so the constructor default of 128 applies.

| Entry (zoo label) | n_envs | n_steps | batch_size | n_epochs | length cap | gradient steps/iter |
|---|---|---|---|---|---|---|
| atari (default block) | 8 | 128 | 256 | 4 | 128 | 16 |
| CartPoleNoVel-v1 (Tuned) | 8 | 32 | 256 | 20 | 32 | 20 |
| PendulumNoVel-v1 (Tuned) | 4 | 1024 | 128 | 10 | 128 | 320 |
| LunarLanderNoVel-v2 (Tuned) | 32 | 512 | 128 | 4 | 128 | 512 |
| MountainCarContinuousNoVel-v0 (Tuned) | 8 | 1024 | 256 | 10 | 256 | 320 |
| BipedalWalker-v3 | 32 | 256 | 256 | 10 | 256 | 320 |
| CarRacing-v3 | 8 | 512 | 128 | 10 | 128 | 320 |

- The RecurrentPPO docs page has no text on sequence length, BPTT, or `batch_size` semantics. It lists zoo replication commands and links a benchmark report on masked-velocity envs — [docs/modules/ppo_recurrent.md](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/a8af63682070df0bda9707400b2e971055600910/docs/modules/ppo_recurrent.md)
- The maintainer's advice in the performance discussion is to try PPO with frame stacking before LSTM, calling the LSTM slowdown expected — [issue #113](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/issues/113)
- CleanRL's docs recommend no window; they present the script's defaults and benchmark it on Atari with frame stacking removed — [docs/rl-algorithms/ppo.md L443-L451](https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/docs/rl-algorithms/ppo.md)
- [PR #53 comment by andrewwarrington, 2022-04-08](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/pull/53#issuecomment-1093361537): offered to add `RECURRENT_SEQ_LEN`-style truncation and asked whether t-BPTT helps. The maintainer's [reply](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/pull/53#issuecomment-1097297131) corrected the premise; the replies I read did not take up the parameter.
- [Issue #110 (closed)](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/issues/110): requested a settable BPTT length, for instance 32. The maintainer asked for the motivation and no parameter resulted. The requester had found the option in `torch-ac`.
- [Issue #303 (open, 2025-07-22)](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/issues/303): asks for fixed-length sequence sampling in place of the split-and-pad logic. The only reply links to #284.
- [Issue #113 (open)](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/issues/113) and [PR #118 (draft)](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/pull/118): whole-episode batching. Here the maintainer notes `batch_size` would then count sequences of varying length, which is unhelpful for the learning rate, and that he wants it optional if it is added.
- [Issue #246 (open)](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/issues/246) and [issue #103](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/issues/103) both show users misreading the step-by-step code as single-step BPTT.
- [CleanRL issue #276 (open, 2022)](https://github.com/vwxyzjn/cleanrl/issues/276): proposes done-splitting with padding for a reported 4x speedup. This is a speed change, not a truncation option. The owner asked for a JAX comparison first.
- Issue #113 also records the SB2 approach: no padding, with constraints on steps and minibatches that keep sequence length constant — [issue #113](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/issues/113)

### Inferences
- Four of the seven listed configs set the cap by `batch_size`, not `n_steps`. So the #110 answer, which bounds length by `n_steps`, overstates length for those. For example, `PendulumNoVel-v1` collects 1024 steps per env but never backpropagates more than 128.
- Those caps and steps are a by-product of tuning over `batch_size`, not evidence for a preferred window. The zoo files do not comment on it.

### Gaps
- `gh search` found no CleanRL issue for "bptt" or "truncated lstm", and no result for "ppo_atari_lstm" even though #276 names that file, so keyword search here is clearly incomplete. "No proposal found" is not proof of absence.
- The W&B benchmark report and SB3 core's documentation on framestack versus LSTM were not read. The frame-stack advice above is known only via the maintainer's comment.
- The zoo labels some entries "Tuned". Whether tuning covered `batch_size` or `n_steps` for these specifically was not checked.
