# Configurable BPTT windows for recurrent PPO in RL libraries other than SB3 and CleanRL

Method note: every claim below comes from source read in shallow clones made on 2026-11-07, unless it is listed under Gaps. "HEAD" means the default branch at the commit linked in each source URL, not a PyPI release. Only RLlib was also compared against release tags. No library was executed; update counts are read from code and computed by hand. Terms: L = window or sequence length, T = rollout steps per env, N = envs, D = N x T = rollout timesteps, S = minibatch size in timesteps, K = minibatches per epoch, E = epochs. "Stale" means the hidden state stored during collection is reused unchanged across all epochs.

## RLlib (the user's "Raylib"): how is max_seq_len implemented, and does minibatch size count timesteps or sequences?

### Takeaway
On the current API stack `minibatch_size` counts timesteps, not sequences: RLlib converts it to floor(S / mean sequence length) sequences, so the update count stays near E x D / S whatever `max_seq_len` is. That has not always held: ray-2.9.0 sliced `minibatch_size` rows of B, meaning sequences, and ray-2.40.0 still did so for the hybrid path without the new EnvRunners. PPO uses stale stored states and has no burn-in.

### Cited Findings
Versions checked: master 891a5a8 (3.0.0.dev0, 2026-11-07), where RLlib now lives under `python/ray/rllib/`, and tags ray-2.60.0 (latest release, 2026-11-04), ray-2.40.0, ray-2.10.0 and ray-2.9.0, where it lives under `rllib/`.

Parameters and defaults
- `DefaultModelConfig.max_seq_len: int = 20`, `use_lstm = False`, `lstm_cell_size = 256`, `lstm_use_prev_action = False`, `lstm_use_prev_reward = False`. The same `max_seq_len` default is at line 188 in ray-2.60.0 — [default_model_config.py L185-192](https://github.com/ray-project/ray/blob/891a5a8946fad05062daebc67bc04da3d78e78c0/python/ray/rllib/core/rl_module/default_model_config.py#L185-L192); [ray-2.60.0](https://github.com/ray-project/ray/blob/ray-2.60.0/rllib/core/rl_module/default_model_config.py#L188)
- Set it with `config.rl_module(model_config=DefaultModelConfig(max_seq_len=...))`, or `model_config={"max_seq_len": ...}` for custom modules — [connector-v2.md L165-176](https://github.com/ray-project/ray/blob/891a5a8946fad05062daebc67bc04da3d78e78c0/doc/source/rllib/connector-v2.md#L165-L176)
- Old-stack dict `MODEL_DEFAULTS` has `max_seq_len: 20`; `lstm_use_prev_action_reward` is marked `DEPRECATED_VALUE` — [models/catalog.py L80-84, L113](https://github.com/ray-project/ray/blob/891a5a8946fad05062daebc67bc04da3d78e78c0/python/ray/rllib/models/catalog.py#L80-L113)
- PPO defaults: `train_batch_size = 4000`, `num_epochs = 30`, `minibatch_size = 128`, `shuffle_batch_per_epoch = True` — [ppo.py L116-123](https://github.com/ray-project/ray/blob/891a5a8946fad05062daebc67bc04da3d78e78c0/python/ray/rllib/algorithms/ppo/ppo.py#L116-L123); same values in [ray-2.60.0 L126-133](https://github.com/ray-project/ray/blob/ray-2.60.0/rllib/algorithms/ppo/ppo.py#L126-L133)
- Deprecated: `sgd_minibatch_size` (PPO config holds `DEPRECATED_VALUE`) — [ppo.py L144](https://github.com/ray-project/ray/blob/891a5a8946fad05062daebc67bc04da3d78e78c0/python/ray/rllib/algorithms/ppo/ppo.py#L144). Deprecated: `num_sgd_iter`, replaced by `num_epochs` — [algorithm_config.py L2675-2681](https://github.com/ray-project/ray/blob/891a5a8946fad05062daebc67bc04da3d78e78c0/python/ray/rllib/algorithms/algorithm_config.py#L2675-L2681). `shuffle_sequences` survives only in a comment; the live knob is `shuffle_batch_per_epoch` — [algorithm_config.py L392-395](https://github.com/ray-project/ray/blob/891a5a8946fad05062daebc67bc04da3d78e78c0/python/ray/rllib/algorithms/algorithm_config.py#L392-L395)
- New stack is the default (`enable_rl_module_and_learner = True`, `enable_env_runner_and_connector_v2 = True`) — [algorithm_config.py L451-452](https://github.com/ray-project/ray/blob/891a5a8946fad05062daebc67bc04da3d78e78c0/python/ray/rllib/algorithms/algorithm_config.py#L451-L452). The old path `_training_step_old_api_stack` still exists — [ppo.py L392-395](https://github.com/ray-project/ray/blob/891a5a8946fad05062daebc67bc04da3d78e78c0/python/ray/rllib/algorithms/ppo/ppo.py#L392-L395)

`lstm_use_prev_action`
- A grep of non-test RLlib source finds readers only in old-stack code (`models/catalog.py`, `models/torch/recurrent_net.py`, `models/tf/recurrent_net.py`); in new-stack `core/` and `connectors/` the field is declared in [default_model_config.py L191](https://github.com/ray-project/ray/blob/891a5a8946fad05062daebc67bc04da3d78e78c0/python/ray/rllib/core/rl_module/default_model_config.py#L191) and never read. The old-stack reader is [recurrent_net.py L153](https://github.com/ray-project/ray/blob/891a5a8946fad05062daebc67bc04da3d78e78c0/python/ray/rllib/models/torch/recurrent_net.py#L153)
- A separate env-to-module connector exists for this purpose and has a dedicated example; `core/`, `algorithms/ppo` and `algorithm_config.py` do not reference it, so it is not added automatically — [prev_actions_prev_rewards.py](https://github.com/ray-project/ray/blob/891a5a8946fad05062daebc67bc04da3d78e78c0/python/ray/rllib/connectors/env_to_module/prev_actions_prev_rewards.py); [example](https://github.com/ray-project/ray/blob/891a5a8946fad05062daebc67bc04da3d78e78c0/python/ray/rllib/examples/connectors/prev_actions_prev_rewards.py)

Sequence construction and zero padding (new stack)
- The Learner pipeline order is `AddObservationsFromEpisodesToBatch`, `AddColumnsFromEpisodesToTrainBatch`, `AddTimeDimToBatchAndZeroPad`, `AddStatesFromEpisodesToBatch`, `BatchIndividualItems`, `NumpyToTensor` — [add_time_dim_to_batch_and_zero_pad.py L42-51](https://github.com/ray-project/ray/blob/891a5a8946fad05062daebc67bc04da3d78e78c0/python/ray/rllib/connectors/common/add_time_dim_to_batch_and_zero_pad.py#L42-L51)
- Each episode chunk is cut into rows of length T = `max_seq_len` via `split_and_zero_pad`, then gets `SEQ_LENS` and `LOSS_MASK` columns. Windows restart at every episode or episode-chunk start, so the sequence count depends on episode ends — [add_time_dim_to_batch_and_zero_pad.py L214-285](https://github.com/ray-project/ray/blob/891a5a8946fad05062daebc67bc04da3d78e78c0/python/ray/rllib/connectors/common/add_time_dim_to_batch_and_zero_pad.py#L214-L285); [create_mask_and_seq_lens](https://github.com/ray-project/ray/blob/891a5a8946fad05062daebc67bc04da3d78e78c0/python/ray/rllib/utils/postprocessing/zero_padding.py#L12-L47)
- PPO losses are means over `LOSS_MASK`-valid entries only, so padding does not dilute the loss — [ppo_torch_learner.py L50-58](https://github.com/ray-project/ray/blob/891a5a8946fad05062daebc67bc04da3d78e78c0/python/ray/rllib/algorithms/ppo/torch/ppo_torch_learner.py#L50-L58)
- One artificial timestep is appended to each episode for value bootstrapping, with `loss_mask=False` on it — [add_one_ts_to_episodes_and_truncate.py L146-150](https://github.com/ray-project/ray/blob/891a5a8946fad05062daebc67bc04da3d78e78c0/python/ray/rllib/connectors/learner/add_one_ts_to_episodes_and_truncate.py#L146-L150)

`state_in` storage
- Stale stored state: the Learner connector takes `[init_or_lookback] + STATE_OUT[:-1]` and keeps every `max_seq_len`-th entry. The prepended state is the module's initial state if the episode starts at t=0, otherwise the previous chunk's last `STATE_OUT`. If no `STATE_OUT` exists (offline data) it repeats the initial state — [add_states_from_episodes_to_batch.py L240-297](https://github.com/ray-project/ray/blob/891a5a8946fad05062daebc67bc04da3d78e78c0/python/ray/rllib/connectors/common/add_states_from_episodes_to_batch.py#L240-L297)
- Burn-in: no PPO option found. `burn_in_len` (default 0) is a DQN setting — [dqn.py L174](https://github.com/ray-project/ray/blob/891a5a8946fad05062daebc67bc04da3d78e78c0/python/ray/rllib/algorithms/dqn/dqn.py#L174); [dqn_torch_learner.py L53-55](https://github.com/ray-project/ray/blob/891a5a8946fad05062daebc67bc04da3d78e78c0/python/ray/rllib/algorithms/dqn/torch/dqn_torch_learner.py#L53-L55). The other hits are DreamerV3 and old replay-buffer config (`replay_burn_in`, `replay_zero_init_states`) — [replay_buffers/utils.py L254-286](https://github.com/ray-project/ray/blob/891a5a8946fad05062daebc67bc04da3d78e78c0/python/ray/rllib/utils/replay_buffers/utils.py#L254-L286)

Minibatch semantics (the key question)
- `MiniBatchCyclicIterator._len_and_step` returns (n, step). For stateful modules n = number of sequences and step = `int(n * (minibatch_size / len(module_batch)))`; if step is 0, master raises `ValueError` — [minibatch_utils.py L113-148](https://github.com/ray-project/ray/blob/891a5a8946fad05062daebc67bc04da3d78e78c0/python/ray/rllib/utils/minibatch_utils.py#L113-L148)
- `len(SampleBatch)` is `count`, computed as `sum(seq_lens)` when `SEQ_LENS` is present, so the denominator is unpadded timesteps — [sample_batch.py L34-59, L263, L272-274](https://github.com/ray-project/ray/blob/891a5a8946fad05062daebc67bc04da3d78e78c0/python/ray/rllib/policy/sample_batch.py#L34-L59)
- Total minibatches = `ceil(num_epochs * n / step)` — [minibatch_utils.py L91-110](https://github.com/ray-project/ray/blob/891a5a8946fad05062daebc67bc04da3d78e78c0/python/ray/rllib/utils/minibatch_utils.py#L91-L110)
- The Learner enables B-dimension slicing for any stateful module or batch with `seq_lens` — [learner.py L1836-1868](https://github.com/ray-project/ray/blob/891a5a8946fad05062daebc67bc04da3d78e78c0/python/ray/rllib/core/learner/learner.py#L1836-L1868). Shuffling acts along B only when there is a time rank — [learner.py L1139-1144](https://github.com/ray-project/ray/blob/891a5a8946fad05062daebc67bc04da3d78e78c0/python/ray/rllib/core/learner/learner.py#L1139-L1144)
- Version history: ray-2.9.0 used `n_steps = self._minibatch_size` with no conversion in the B-sliced branch, so it took that many sequences — [ray-2.9.0 minibatch_utils.py](https://github.com/ray-project/ray/blob/ray-2.9.0/rllib/utils/minibatch_utils.py). The conversion appears from ray-2.10.0 onward in the tags sampled (2.10, 2.12, 2.20, 2.30, 2.35, 2.38, 2.40, 2.44, 2.60) — [ray-2.10.0](https://github.com/ray-project/ray/blob/ray-2.10.0/rllib/utils/minibatch_utils.py). In ray-2.40.0 it was conditional on `self._uses_new_env_runners` — [ray-2.40.0 L132-147](https://github.com/ray-project/ray/blob/ray-2.40.0/rllib/utils/minibatch_utils.py#L132-L147). In ray-2.60.0 it is unconditional, and `n_steps` can round to 0 with no `ValueError` in that code — [ray-2.60.0 L118-145](https://github.com/ray-project/ray/blob/ray-2.60.0/rllib/utils/minibatch_utils.py#L118-L145)
- A `TODO (sven): Fix this bug for LSTMs` comment still sits above this code on master, even though the following lines implement the conversion — [minibatch_utils.py L161-171](https://github.com/ray-project/ray/blob/891a5a8946fad05062daebc67bc04da3d78e78c0/python/ray/rllib/utils/minibatch_utils.py#L161-L171)

Old API stack
- `multi_gpu_train_one_step` reads `minibatch_size`, falling back to `sgd_minibatch_size`, and uses `num_batches = max(1, samples_per_device // per_device_batch_size)`, in timesteps — [train_ops.py L49-51, L128, L171](https://github.com/ray-project/ray/blob/891a5a8946fad05062daebc67bc04da3d78e78c0/python/ray/rllib/execution/train_ops.py#L49-L51)
- Slicing is timestep-indexed through `_slice_map` and snapped to sequence boundaries — [sample_batch.py L1158-1190](https://github.com/ray-project/ray/blob/891a5a8946fad05062daebc67bc04da3d78e78c0/python/ray/rllib/policy/sample_batch.py#L1158-L1190)
- `do_minibatch_sgd` raises if `max_seq_len > sgd_minibatch_size` — [sgd.py L121-133](https://github.com/ray-project/ray/blob/891a5a8946fad05062daebc67bc04da3d78e78c0/python/ray/rllib/utils/sgd.py#L121-L133)
- `pad_batch_to_sequences_of_same_size` and `chop_into_sequences` handle chunking; state inputs are reduced to their values at each chunk start, i.e. stale — [rnn_sequencing.py L115-122, L268-277](https://github.com/ray-project/ray/blob/891a5a8946fad05062daebc67bc04da3d78e78c0/python/ray/rllib/policy/rnn_sequencing.py#L115-L122)

### Inferences
- Update count per training iteration (new stack) = ceil(E x n / floor(n x S / D_ts)), which is approximately E x D_ts / S, with D_ts the unpadded timesteps. `max_seq_len` enters only through rounding.
- Worked rounding example with RLlib defaults and no early dones: D=4000, S=128, L=20 gives n=200, step=6, so 34 minibatches per epoch instead of 31.25, each holding 120 timesteps rather than 128. Rounding always shrinks a minibatch and adds updates.
- Short episodes create more, shorter sequences, so mean sequence length drops and each minibatch holds more sequences. The timestep budget holds, not the sequence count.
- Seeing `sum(seq_lens)` used as the batch length, the TODO comment on master looks stale rather than describing a live bug. The comment itself cannot confirm this.
- Citing "RLlib counts timesteps" needs a version qualifier. The 2.9.0 behaviour, sequences per minibatch, is the same semantics as this repo's envs-per-minibatch.
- ray-2.60.0 with `minibatch_size` below the mean sequence length gets `n_steps = 0`; from the code this would likely yield empty minibatches, but this was not run.

### Gaps
- The exact commit that introduced the conversion between 2.9.0 and 2.10.0 was not identified, nor the first version that made it unconditional (somewhere after 2.40.0).
- The `PrevActionsPrevRewards` connector's internals were confirmed by name, placement and example only, not line by line.
- Whether `lstm_use_prev_action=True` on the new stack silently does nothing was inferred from grep, not from a run.
- TF old-stack multi-GPU slicing (`dynamic_tf_policy_v2.py`) was only partly read.

## Sample Factory: recurrence, rollout, recurrence=-1 and batch_size semantics

### Takeaway
`--recurrence` is an explicit BPTT length and `--batch_size` counts transitions, so raising recurrence lowers trajectories per minibatch and leaves the update count unchanged. Each trajectory starts from the stored stale hidden state, is split at dones with PackedSequence rather than padded, and has no burn-in.

### Cited Findings
Version: `2.1.3` at commit 7a83798 (2026-10-18) — [setup.py](https://github.com/alex-petrenko/sample-factory/blob/7a837983d3625671650e75c9e9d1c1a99f7b7454/setup.py#L47)
- Defaults: `--batch_size 1024` (help text calls it the SGD minibatch size), `--num_batches_per_epoch 1`, `--num_epochs 1`, `--rollout 32`, `--recurrence -1`, `--shuffle_minibatches False` — [cfg.py L128-171](https://github.com/alex-petrenko/sample-factory/blob/7a837983d3625671650e75c9e9d1c1a99f7b7454/sample_factory/cfg/cfg.py#L128-L171)
- `--num_batches_per_epoch` help text states the unit: `batch_size=128`, `num_batches_per_epoch=2` means 256 environment transitions per iteration — [cfg.py L129-135](https://github.com/alex-petrenko/sample-factory/blob/7a837983d3625671650e75c9e9d1c1a99f7b7454/sample_factory/cfg/cfg.py#L129-L135)
- `recurrence == -1` resolves to `cfg.rollout` for RNNs and 1 otherwise — [arguments.py L98-100](https://github.com/alex-petrenko/sample-factory/blob/7a837983d3625671650e75c9e9d1c1a99f7b7454/sample_factory/cfg/arguments.py#L98-L100)
- `use_rnn` defaults True, `rnn_size` 512, `rnn_type` gru — [cfg.py L529-543](https://github.com/alex-petrenko/sample-factory/blob/7a837983d3625671650e75c9e9d1c1a99f7b7454/sample_factory/cfg/cfg.py#L529-L543)
- With an RNN, `recurrence <= 1` is a config error, and V-trace requires `recurrence == rollout` — [arguments.py L188-194](https://github.com/alex-petrenko/sample-factory/blob/7a837983d3625671650e75c9e9d1c1a99f7b7454/sample_factory/cfg/arguments.py#L188-L194)
- `_get_minibatches` asserts `rollout % recurrence == 0` and `experience_size % batch_size == 0`. With `num_batches_per_epoch == 1` the whole buffer is the minibatch. With shuffling, start indices every `recurrence` steps are permuted, expanded to whole segments, and `np.split` into `experience_size // batch_size` parts. Without shuffling, fixed `batch_size` slices are used in order — [learner.py L498-528](https://github.com/alex-petrenko/sample-factory/blob/7a837983d3625671650e75c9e9d1c1a99f7b7454/sample_factory/algo/learning/learner.py#L498-L528)
- Trajectories per minibatch are computed as `minibatch_size // recurrence` — [learner.py L583](https://github.com/alex-petrenko/sample-factory/blob/7a837983d3625671650e75c9e9d1c1a99f7b7454/sample_factory/algo/learning/learner.py#L583)
- Total update estimate is `cfg.train_for_env_steps // cfg.batch_size * cfg.num_epochs`, with no `recurrence` term — [learner.py L90](https://github.com/alex-petrenko/sample-factory/blob/7a837983d3625671650e75c9e9d1c1a99f7b7454/sample_factory/algo/learning/learner.py#L90)
- Hidden states: `build_rnn_inputs` selects stored `rnn_states` at sequence starts and zeroes those that begin a new episode — [rnn_utils.py L140-155](https://github.com/alex-petrenko/sample-factory/blob/7a837983d3625671650e75c9e9d1c1a99f7b7454/sample_factory/algo/learning/rnn_utils.py#L140-L155). Sequences break at dones and every T steps through a `PackedSequence` — [rnn_utils.py L32-47](https://github.com/alex-petrenko/sample-factory/blob/7a837983d3625671650e75c9e9d1c1a99f7b7454/sample_factory/algo/learning/rnn_utils.py#L32-L47). Invalid steps are treated like dones — [learner.py L557-566](https://github.com/alex-petrenko/sample-factory/blob/7a837983d3625671650e75c9e9d1c1a99f7b7454/sample_factory/algo/learning/learner.py#L557-L566)
- Docs: each rollout splits into n = rollout / recurrence segments, and `rollout == recurrence` is the usual choice — [cfg-params.md L233-247](https://github.com/alex-petrenko/sample-factory/blob/7a837983d3625671650e75c9e9d1c1a99f7b7454/docs/02-configuration/cfg-params.md#L233-L247). Published DMLab commands use `--rollout=32 --recurrence=32 --batch_size=2048` — [dmlab.md L23](https://github.com/alex-petrenko/sample-factory/blob/7a837983d3625671650e75c9e9d1c1a99f7b7454/docs/09-environment-integrations/dmlab.md#L23)
- Paper: Table A.5 (VizDoom) lists rollout length T = 32, batch size 2048 samples, 1 training epoch, with batch size labelled in samples; Section 3.4 says larger T in the 2^5 to 2^7 range is generally wanted for BPTT with recurrent policies — [Petrenko et al. 2020, arXiv:2006.11751](https://arxiv.org/abs/2006.11751)
- A case-insensitive grep for burn-in across `.py` and `.yaml` found nothing — [repository](https://github.com/alex-petrenko/sample-factory/tree/7a837983d3625671650e75c9e9d1c1a99f7b7454)

### Inferences
- Updates per training iteration = `num_epochs x num_batches_per_epoch`; dataset size = `batch_size x num_batches_per_epoch` transitions. Sequences per minibatch = `batch_size // recurrence`.
- The default configuration (`num_batches_per_epoch=1`, `num_epochs=1`) is one SGD step on the whole buffer, which is a very different regime from a 256 x 256 synchronous rollout.
- Without `--shuffle_minibatches`, minibatch membership is fixed across epochs.
- `batch_size % recurrence == 0` is not asserted directly, but the `(-1, T)` reshape of dones in `_build_pack_info_from_dones` needs it.

### Gaps
- Table A.5 lists rollout but no separate recurrence row in the text extracted; the paper's explicit BPTT length rests on the rollout row and docs commands.
- Whether `rnn_states` are ever refreshed before training was checked by reading `_prepare_batch` and `_calculate_losses` only; no recompute path was seen there.

## rl_games: seq_length, zero_rnn_on_done and minibatch construction

### Takeaway
`seq_length` (default 4) is a true BPTT window and `minibatch_size` counts samples, so `num_minibatches = batch_size // minibatch_size` does not depend on it. States are stored every `seq_length` steps and reused, there is no padding or burn-in, and RNN minibatches are not shuffled.

### Cited Findings
Version: `2.0.0` in pyproject at commit f7e4345 (2026-11-01) — [pyproject.toml](https://github.com/Denys88/rl_games/blob/f7e43450628c9e3eb3c9927b825b6ca3d1e0e286/pyproject.toml)
- `seq_length` defaults to 4. `seq_len` is deprecated and triggers a warning. `bptt_length` defaults to `seq_length` but a source comment says it is currently unused. `zero_rnn_on_done` defaults True — [a2c_common.py L413-420](https://github.com/Denys88/rl_games/blob/f7e43450628c9e3eb3c9927b825b6ca3d1e0e286/rl_games/common/a2c_common.py#L413-L420)
- The RNN layer also ignores `bptt_len`; the loop that honoured it is commented out — [recurrent.py L26-27, L60-74](https://github.com/Denys88/rl_games/blob/f7e43450628c9e3eb3c9927b825b6ca3d1e0e286/rl_games/common/layers/recurrent.py#L26-L27)
- `batch_size = horizon_length * num_actors * num_agents`. `minibatch_size` falls back to `num_actors * minibatch_size_per_env`. `games_num = minibatch_size // seq_length`, and `num_minibatches = batch_size // minibatch_size` — [a2c_common.py L447-470](https://github.com/Denys88/rl_games/blob/f7e43450628c9e3eb3c9927b825b6ca3d1e0e286/rl_games/common/a2c_common.py#L447-L470)
- Divisibility is enforced: `batch_size % minibatch_size`, `batch_size % seq_length`, and for RNNs `minibatch_size % seq_length` — [datasets.py L8-13](https://github.com/Denys88/rl_games/blob/f7e43450628c9e3eb3c9927b825b6ca3d1e0e286/rl_games/common/datasets.py#L8-L13). `horizon*agents // num_minibatches` must also divide by `seq_length` — [a2c_common.py L793-796](https://github.com/Denys88/rl_games/blob/f7e43450628c9e3eb3c9927b825b6ca3d1e0e286/rl_games/common/a2c_common.py#L793-L796)
- Stale state: `play_steps_rnn` copies `self.rnn_states` to `mb_rnn_states` when `n % seq_length == 0` — [a2c_common.py L1235-1238](https://github.com/Denys88/rl_games/blob/f7e43450628c9e3eb3c9927b825b6ca3d1e0e286/rl_games/common/a2c_common.py#L1235-L1238)
- Minibatches: `num_games_batch = minibatch_size // seq_length`, and `_get_item_rnn` takes contiguous ranges of sequences plus matching states — [datasets.py L24, L62-73](https://github.com/Denys88/rl_games/blob/f7e43450628c9e3eb3c9927b825b6ca3d1e0e286/rl_games/common/datasets.py#L62-L73)
- No shuffling for RNNs: `apply_permutation` requires `self.permute and not self.is_rnn` — [datasets.py L46-55](https://github.com/Denys88/rl_games/blob/f7e43450628c9e3eb3c9927b825b6ca3d1e0e286/rl_games/common/datasets.py#L46-L55). `PPODataset` is also built without `permute` — [a2c_continuous.py L79](https://github.com/Denys88/rl_games/blob/f7e43450628c9e3eb3c9927b825b6ca3d1e0e286/rl_games/algos_torch/a2c_continuous.py#L79). Data order is `transpose(0, 1)` then flatten, i.e. env by env — [a2c_common.py L34-41](https://github.com/Denys88/rl_games/blob/f7e43450628c9e3eb3c9927b825b6ca3d1e0e286/rl_games/common/a2c_common.py#L34-L41)
- Dones during training: `RnnWithDones.forward` splits a sequence at any done in the batch and multiplies hidden states by `not_done`, so there is no padding — [recurrent.py L20-58](https://github.com/Denys88/rl_games/blob/f7e43450628c9e3eb3c9927b825b6ca3d1e0e286/rl_games/common/layers/recurrent.py#L20-L58)
- Dones during collection: hidden state is zeroed at done indices when `zero_rnn_on_done` is True — [a2c_common.py L1306-1309](https://github.com/Denys88/rl_games/blob/f7e43450628c9e3eb3c9927b825b6ca3d1e0e286/rl_games/common/a2c_common.py#L1306-L1309). README documents False as keeping state across resets, potentially useful with domain randomization — [README.md L389](https://github.com/Denys88/rl_games/blob/f7e43450628c9e3eb3c9927b825b6ca3d1e0e286/README.md#L389)
- A grep for burn-in across `.py` and `.yaml` found nothing — [repository](https://github.com/Denys88/rl_games/tree/f7e43450628c9e3eb3c9927b825b6ca3d1e0e286)
- Shipped configs use `seq_length` from 4 to 32, e.g. 32 for `ppo_walker_rnn.yaml` — [ppo_walker_rnn.yaml L66](https://github.com/Denys88/rl_games/blob/f7e43450628c9e3eb3c9927b825b6ca3d1e0e286/rl_games/configs/ppo_walker_rnn.yaml#L66)

### Inferences
- Updates per iteration = `mini_epochs x (horizon x actors x agents) // minibatch_size`. Sequences per minibatch = `minibatch_size // seq_length`.
- Because order is env by env and unshuffled, each minibatch holds runs of consecutive windows from the same envs, with identical membership in every mini-epoch. Correlation is thus higher than in this repo's env shuffling, and a minibatch can contain window k and window k+1 of one env.

### Gaps
- Whether `dones` reach the RNN when `zero_rnn_on_done=False` was not traced through every network builder.
- `seq_length` is not listed in the README parameter table lines inspected, so semantics come from code alone.

## rsl_rl and IsaacLab: splitting at dones, padding and mini-batching

### Takeaway
rsl_rl has no BPTT or sequence-length parameter for PPO: the window is the whole rollout, cut only at dones, padded and masked, and `num_mini_batches` partitions envs. The update count is simply epochs x mini-batches. IsaacLab adds only configuration around this and inherits `seq_length` when run with rl_games.

### Cited Findings
Versions: rsl_rl `5.5.2` at 1156331 (2026-10-28) — [pyproject.toml](https://github.com/leggedrobotics/rsl_rl/blob/11563319134f987f70fb034410496fe3cc7327a9/pyproject.toml#L7); `isaaclab_rl 5.3.1` at IsaacLab e3a7974 (2026-11-07) — [pyproject.toml](https://github.com/isaac-sim/IsaacLab/blob/e3a79743e3119b82ebb72e8e3e2410c9275cc168/source/isaaclab_rl/pyproject.toml#L12)
- `PPO` defaults: `num_learning_epochs = 5`, `num_mini_batches = 4` — [ppo.py L45-46](https://github.com/leggedrobotics/rsl_rl/blob/11563319134f987f70fb034410496fe3cc7327a9/rsl_rl/algorithms/ppo.py#L45-L46). `num_steps_per_env` is required — [configuration.rst L55-58](https://github.com/leggedrobotics/rsl_rl/blob/11563319134f987f70fb034410496fe3cc7327a9/docs/guide/configuration.rst#L55-L58)
- PPO selects `recurrent_mini_batch_generator` when actor or critic is recurrent — [ppo.py L204-207](https://github.com/leggedrobotics/rsl_rl/blob/11563319134f987f70fb034410496fe3cc7327a9/rsl_rl/algorithms/ppo.py#L204-L207)
- The generator requires `1 <= num_mini_batches <= num_envs`, pads once via `split_and_pad_trajectories`, then slices envs `start = i * num_envs // num_mini_batches` with no permutation. The number of trajectories in a mini-batch is the count of episode starts in those envs, so it varies — [rollout_storage.py L262-331](https://github.com/leggedrobotics/rsl_rl/blob/11563319134f987f70fb034410496fe3cc7327a9/rsl_rl/storage/rollout_storage.py#L262-L331). The feed-forward generator, by contrast, uses `randperm` — [rollout_storage.py L225-247](https://github.com/leggedrobotics/rsl_rl/blob/11563319134f987f70fb034410496fe3cc7327a9/rsl_rl/storage/rollout_storage.py#L225-L247)
- Splitting: `dones[-1] = 1`, trajectories split at each done and padded with zeros to the longest length, with a boolean mask; a zero trajectory of full length is added and removed so padding reaches T — [utils.py L395-437](https://github.com/leggedrobotics/rsl_rl/blob/11563319134f987f70fb034410496fe3cc7327a9/rsl_rl/utils/utils.py#L395-L437)
- Hidden states: all are saved per step, but training selects only entries where `last_was_done` is true, with `last_was_done[0] = True`, so each trajectory starts from its stale stored state — [rollout_storage.py L279-298](https://github.com/leggedrobotics/rsl_rl/blob/11563319134f987f70fb034410496fe3cc7327a9/rsl_rl/storage/rollout_storage.py#L279-L298). Done envs have their state zeroed at collection — [rnn.py L52-66](https://github.com/leggedrobotics/rsl_rl/blob/11563319134f987f70fb034410496fe3cc7327a9/rsl_rl/modules/rnn.py#L52-L66)
- In batch mode the RNN runs on padded input with the given hidden state, then `unpad_trajectories` restores the [T, env] layout — [rnn.py L35-51](https://github.com/leggedrobotics/rsl_rl/blob/11563319134f987f70fb034410496fe3cc7327a9/rsl_rl/modules/rnn.py#L35-L51); [utils.py L452-463](https://github.com/leggedrobotics/rsl_rl/blob/11563319134f987f70fb034410496fe3cc7327a9/rsl_rl/utils/utils.py#L452-L463)
- Truncated gradients exist only in `Distillation`: `gradient_length = 15`. This is student-teacher training, not PPO — [distillation.py L44, L94-100](https://github.com/leggedrobotics/rsl_rl/blob/11563319134f987f70fb034410496fe3cc7327a9/rsl_rl/algorithms/distillation.py#L44)
- A grep for burn-in found nothing — [repository](https://github.com/leggedrobotics/rsl_rl/tree/11563319134f987f70fb034410496fe3cc7327a9)
- IsaacLab `RslRlRNNModelCfg` exposes `rnn_type` and `rnn_hidden_dim` and no sequence length; `RslRlPpoAlgorithmCfg` exposes `num_mini_batches` and `num_learning_epochs`; the runner config exposes `num_steps_per_env` — [rl_cfg.py L72-85, L143-146, L217](https://github.com/isaac-sim/IsaacLab/blob/e3a79743e3119b82ebb72e8e3e2410c9275cc168/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py#L72-L85)
- Shipped recurrent example: `AnymalDFlatPPORunnerRecurrentCfg` uses LSTM, 256 units, 1 layer; the file uses `num_steps_per_env = 24`, 5 epochs, 4 mini-batches — [rsl_rl_ppo_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/e3a79743e3119b82ebb72e8e3e2410c9275cc168/source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/anymal_d/agents/rsl_rl_ppo_cfg.py#L21)
- IsaacLab rl_games YAMLs set `seq_length: 4`, including the Shadow Hand LSTM config: horizon 16, minibatch 16384, mini_epochs 4 — [rl_games_ppo_lstm_cfg.yaml L84-89](https://github.com/isaac-sim/IsaacLab/blob/e3a79743e3119b82ebb72e8e3e2410c9275cc168/source/isaaclab_tasks/isaaclab_tasks/contrib/reorient/config/shadow_hand/agents/rl_games_ppo_lstm_cfg.yaml#L84-L89)

### Inferences
- Updates per iteration = `num_learning_epochs x num_mini_batches`. Samples per mini-batch = `(num_envs / num_mini_batches) x num_steps_per_env`.
- The effective BPTT window is at most `num_steps_per_env`, so 24 steps in the example. To change the window you must change the rollout length, which also moves GAE horizon and batch size.
- This is the same structure as this repo at window = T, except rsl_rl fixes the mini-batch count rather than envs per mini-batch, and it never subdivides a rollout. Mini-batch composition is fixed across epochs.

### Gaps
- Exact rsl-rl version pinned by IsaacLab was not read.
- IsaacLab's skrl and TorchRL wrappers were not inspected for RNN handling; they are assumed to inherit those libraries' behaviour.

## Tianshou, TorchRL, skrl, PureJaxRL, MAPPO, DD-PPO and Habitat, Acme, EnvPool

### Takeaway
Explicit recurrent PPO windows exist in TorchRL (`slice_len`, `sub_traj_len`), skrl (`sequence_length`) and MAPPO (`data_chunk_length`). PureJaxRL and Habitat use the whole rollout, Tianshou's `stack_num` is a different mechanism with zero initial state, Acme PPO is not recurrent, and EnvPool has no learner. `num_recurrent_steps` does not exist in Habitat at this commit.

### Cited Findings
Tianshou 2.0.1 (f240205, 2026-04-03)
- `stack_num` is a replay-buffer frame-stack sampling argument, default 1 — [buffer_base.py L32-33, L74](https://github.com/thu-ml/tianshou/blob/f2402056b03acafeedf2518a5088db55add004ab/tianshou/data/buffer/buffer_base.py#L32-L33). High-level name: `replay_buffer_stack_num = 1` — [config.py L124](https://github.com/thu-ml/tianshou/blob/f2402056b03acafeedf2518a5088db55add004ab/tianshou/highlevel/config.py#L124)
- PPO's update calls `self.policy(minibatch)` and `self.critic(minibatch.obs)` with no state, and minibatches come from `batch.split(batch_size, merge_last=True)` — [ppo.py L167-200](https://github.com/thu-ml/tianshou/blob/f2402056b03acafeedf2518a5088db55add004ab/tianshou/algorithm/modelfree/ppo.py#L167-L200). `split` shuffles individual indices by default — [batch.py L1202-1210](https://github.com/thu-ml/tianshou/blob/f2402056b03acafeedf2518a5088db55add004ab/tianshou/data/batch.py#L1202-L1210)
- `Recurrent` and `RecurrentActorProb` run the LSTM from a zero state when `state is None` — [common.py L372-440](https://github.com/thu-ml/tianshou/blob/f2402056b03acafeedf2518a5088db55add004ab/tianshou/utils/net/common.py#L372-L440). `RecurrentActorProb` uses only the last output, `obs[:, -1]` — [continuous.py L262-320](https://github.com/thu-ml/tianshou/blob/f2402056b03acafeedf2518a5088db55add004ab/tianshou/utils/net/continuous.py#L262-L320)
- The Collector saves hidden state into the buffer — [collector.py L752-756](https://github.com/thu-ml/tianshou/blob/f2402056b03acafeedf2518a5088db55add004ab/tianshou/data/collector.py#L752-L756), but a grep of `tianshou/algorithm` finds no `hidden_state` reader. `RecurrentActorProb` appears only in library code and `test/base/test_utils.py`; the RNN `stack_num` test is off-policy DRQN — [test_drqn.py L111](https://github.com/thu-ml/tianshou/blob/f2402056b03acafeedf2518a5088db55add004ab/test/discrete/test_drqn.py#L111)

TorchRL 0.15.0 (003c5f4, 2026-11-06)
- `SliceSampler` takes exclusive `num_slices` or `slice_len`, and `batch_size` stays in transitions: `num_slices = batch_size // slice_len`, with divisibility required — [slice.py L46-51, L902-917](https://github.com/pytorch/rl/blob/003c5f467f0f720e85aa8eb5884ce55e414c7f1d/torchrl/data/replay_buffers/samplers/slice.py#L902-L917)
- `PPOTrainer` arguments `minibatch_size = 256` (transitions per optimization step) and `sub_traj_len = None` (consecutive steps per recurrent window); `sub_traj_len` must divide `minibatch_size` — [ppo.py L67-69, L95-100, L168-174](https://github.com/pytorch/rl/blob/003c5f467f0f720e85aa8eb5884ce55e414c7f1d/torchrl/trainers/algorithms/ppo.py#L95-L100). `optim_steps_per_batch = ceil(collected_batch_size / minibatch_size)` in the recurrent path, with no `sub_traj_len` term — [ppo.py L248-252](https://github.com/pytorch/rl/blob/003c5f467f0f720e85aa8eb5884ce55e414c7f1d/torchrl/trainers/algorithms/ppo.py#L248-L252). The trainer runs `optim_steps_per_batch * num_epochs` steps, and PPO's default `num_epochs` is 4 — [trainers.py L1896](https://github.com/pytorch/rl/blob/003c5f467f0f720e85aa8eb5884ce55e414c7f1d/torchrl/trainers/trainers.py#L1896); [ppo.py L57](https://github.com/pytorch/rl/blob/003c5f467f0f720e85aa8eb5884ce55e414c7f1d/torchrl/trainers/algorithms/ppo.py#L57)
- `BatchSubSampler` returns `[batch_size // sub_traj_len, sub_traj_len]` using `torch.randint` over trajectories, so draws are with replacement — [trainers.py L2790-2880](https://github.com/pytorch/rl/blob/003c5f467f0f720e85aa8eb5884ce55e414c7f1d/torchrl/trainers/trainers.py#L2790-L2880)
- `LSTMModule` in recurrent mode starts from the stored state at slice index 0 (`hidden0_in[..., 0, :, :]`), and zeroing at `is_init` applies only in non-recurrent mode because recurrent mode may begin at any intermediate state — [rnn.py L1392-1400, L1508-1509](https://github.com/pytorch/rl/blob/003c5f467f0f720e85aa8eb5884ce55e414c7f1d/torchrl/modules/tensordict_module/rnn.py#L1392-L1400)
- Mid-slice dones use split-and-pad (pad backend) or direct resets from `is_init` (scan, triton) — [recurrent_state_lifecycle.rst L270-281](https://github.com/pytorch/rl/blob/003c5f467f0f720e85aa8eb5884ce55e414c7f1d/docs/source/reference/recurrent_state_lifecycle.rst#L270-L281)
- Example `isaaclab_rnn_ppo_memory.py` samples whole rollouts: `--mini-batch-steps 8192`, `sample_num_slices = mini_batch_steps // rollout_steps`, `updates_per_epoch = ceil(num_envs / sample_num_slices)` — [example L297-298, L475](https://github.com/pytorch/rl/blob/003c5f467f0f720e85aa8eb5884ce55e414c7f1d/examples/collectors/isaaclab_rnn_ppo_memory.py#L297-L298)
- `BurnInTransform` exists with config default `burn_in = 10`, but neither `trainers/algorithms/ppo.py` nor `objectives/ppo.py` mentions burn-in — [configs/transforms.py L688-694](https://github.com/pytorch/rl/blob/003c5f467f0f720e85aa8eb5884ce55e414c7f1d/torchrl/trainers/algorithms/configs/transforms.py#L688-L694); [_env.py L1924](https://github.com/pytorch/rl/blob/003c5f467f0f720e85aa8eb5884ce55e414c7f1d/torchrl/envs/transforms/_env.py#L1924)

skrl 2.1.1 (db8b1ef, 2026-11-03)
- `PPO_RNN` lives in `ppo_rnn.py` — [ppo.rst L108-110](https://github.com/Toni-SM/skrl/blob/db8b1ef534229866ee6ddde7b3dd1468d3ea6c13/docs/source/api/agents/ppo.rst#L108-L110). Sequence length comes from the model's specification `["rnn"]["sequence_length"]`, default 1 in the agent — [ppo_rnn.py L203](https://github.com/Toni-SM/skrl/blob/db8b1ef534229866ee6ddde7b3dd1468d3ea6c13/skrl/agents/torch/ppo/ppo_rnn.py#L203). Doc snippets use `sequence_length=10` — [gaussian_model.py L730](https://github.com/Toni-SM/skrl/blob/db8b1ef534229866ee6ddde7b3dd1468d3ea6c13/docs/source/snippets/gaussian_model.py#L730)
- `PPO_CFG`: `rollouts = 16`, `learning_epochs = 8`, `mini_batches = 2` — [ppo_cfg.py L14-20](https://github.com/Toni-SM/skrl/blob/db8b1ef534229866ee6ddde7b3dd1468d3ea6c13/skrl/agents/torch/ppo/ppo_cfg.py#L14-L20)
- `sample_all` with `sequence_length > 1` does `np.array_split(all_sequence_indexes, mini_batches)`, and `all_sequence_indexes` is env-ordered with no shuffle — [base.py L64-65, L321-326](https://github.com/Toni-SM/skrl/blob/db8b1ef534229866ee6ddde7b3dd1468d3ea6c13/skrl/memories/torch/base.py#L321-L326)
- Model code reshapes to `(-1, sequence_length, ...)`, takes `hidden_states[:, :, 0, :]` as initial state, and splits at `terminated` with zero resets — [gaussian_model.py L789-815](https://github.com/Toni-SM/skrl/blob/db8b1ef534229866ee6ddde7b3dd1468d3ea6c13/docs/source/snippets/gaussian_model.py#L789-L815)
- States are stored per step in memory as `rnn_policy_i` — [ppo_rnn.py L334-354](https://github.com/Toni-SM/skrl/blob/db8b1ef534229866ee6ddde7b3dd1468d3ea6c13/skrl/agents/torch/ppo/ppo_rnn.py#L334-L354)

PureJaxRL (31756b1, 2024-09-09)
- `purejaxrl/ppo_rnn.py` has no window parameter. Defaults `NUM_ENVS 4`, `NUM_STEPS 128`, `UPDATE_EPOCHS 4`, `NUM_MINIBATCHES 4` — [ppo_rnn.py L345-349](https://github.com/luchris429/purejaxrl/blob/31756b197773a52db763fdbe6d635e4b46522a73/purejaxrl/ppo_rnn.py#L345-L349)
- `initial_hstate` from the rollout start is reused for all epochs — [ppo_rnn.py L175, L291](https://github.com/luchris429/purejaxrl/blob/31756b197773a52db763fdbe6d635e4b46522a73/purejaxrl/ppo_rnn.py#L175). Minibatches permute envs, then reshape to `[T, NUM_MINIBATCHES, -1]` — [ppo_rnn.py L258-269](https://github.com/luchris429/purejaxrl/blob/31756b197773a52db763fdbe6d635e4b46522a73/purejaxrl/ppo_rnn.py#L258-L269). `ScannedRNN` resets inside the scan from `resets`, so there is no padding — [ppo_rnn.py L17-35](https://github.com/luchris429/purejaxrl/blob/31756b197773a52db763fdbe6d635e4b46522a73/purejaxrl/ppo_rnn.py#L17-L35)

MAPPO on-policy (de66d7a, 2024-07-18)
- `--data_chunk_length` default 10, `--num_mini_batch` default 1, `--ppo_epoch` default 15, `--episode_length` default 200 — [config.py L187-247](https://github.com/marlbenchmark/on-policy/blob/de66d7a4b23fac2513f56f96f73b3f5cb96695ac/onpolicy/config.py#L187-L247)
- `recurrent_generator`: `data_chunks = batch_size // data_chunk_length`, `mini_batch_size = data_chunks // num_mini_batch`, `randperm` over chunks, state from the chunk's first index `rnn_states[ind]` — [shared_buffer.py L499-512, L568-569](https://github.com/marlbenchmark/on-policy/blob/de66d7a4b23fac2513f56f96f73b3f5cb96695ac/onpolicy/utils/shared_buffer.py#L499-L512)
- `num_updates = ppo_epoch * num_mini_batch` — [r_mappo.py L219](https://github.com/marlbenchmark/on-policy/blob/de66d7a4b23fac2513f56f96f73b3f5cb96695ac/onpolicy/algorithms/r_mappo/r_mappo.py#L219)
- `RNNLayer` zeroes hidden state via `masks` mid-sequence, with no padding — [rnn.py L24-66](https://github.com/marlbenchmark/on-policy/blob/de66d7a4b23fac2513f56f96f73b3f5cb96695ac/onpolicy/algorithms/utils/rnn.py#L24-L66)
- Paper Algorithm 1 splits trajectories into chunks of length L and initializes each chunk from its first stored hidden state; the listed chunk length is 10 — [Yu et al., arXiv:2103.01955](https://arxiv.org/abs/2103.01955)

DD-PPO and Habitat (0fb6f43, 2026-05-07)
- `num_recurrent_steps` has no match in `.py`, `.yaml` or `.md` under the checked-out `habitat-baselines`; the nearest names are `ppo.num_steps` and `num_recurrent_layers` — [default_structured_configs.py L292-299, L349](https://github.com/facebookresearch/habitat-lab/blob/0fb6f43ffe806a8088a171b036336c093bcf604e/habitat-baselines/habitat_baselines/config/default_structured_configs.py#L292-L299)
- Defaults: `ppo_epoch = 4`, `num_mini_batch = 2`, `num_steps = 5`. `ddppo_pointnav.yaml` sets 2, 2 and 128 — [ddppo_pointnav.yaml L33-41](https://github.com/facebookresearch/habitat-lab/blob/0fb6f43ffe806a8088a171b036336c093bcf604e/habitat-baselines/habitat_baselines/config/pointnav/ddppo_pointnav.yaml#L33-L41)
- `data_generator` permutes envs, chunks them into `num_mini_batch` groups, keeps hidden state only at `[0:1]`, and builds packed sequences from dones — [rollout_storage.py L207-258](https://github.com/facebookresearch/habitat-lab/blob/0fb6f43ffe806a8088a171b036336c093bcf604e/habitat-baselines/habitat_baselines/common/rollout_storage.py#L207-L258). Sequence starts after a done are zeroed — [rnn_state_encoder.py L187-241](https://github.com/facebookresearch/habitat-lab/blob/0fb6f43ffe806a8088a171b036336c093bcf604e/habitat-baselines/habitat_baselines/rl/models/rnn_state_encoder.py#L187-L241)
- Update loop is `ppo_epoch` times the generator — [ppo.py L300-316](https://github.com/facebookresearch/habitat-lab/blob/0fb6f43ffe806a8088a171b036336c093bcf604e/habitat-baselines/habitat_baselines/rl/ppo/ppo.py#L300-L316)
- Paper: each worker collects up to 128 frames from 4 agents, then runs 2 epochs with 2 mini-batches, using a 2-layer 512-d LSTM — [Wijmans et al., arXiv:1911.00357](https://arxiv.org/abs/1911.00357)

Acme 0.4.1 (2e09e3d, 2026-10-23)
- `PPOConfig`: `unroll_length = 8`, `num_minibatches = 8`, `num_epochs = 2`, `batch_size = 256` segments; the docstring gives minibatch size as `batch_size * unroll_length / num_minibatches` — [config.py L29-33, L64-67](https://github.com/google-deepmind/acme/blob/2e09e3ddecf4a1ba783688c92aa493c596a2beec/acme/agents/jax/ppo/config.py#L29-L33)
- The learner flattens `[num_sequences, num_steps]` into individual timesteps before minibatching, and grep finds no recurrent state in `acme/agents/jax/ppo` — [learning.py L338-357](https://github.com/google-deepmind/acme/blob/2e09e3ddecf4a1ba783688c92aa493c596a2beec/acme/agents/jax/ppo/learning.py#L338-L357)
- Recurrent training is in other agents: R2D2 has `burn_in_length = 40`, `trace_length = 80`, `batch_size = 64` — [r2d2/config.py L32-45](https://github.com/google-deepmind/acme/blob/2e09e3ddecf4a1ba783688c92aa493c596a2beec/acme/agents/jax/r2d2/config.py#L32-L45); IMPALA has `sequence_length = 20`, `batch_size = 32` — [impala/config.py L30-35](https://github.com/google-deepmind/acme/blob/2e09e3ddecf4a1ba783688c92aa493c596a2beec/acme/agents/jax/impala/config.py#L30-L35)

EnvPool 1.2.8 (68e5ada, 2026-10-21)
- It is an environment library with examples for other frameworks. A grep of its `.py`, `.md` and `.rst` files finds no `bptt`, `seq_len`, `burn_in`, LSTM, RNN or recurrent references; the only `sequence_length` is in an Acme adapter — [examples/acme_examples/helpers.py L68](https://github.com/sail-sg/envpool/blob/68e5ada42789d8fa1b8a3f0b366731c9caa187ef/examples/acme_examples/helpers.py#L68)

### Inferences
- Tianshou PPO updates = repeat x floor(N / B); `stack_num` does not change it. Each sample would replay `stack_num` steps from zeros and take the loss at the last step only, so windows overlap, stored states go unused, and cost scales with `stack_num`. This is closer to a frame stack fed to an RNN than to TBPTT. No recurrent PPO test or example supports this path, so treat it as unsupported rather than designed.
- TorchRL with `sub_traj_len` matches RLlib semantics: fixed transitions per step, so sequences = minibatch / window. But `BatchSubSampler` is random sampling rather than a partition, so one epoch need not cover all data.
- skrl: updates = `learning_epochs x mini_batches`. Sequences per mini-batch = samples / L, set implicitly by the model's `view`. Windows are consecutive over env-ordered data, so if `rollouts % sequence_length != 0` a window would cross from one env into the next. That combination is the documented snippet value 10 against the default `rollouts = 16`. This comes from reading code, not from running it.
- MAPPO: `num_mini_batch` holds updates constant and `mini_batch_size` (in chunks) shrinks as L grows, so samples per minibatch stay T x N x M / K.
- PureJaxRL and Habitat are the envs-per-minibatch design at window = T, where the repo's problem does not arise because the window is not a free parameter. Updates = E x K in both.
- Acme `unroll_length` is not a BPTT knob for PPO, and Acme's sequence-counted `batch_size` belongs to off-policy or actor-learner agents, so it is no counterexample.

### Gaps
- Whether TorchRL's `BurnInTransform` is compatible with `ClipPPOLoss` was not tested; `PPOTrainer` does not visibly configure it.
- Habitat `num_recurrent_steps` may exist in older releases; only this commit was checked. Please confirm where the name came from if it matters.
- skrl's non-PyTorch backends (JAX, Warp) were not read beyond seeing `sequence_length` in their memory modules.
- Tianshou's `buffer.sample(sample_size)` value for on-policy algorithms was confirmed only by locating the call at [algorithm_base.py L622](https://github.com/thu-ml/tianshou/blob/f2402056b03acafeedf2518a5088db55add004ab/tianshou/algorithm/algorithm_base.py#L622).

## Cross-library: is the minibatch counted in samples or sequences, and what is the update count?

### Takeaway
Among current PPO implementations checked, none lets a longer window reduce the update count: each fixes either timesteps per minibatch (RLlib, Sample Factory, rl_games, TorchRL) or the number of minibatches (skrl, MAPPO, rsl_rl, PureJaxRL, Habitat). This repo's formula, which fixes sequences per minibatch and multiplies by T / L, matched only RLlib 2.9.0-era slicing and is the outlier. All libraries here that use stored states use them stale, and no PPO path in them has a dedicated burn-in setting.

### Cited Findings
Each row is backed by the sections above; links go to the main file.

| Library (version) | Window parameter (default) | Initial state | Burn-in for PPO | Done handling | Minibatch defined by | Shuffling | Updates per iteration |
|---|---|---|---|---|---|---|---|
| [RLlib new stack (2.60, master)](https://github.com/ray-project/ray/blob/ray-2.60.0/rllib/utils/minibatch_utils.py) | `max_seq_len` (20) | Stale stored | None | New row plus right zero pad, `LOSS_MASK` | `minibatch_size` timesteps, floored to sequences | Sequences | about E x D / S, plus rounding |
| [RLlib old stack (labelled legacy)](https://github.com/ray-project/ray/blob/891a5a8946fad05062daebc67bc04da3d78e78c0/python/ray/rllib/execution/train_ops.py) | `model["max_seq_len"]` (20) | Stale stored | None | Zero pad plus mask | `sgd_minibatch_size` (deprecated), now `minibatch_size`, timesteps | Offset permutation | E x D // S |
| [Sample Factory 2.1.3](https://github.com/alex-petrenko/sample-factory/blob/7a837983d3625671650e75c9e9d1c1a99f7b7454/sample_factory/algo/learning/learner.py) | `recurrence` (-1 = rollout, 32) | Stale stored | None | PackedSequence split | `batch_size` transitions (1024) | Optional, off by default | E x `num_batches_per_epoch` |
| [rl_games 2.0.0](https://github.com/Denys88/rl_games/blob/f7e43450628c9e3eb3c9927b825b6ca3d1e0e286/rl_games/common/datasets.py) | `seq_length` (4) | Stale stored | None | Split and zero state in layer | `minibatch_size` samples, must divide by L | None for RNN | E x D // S |
| [rsl_rl 5.5.2](https://github.com/leggedrobotics/rsl_rl/blob/11563319134f987f70fb034410496fe3cc7327a9/rsl_rl/storage/rollout_storage.py) | None (window = rollout up to `num_steps_per_env`) | Stale at t=0, zero after done | None | Split, pad, mask | `num_mini_batches` env groups (4) | None, fixed env groups | E x K |
| [TorchRL 0.15.0](https://github.com/pytorch/rl/blob/003c5f467f0f720e85aa8eb5884ce55e414c7f1d/torchrl/trainers/algorithms/ppo.py) | `sub_traj_len` (None), `slice_len` | Stale at slice start | `BurnInTransform` exists, not PPO-wired | Pad or `is_init` reset | `minibatch_size` transitions (256) | Random slices, with replacement | ceil(D / S) x `num_epochs` |
| [skrl 2.1.1](https://github.com/Toni-SM/skrl/blob/db8b1ef534229866ee6ddde7b3dd1468d3ea6c13/skrl/agents/torch/ppo/ppo_rnn.py) | `sequence_length` (agent 1, snippet 10) | Stale stored | None | Split and zero in model | `mini_batches` count (2) | None | E x K |
| [MAPPO](https://github.com/marlbenchmark/on-policy/blob/de66d7a4b23fac2513f56f96f73b3f5cb96695ac/onpolicy/utils/shared_buffer.py) | `data_chunk_length` (10) | Stale at chunk start | None | Mask zeroing | `num_mini_batch` count (1) | `randperm` over chunks | E x K |
| [PureJaxRL](https://github.com/luchris429/purejaxrl/blob/31756b197773a52db763fdbe6d635e4b46522a73/purejaxrl/ppo_rnn.py) | None (full rollout) | Stale at rollout start | None | Reset in scan | `NUM_MINIBATCHES` env groups (4) | Envs per epoch | E x K |
| [Habitat, DD-PPO](https://github.com/facebookresearch/habitat-lab/blob/0fb6f43ffe806a8088a171b036336c093bcf604e/habitat-baselines/habitat_baselines/common/rollout_storage.py) | `ppo.num_steps` (5; 128 in DD-PPO), full rollout | Stale at t=0, zero after done | None | PackedSequence split | `num_mini_batch` env groups (2) | Envs per epoch | E x K |
| [Tianshou 2.0.1](https://github.com/thu-ml/tianshou/blob/f2402056b03acafeedf2518a5088db55add004ab/tianshou/algorithm/modelfree/ppo.py) | `stack_num` (1), not TBPTT | Zero state | N/A | Stacking only | `batch_size` transitions | Individual steps | repeat x N // B |
| [Acme 0.4.1 PPO](https://github.com/google-deepmind/acme/blob/2e09e3ddecf4a1ba783688c92aa493c596a2beec/acme/agents/jax/ppo/learning.py) | None; feed-forward | N/A | N/A | N/A | `num_minibatches` count (8) | Individual steps | E x K |
| [EnvPool 1.2.8](https://github.com/sail-sg/envpool) | N/A, no learner | N/A | N/A | N/A | N/A | N/A | N/A |

Worked numbers for N = T = 256 (D = 65,536), E = 1. Column 2 uses this repo's formula from the brief with 16 envs per minibatch, chosen for illustration and not read from a repo config. Column 3 uses S = 4,096 timesteps. RLlib's rounding logic, computed by hand, gives 16 at L = 8, 32, 128 and 256, with 512, 128, 32 and 16 sequences per minibatch.

| L | Repo-style updates per epoch | Timestep-denominated updates | Sequences per update (timestep-denominated) |
|---|---|---|---|
| 8 | 512 | 16 | 512 |
| 16 | 256 | 16 | 256 |
| 32 | 128 | 16 | 128 |
| 64 | 64 | 16 | 64 |
| 128 | 32 | 16 | 32 |
| 256 | 16 | 16 | 16 |

### Inferences
- Comparing window lengths in this repo today confounds BPTT length with gradient steps: moving from L=256 to L=8 multiplies updates by 32 under fixed envs per minibatch, and each update sees 32 times less data. Published library defaults do not impose that coupling.
- Matching the RLlib, rl_games, Sample Factory and TorchRL convention means fixing S in timesteps and deriving sequences per minibatch as S / L, which needs L to divide S and S to divide D. Equivalently, the skrl and MAPPO route fixes K.
- Neither convention removes every confound. Under fixed S, sequences per update fall as L rises, which raises correlation within a minibatch. A paper could report this as a design choice rather than as neutral. This repo's "sequences fixed" design is defensible as its own ablation only if update count is reported or controlled.
- Stale stored states with no burn-in, as this repo does, is the mainstream PPO design here, so that part needs no justification as a departure.
- Shuffling differs more than expected: rl_games, rsl_rl and skrl do not shuffle recurrent minibatches, whereas RLlib, MAPPO, PureJaxRL and Habitat do. This repo's env shuffling is on the mainstream side.

### Gaps
- No library's documentation explains the rationale for its semantics; the reasoning above is inferred from code.
- No empirical study was found or searched for on update-count effects across BPTT lengths in these libraries; that was outside this source-reading pass.
- SB3 and CleanRL are excluded by scope, so the claim that this repo is an outlier covers only the libraries listed here.
