---
type: llm
focus: trace
weight: 1
---

Only the envs that just entered the junction get their recurrent state zeroed (a per-env mask), and each env is reset once per episode, not every step it spends in the junction.
