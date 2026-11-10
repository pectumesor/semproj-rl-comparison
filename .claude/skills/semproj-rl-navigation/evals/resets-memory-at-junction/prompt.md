---
tags: [semproj]
max_turns: 40
timeout_seconds: 1200
allowed_tools: [Read, Glob, Grep, Skill, Write, Edit, Bash]
---

for the E2 memory check i need the 'reset at the junction' diagnostic: during evaluation only, zero the recurrent hidden state the moment the agent enters the T-junction of the twin rooms, so success should drop to ~0.5 if the agent really uses memory. add it so i can run it on a trained checkpoint, for lstm and gru. keep my style
