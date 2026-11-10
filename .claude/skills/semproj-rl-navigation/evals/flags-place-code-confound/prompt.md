---
tags: [semproj]
max_turns: 30
timeout_seconds: 900
allowed_tools: [Read, Glob, Grep, Skill, Write, Edit, Bash]
---

I trained MLP and LSTM on rooms/two_room.json and rooms/C_maze.json with the default env config, 5 seeds each. MLP gets ~98% success and higher AUC everywhere, LSTM is slower and worse. Can I conclude in the paper that memory doesn't help for indoor navigation? Write me a short answer I can paste to my supervisor.
