# Actual coding-agent pilot videos

These are the native execution videos from the first Opus 5.5 / Claude Code pilot, Standard mode, one independent agent attempt per task. They are not oracle replays. All ten agent tasks are now complete: two native successes (drone landing and aerial-manipulator peg insertion) and eight unsuccessful tasks. Drone landing passed all six held-out cases with native score 98.5359894. The aerial trial used a recorded runtime CPU-affinity overlay matching the original simulator CPU quota to avoid JAX thread-pool oversubscription. Its analysis process hit the enforced 4 GiB container memory cap once and was killed; the agent recovered and completed the task. Native resource flags are preserved in results.json.

`results.json` records native verdicts, EFE revision, task-tree hashes, video SHA256 and completion timestamps. Drone files additionally include six held-out cases and the first case from each of the three adaptation rounds. MP4 files are unchanged native exports; their frame rates and native slow motion are retained.

The interactive gallery is hosted at https://embodied-agent-task-atlas.t1357489297.chatgpt.site/efe-videos.html . Benchmark and simulator/model asset attributions remain those in [THIRD_PARTY_NOTICES.md](../../../THIRD_PARTY_NOTICES.md) and the pinned EFE task suites.
