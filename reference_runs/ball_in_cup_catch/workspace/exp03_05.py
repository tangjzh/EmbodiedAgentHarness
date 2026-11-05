"""Replicate frozen v2 on three new seeds. Each exp is one real dev episode."""
import run_real

results = {}
for exp in (3, 4, 5):
    results[exp] = run_real.run(exp=exp, module="controller")
print("SUMMARY (return, first_reward_tick):", results)
