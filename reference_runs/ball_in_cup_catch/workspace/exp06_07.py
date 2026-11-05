"""Exp 6: normal v3 episode. Exp 7: multi-trial eject/shake/handover recovery test on the real plant."""
import handover
import run_real

print("EXP 6:", run_real.run(exp=6, module="controller"))
handover.run(exp=7, seed=7, module="controller")
