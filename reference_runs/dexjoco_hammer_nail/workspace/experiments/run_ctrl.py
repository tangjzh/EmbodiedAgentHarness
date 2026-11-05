"""Development wrapper: run the controller in the live exploration episode.

usage: run_ctrl.py [stop_after_state]
"""
import sys

sys.path.insert(0, '.')
sys.path.insert(0, 'code/hammer')
from api.runtime import Context
from controller import main

stop = sys.argv[1] if len(sys.argv) > 1 else None
with Context() as session:
    main(session, stop_after=stop)
