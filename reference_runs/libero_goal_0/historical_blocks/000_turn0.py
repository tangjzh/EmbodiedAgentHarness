<invoke name="python">
<parameter name="code">import numpy as np
obs = get_observation()
print({k: (np.array(v).shape if hasattr(v,'__len__') else v) for k,v in obs.items()} if isinstance(obs,dict) else type(obs))
print("tcp", obs.get("tcp_cartesian_pos") if isinstance(obs,dict) else None)
for name in ["cabinet", "drawer", "middle drawer handle", "middle handle", "drawer handle", "top drawer handle", "bottom drawer handle"]:
    try:
        r = get_object_pose(name)
        print(name, r)
    except Exception as e:
        print(name, "ERR", e)
</parameter>
</invoke>