"""Fresh independent Factory task registration. No course, prefix, or migration."""
def install():
    import soft_envs
    from soft_factory_env import FactoryVisualEnv
    soft_envs.TASKS.update({
        'factory_peg_insert':dict(benchmark='NVIDIA Factory / Isaac Lab',
            environment='Isaac-Factory-PegInsert-Direct-v0',factory_configuration='FactoryTaskPegInsertCfg',
            goal='Insert the already held 8 mm peg into its matching socket.',horizon=150,native_control_hz=15,
            metric_kind='native final-step success',executor_note='Native pregrasp, upright hand, closed gripper and workspace clamp retained.'),
        'factory_gear_mesh':dict(benchmark='NVIDIA Factory / Isaac Lab',
            environment='Isaac-Factory-GearMesh-Direct-v0',factory_configuration='FactoryTaskGearMeshCfg',
            goal='Mesh the already held medium gear with the native gear assembly.',horizon=300,native_control_hz=15,
            metric_kind='native final-step success',executor_note='Native pregrasp, upright hand, closed gripper and workspace clamp retained.')
    })
    original=soft_envs.make_env
    def create(task):
        return FactoryVisualEnv(task) if task.startswith('factory_') else original(task)
    soft_envs.make_env=create
    import soft_pilot
    soft_pilot.make_env=create
