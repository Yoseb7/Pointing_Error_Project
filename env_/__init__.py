from gymnasium.envs.registration import register

register(
    id="HAPS_FSO_Env-v0",
    entry_point="env_.HAPS:HAPSEnv",
)
