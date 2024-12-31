""" Define all environments and provide helper functions to load environments. """

# OpenAI gym interface
from gym.envs.registration import register

from furniture.env.base import make_env, make_vec_env

# register all environment to use
from .furniture_sawyer import FurnitureSawyerEnv
from .furniture_sawyer_dense import FurnitureSawyerDenseRewardEnv

# add sawyer environment to Gym
register(
    id="IKEASawyer-v0",
    entry_point="furniture.env.furniture_gym:FurnitureGym",
    kwargs={
        "id": "IKEASawyer-v0",
        "name": "FurnitureSawyerEnv",
        "furniture_name": "swivel_chair_0700",
        "background": "Industrial",
        "port": 1050,
    },
)

# add sawyer dense reward environment to Gym
register(
    id="IKEASawyerDense-v0",
    entry_point="furniture.env.furniture_gym:FurnitureGym",
    kwargs={"id": "IKEASawyerDense-v0", "name": "FurnitureSawyerDenseRewardEnv", "unity": False},
)


register(
    id="furniture-sawyer-densereward-v0",
    entry_point="furniture.env.furniture_gym:FurnitureGym",
    kwargs={"id": "IKEASawyerDense-v0", "name": "FurnitureSawyerDenseRewardEnv", "unity": False},
)
