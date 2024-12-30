## (0) IKEA Furniture Assembly Environment 

[[Environment website (https://clvrai.com/furniture)](https://clvrai.com/furniture)]<br/>


|![](docs/img/agents/video_sawyer_swivel_chair.gif)|![](docs/img/agents/video_baxter_chair.gif)|![](docs/img/agents/video_cursor_round_table.gif)|![](docs/img/agents/video_jaco_tvunit.gif)|![](docs/img/agents/video_panda_table.gif)|
| :---: | :---: | :---: |:---: |:---: |
| Sawyer | Baxter | Cursors | Jaco | Panda |

The IKEA Furniture Assembly environment provides:
- Comprehensive modeling of **furniture assembly** task
- 60+ furniture models
- Configurable and randomized backgrounds, lighting, textures
- Realistic robot simulation for Baxter, Sawyer, Jaco, Panda, and more
- Gym interface for easy RL training
- Reinforcement learning and imitation learning benchmarks
- Teleopration with 3D mouse/VR controller

<br>

## (1) Installation

### Prerequisites
- Ubuntu 18.04, MacOS Catalina, Windows10
- Python 3.7 (pybullet may not work with Python 3.8 or higher)
- Mujoco 2.0
- Unity 2018.4.23f1 ([Install using Unity Hub](https://unity3d.com/get-unity/download))

### Installation
```bash
git clone https://github.com/clvrai/furniture.git
cd furniture
pip install -e .
```


### (2) Example training and testing
We provide example commands for `table_lack_0825`. You can simply change the furniture name to test on other furniture models.
For evaluation, you can add `--is_train False --num_eval 50` to the training command:
```shell script
# train
python -m run --algo sac --run_prefix sac_table_lack_0825 --env FurnitureSawyerDenseRewardEnv --furniture_name table_lack_0825

