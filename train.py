# Adapted from Isaac Sim Jetbot Gymnasium example

import argparse

import carb
import torch as th
from env_no_warehouse import JetBotEnv
import wandb
from wandb.integration.sb3 import WandbCallback

parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
args, unknown = parser.parse_known_args()



# set headles to false to visualize training
my_env = JetBotEnv(headless=True)

# in test mode we manually install sb3
if args.test is True:
    import omni.kit.pipapi

    omni.kit.pipapi.install("stable-baselines3==2.0.0", module="stable_baselines3")
    omni.kit.pipapi.install("tensorboard")

# import stable baselines
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.ppo import MlpPolicy
except Exception as e:
    carb.log_error(e)
    carb.log_error(
        "please install stable-baselines3 in the current python environment or run the following to install into the builtin python environment ./python.sh -m pip install stable-baselines3"
    )
    exit()

try:
    import tensorboard
except Exception as e:
    carb.log_error(e)
    carb.log_error(
        "please install tensorboard in the current python environment or run the following to install into the builtin python environment ./python.sh -m pip install tensorboard"
    )
    exit()


config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 2000000,
    "env_name": "JetBotEnv",
}
run = wandb.init(
    project="jb_sb3ppo",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)


log_dir = "./cnn_policy"


policy = config["policy_type"]
policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[dict(vf=[128, 128, 128], pi=[128, 128, 128])])

total_timesteps = config["total_timesteps"]

if args.test is True:
    total_timesteps = 10000

checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir, name_prefix="jetbot_policy_checkpoint")
model = PPO(
    policy,
    my_env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    n_steps=2560,
    batch_size=64,
    learning_rate=0.000125,
    gamma=0.9,
    ent_coef=7.5e-08,
    clip_range=0.3,
    n_epochs=5,
    gae_lambda=1.0,
    max_grad_norm=0.9,
    vf_coef=0.95,
    device="cuda:0",
    tensorboard_log=f"runs/{run.id}",
)
model.learn(
    total_timesteps=total_timesteps, 
    callback=[
        WandbCallback(
            gradient_save_freq=50,
            model_save_freq=50000,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
        CheckpointCallback(
            save_freq=50000, 
            save_path=log_dir, 
            name_prefix="jetbot_policy_checkpoint")
    ],
    progress_bar=True,
    )

model.save(log_dir + "/jetbot_policy")

my_env.close()
