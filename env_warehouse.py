 # Adapted from Isaac Sim Jetbot Gymnasium example

import math

import carb
import gymnasium
import numpy as np
from gymnasium import spaces


class JetBotEnv(gymnasium.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        skip_frame=1,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        max_episode_length=2048,
        seed=1,
        headless=True,
    ) -> None:
        from omni.isaac.kit import SimulationApp
        CONFIG = {
            "width": 1280,
            "height": 720,
            "window_width": 1920,
            "window_height": 1080,
            "headless": True,
            "renderer": "RayTracedLighting",
            "display_options": 3286,  # Set display options to show default grid
            "anti_aliasing": 0,
        }


        self.headless = headless
        #self._simulation_app = SimulationApp({"headless": self.headless, "anti_aliasing": 0})
        self._simulation_app = SimulationApp(launch_config=CONFIG)


        self._skip_frame = skip_frame
        self._dt = physics_dt * self._skip_frame
        self._max_episode_length = max_episode_length
        self._steps_after_reset = int(rendering_dt / physics_dt)

        self._simulation_app.set_setting("/app/window/drawMouse", True)
        self._simulation_app.set_setting("/app/livestream/proto", "ws")
        self._simulation_app.set_setting("/app/livestream/websocket/framerate_limit", 120)
        self._simulation_app.set_setting("/ngx/enabled", False)

        from omni.isaac.core.utils.extensions import enable_extension
        enable_extension("omni.kit.livestream.native")
       # enable_extension("omni.isaac.ros2_bridge")

        from omni.isaac.core import World
        from omni.isaac.core.objects import VisualCuboid
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
        from omni.isaac.wheeled_robots.robots import WheeledRobot
        import omni.kit.commands
        import omni.isaac.core.utils.stage as stage_utils

        self._my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0)
        self._my_world.scene.add_default_ground_plane()
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return
        jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
        self.jetbot = self._my_world.scene.add(
            WheeledRobot(
                prim_path="/World/jetbot",
                name="my_jetbot",
                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                create_robot=True,
                usd_path=jetbot_asset_path,
                position=np.array([0, -5.0, 0.03]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),                
            )
        )
        robot_prim = self.jetbot.prim
        robot_prim.GetAttribute("xformOp:scale").Set((5.0, 5.0, 5.0))
        #print(f"Jetbot created at {self.jetbot.get_world_pose()} with attrbutes {robot_prim.GetAttributes()}")
        self.jetbot_controller = DifferentialController(name="simple_control", wheel_radius=0.0325, wheel_base=0.1125)
        self.goal = self._my_world.scene.add(
            VisualCuboid(
                prim_path="/World/new_cube_1",
                name="visual_cube",
                position=np.array([0.0, 4.0, 0.15]),
                size=0.3,
                color=np.array([1.0, 0, 0]),
            )
        )
        #print(f"Goal cube created at {self.goal.get_world_pose()}, with attributes {self.goal.prim.GetAttributes()}")
        
        #File path to the warehouse USD file
        warehouse_usd_file_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"

        # Add the warehouse to the scene
        warehouse_prim = stage_utils.add_reference_to_stage(usd_path=warehouse_usd_file_path, prim_path="/World/warehouse")
        #print(f"Warehouse created at {warehouse_prim.GetPath()}")
        # Subscribe to collision events        
        subscription_id = omni.kit.commands.execute(
            "IsaacRegisterCustomEventHandler",
            event_type="collision",
            callback_fn=self.on_collision_event,
            immediate=True)

        self.seed(seed)
        self.reward_range = (-float("inf"), float("inf"))
        gymnasium.Env.__init__(self)
        self.action_space = spaces.Box(low=-1, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=float("inf"), high=float("inf"), shape=(16,), dtype=np.float32)

        self.max_velocity = 3
        self.max_angular_velocity = math.pi
        self.reset_counter = 0
        return

    def get_dt(self):
        return self._dt

    def step(self, action):
        previous_jetbot_position, _ = self.jetbot.get_world_pose()
        # action forward velocity , angular velocity on [-1, 1]
        raw_forward = action[0]
        raw_angular = action[1]

        # we want to force the jetbot to always drive forward
        # so we transform to [0,1].  we also scale by our max velocity
        forward = (raw_forward + 1.0) / 2.0
        forward_velocity = forward * self.max_velocity

        # we scale the angular, but leave it on [-1,1] so the
        # jetbot can remain an ambiturner.
        angular_velocity = raw_angular * self.max_angular_velocity

        # we apply our actions to the jetbot
        for i in range(self._skip_frame):
            self.jetbot.apply_wheel_actions(
                self.jetbot_controller.forward(command=[forward_velocity, angular_velocity])
            )
            self._my_world.step(render=False)

        observations = self.get_observations()
        info = {}
        done = False
        truncated = False
        if self._my_world.current_time_step_index - self._steps_after_reset >= self._max_episode_length:
            done = True
            truncated = True
        goal_world_position, _ = self.goal.get_world_pose()
        current_jetbot_position, _ = self.jetbot.get_world_pose()
        previous_dist_to_goal = np.linalg.norm(goal_world_position - previous_jetbot_position)
        current_dist_to_goal = np.linalg.norm(goal_world_position - current_jetbot_position)
        reward = previous_dist_to_goal - current_dist_to_goal
        
        if current_dist_to_goal < 0.5:
            done = True
        return observations, reward, done, truncated, info

    def reset(self, seed=None):
        self._my_world.reset()
        self.reset_counter = 0
        # randomize goal location in circle around robot
        # alpha = 2 * math.pi * np.random.rand()
        # r = 1.00 * math.sqrt(np.random.rand()) + 0.20
        # self.goal.set_world_pose(np.array([math.sin(alpha) * r, math.cos(alpha) * r, 0.05]))
        self.goal.set_world_pose(np.array([(np.random.rand()-0.5)*4, 4+np.random.rand(), 0.15]))
        #self.goal.set_world_pose(np.array([0, 4, 0.05]))
        observations = self.get_observations()
        self._simulation_app.update()
        #print(f"Resetting environment, goal at {self.goal.get_world_pose()}")
        return observations, {}

    def get_observations(self):
        #self._my_world.render()
        jetbot_world_position, jetbot_world_orientation = self.jetbot.get_world_pose()
        jetbot_linear_velocity = self.jetbot.get_linear_velocity()
        jetbot_angular_velocity = self.jetbot.get_angular_velocity()
        goal_world_position, _ = self.goal.get_world_pose()
        obs = np.concatenate(
            [
                jetbot_world_position,
                jetbot_world_orientation,
                jetbot_linear_velocity,
                jetbot_angular_velocity,
                goal_world_position,
            ]
        )
        return obs

    def render(self, mode="human"):
        self._simulation_app.update()
        return

    def close(self):
        self._simulation_app.close()
        return

    def seed(self, seed=None):
        self.np_random, seed = gymnasium.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]

    def on_collision_event(prim_path_1, prim_path_2):
    # Your collision handling logic here:
        
        print(f"Collision: {prim_path_1} collided with {prim_path_2}")
        # Implement your collision response (e.g., reverse, stop, etc.)
