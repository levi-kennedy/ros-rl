
import signal
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Float64, MultiArrayDimension
from geometry_msgs.msg import Twist
from rclpy.executors import MultiThreadedExecutor
import threading

from env_no_warehouse import JetBotEnv
from stable_baselines3 import PPO
import numpy as np
import time

# specify topic names
action_topic = 'jetbot_action'
observation_topic = 'jetbot_observation'
reward_topic = 'jetbot_reward'
goal_achieved_topic = 'jetbot_goal_achieved'


class JetBotAgent(Node):
    def __init__(self, policy_path=None):
        super().__init__('jetbot_agent')
        self.model = PPO.load(policy_path)
        self.act_publisher_ = self.create_publisher(Twist, action_topic, 10)
        self.obs_subscription = self.create_subscription(Float64MultiArray, observation_topic, self.observation_callback, 10)


    # publish the action to the topic
    def publish_action(self, action):
        msg = Twist()
        msg.linear.x = action[0]
        msg.angular.z = action[1]
        self.get_logger().info(f'Jetbot Published Action: {msg}')
        self.act_publisher_.publish(msg)


    # callback function for the observation topic
    def observation_callback(self, msg):
        self.get_logger().info(f'Jetbot Received Observation: {msg}')
        obs = msg.data       
        action, _ = self.model.predict(observation=obs, deterministic=True)
        action = action.tolist()
        self.get_logger().info(f'Jetbot Predicted Action: {action}')
        self.publish_action(action)
        



class RosSimEnv(Node):
    def __init__(self, env):
        super().__init__('ros_sim_env')        
        self.sim_env = env       
        self.obs_publisher_ = self.create_publisher(Float64MultiArray, observation_topic, 10)
        self.rw_publisher_ = self.create_publisher(Float64, reward_topic, 10)
        self.act_subscription = self.create_subscription(Twist, action_topic, self.action_callback, 10)        


    # callback function for the action topic
    def action_callback(self, msg):
        self.get_logger().info(f'Env Received Action msg: {msg}')
        action = [msg.linear.x, msg.angular.z]
        self.get_logger().info(f'Env Stepping with Action: {action}')
        obs, reward, done, truncated, info = self.sim_env.step(action)
        obs = obs.tolist()
        reward = reward.tolist()
        self.get_logger().info(f'Env Computed Obs: {obs}, Done: {done}, Reward: {reward}')
        if done:
            # reset the environment and send the observation to the agent
            obs, _ = self.sim_env.reset()
            obs = obs.tolist()
        self.publish_observation(obs)
        self.publish_reward(reward)        


    # publish observations to the topic
    def publish_observation(self, obs):
        msg = Float64MultiArray()
        msg.data = obs
        mad = MultiArrayDimension()
        mad.label = 'observation'
        mad.size = len(obs)
        mad.stride = len(obs)
        msg.layout.dim.append(mad)
        self.get_logger().info(f'Env Published Observation: {msg}')
        self.obs_publisher_.publish(msg)


    # publish rewards to the topic
    def publish_reward(self, reward):
        msg = Float64()
        msg.data = reward
        self.get_logger().info(f'Env Published Reward: {msg}')
        self.rw_publisher_.publish(msg)

    

# Create callback for spinning the executor
def spin_executor(executor):
    executor.spin()

def main(args=None):
    rclpy.init(args=args)
    jetbot_env = JetBotEnv(headless=True)    

    # initialize the ROS nodes
    ros_sim_env = RosSimEnv(jetbot_env)
    policy_path = "./cnn_policy/jetbot_policy_550K.zip"
    ros_agent = JetBotAgent(policy_path)

    # add the nodes to the executor
    executor = MultiThreadedExecutor()
    executor.add_node(ros_sim_env)
    executor.add_node(ros_agent)

    # Create and start a separate thread for spinning the executor
    executor_thread = threading.Thread(target=spin_executor, args=(executor,))
    executor_thread.start()

     # reset the environment and send the first observation to the agent
    obs, _ = ros_sim_env.sim_env.reset()
    obs = obs.tolist()
    ros_sim_env.publish_observation(obs)

    shut_flag = False
    def signal_handler(sig, frame):
        print("Signal handler ........... Shutting down")
        executor.shutdown()
        ros_sim_env.destroy_node()
        ros_agent.destroy_node()
        jetbot_env.close()
        rclpy.shutdown()
        executor_thread.join()
        shut_flag = True
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    while not shut_flag:
        pass

if __name__ == '__main__':
    main()
