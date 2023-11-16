
import gym
import numpy as np
import math
import pybullet as p
import matplotlib.pyplot as plt
import pandas as pd

from Env.resources.x10car_v1 import x10car_v1
from Env.resources.x10car_v0 import x10car_v0
from Env.resources.x_mycar import x_mycar
from Env.resources.x_mycar2 import x_mycar2

from Env.resources.Walls_20 import walls_20
from Env.resources.Ground_20 import ground_20

from Env.resources.Rock_1 import rock_1
from Env.resources.Rock_2 import rock_2
from Env.resources.Rock_3 import rock_3
from Env.resources.Rock_4 import rock_4
from Env.resources.Rock_5 import rock_5
from Env.resources.Rock_6 import rock_6
from Env.resources.Rock_7 import rock_7
from Env.resources.ball1 import ball1
from Env.resources.ball2 import ball2
from Env.resources.ball3 import ball3
from Env.resources.ball4 import ball4
from Env.resources.apple import apple
from Env.envs.envs.agents import RaceCar
from Env.envs.envs.agents import Drone
from Env.envs.envs.agents import MJCFAgent
from Env.resources.Tree_1 import tree_1
from Env.resources.Tree_2 import tree_2
from Env.resources.Tree_3 import tree_3
from Env.resources.Tree_4 import tree_4

class X10Car_Env_out20(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32))

        distlow = np.zeros(100)
        disthigh = [1000]*100

        # Observation space for 90 deg FoV lidar with 100 points. For alternate FoV and number of points modify get_distance() in the x10car file

        self.observation_space = gym.spaces.Box(
            low=np.array(distlow, dtype=np.float32),
            high=np.array(disthigh, dtype=np.float32))
        self.np_random, _ = gym.utils.seeding.np_random()

        self.client = p.connect(p.GUI)
        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1/30, self.client)
        # 设置相机位置
        camera_distance = 15  # 相机距离目标点的距离
        camera_yaw = 150  # 相机水平旋转角度
        camera_pitch = -45  # 相机垂直旋转角度
        camera_target_position = [0, 0, 0]  # 相机目标点位置，设为场景中心点或感兴趣区域的坐标

        # 设置相机视角
        p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_position)
        self.car = None
        self.wall = None
        self.plane = None
        self.rock_1 = None
        self.rock_2 = None
        self.rock_3 = None
        self.rock_4 = None
        self.rock_5 = None
        self.rock_6 = None
        self.rock_7 = None
        self.tree_1 = None
        self.tree_2 = None
        self.tree_3 = None
        self.tree_4 = None

        self.done = False
        self.prev_dist_to_wall = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.trajectory = pd.DataFrame({'Agent Steps':[], 'Car Observation X':[], 'Car Observation Y':[], 'Minimum Distance':[], 'Reward':[]})
        self.writeReward = pd.DataFrame({'Reward':[]})
        self.store_agent_steps = 0
        self.store_reward = 0
        self.episode_reward = 0
        self.writeEpisodeReward = pd.DataFrame({'Episode Reward':[]})
        self.reset()

    def step(self, action):

        self.car.apply_action([0, 0])
        self.car2.apply_action([0.5, 0])# perform action
        self.car3.apply_action([0.5, 0])
        self.car4.apply_action([0.5, 0])
        self.air.apply_action([0.3, 0.3, 0.3, 0.3])
        # self.M.apply_action(action)

        p.stepSimulation()
        car_ob = self.car.get_observation() # return state

        dist_to_wall = self.car.get_distance()
        min_dist_to_wall = min(dist_to_wall) 
        #print(min_dist_to_wall)

        self.store_agent_steps = self.store_agent_steps + 1
        self.episode_reward = self.episode_reward
        
        # Compute reward 

        reward = 0.005 * (min_dist_to_wall**2) + 5.0*((min(max(action[0], 0), 1))**2) - 2.0*((max(min(action[1], 0.36), -0.36))**2)
        
        if (min_dist_to_wall > 2.0 and min_dist_to_wall < 2.5):
            reward = reward + 2.0
            
        if (min_dist_to_wall < 1.0):
            reward = reward - 50.0            
        
        self.writeReward = pd.concat([self.writeReward, pd.DataFrame({'Reward':[reward]})], ignore_index=True)
        self.episode_reward = self.episode_reward + reward
        
        # Record reward at each step every 10,000,000 steps. Modify number for higher data recording frequency. Training may be slower with higher writing frequency

        if(self.store_agent_steps == 10000000):
            self.writeReward.to_csv('reward_out20.csv', index = False)
            self.store_agent_steps = 0
        
        # Terminate episode and record episode reward

        if (min_dist_to_wall < 0.6):
            self.store_reward = 0
            self.done = True
            self.writeEpisodeReward = pd.concat([self.writeEpisodeReward, pd.DataFrame({'Episode Reward':[self.episode_reward]})], ignore_index=True)
            self.writeEpisodeReward.to_csv('episode_reward_out20.csv', index = False)
            self.episode_reward = 0

        self.store_reward = self.store_reward + reward            
        
        # Record trajectory position data if agent achieves 5000 steps. Modify number to store trajectories for alternate target steps

        if (self.store_agent_steps <= 5000):
            self.trajectory = pd.concat([self.trajectory, pd.DataFrame({'Agent Steps':[self.store_agent_steps], 'Car Observation X':[car_ob[0]], 'Car Observation Y': [car_ob[1]],'Minimum Distance':[min_dist_to_wall], 'Reward':[self.store_reward]})], ignore_index=True)
            if (self.store_agent_steps == 5000):
                self.trajectory.to_csv('trajectory_out20.csv', index = False)
                print(f'stored trajectory for reward {reward} and agent_steps {self.store_agent_steps}')
                self.store_agent_steps = 0

        ob = np.array(dist_to_wall, dtype=np.float32)
        return ob, reward, self.done, dict()
        # return ob, reward,False, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.81)
        # Reload environment assets
        self.plane = ground_20(self.client)
        self.car = x10car_v0(self.client)
        self.car2 = x_mycar(self.client)
        self.car3 = x_mycar2(self.client)
        self.car4 = RaceCar(self.client)
        self.air = Drone(self.client)
        # self.M = MJCFAgent(self.client)

        self.wall = walls_20(self.client)
        self.rock_1 = rock_1(self.client)
        self.rock_2 = rock_2(self.client)
        self.rock_3 = rock_3(self.client)
        self.rock_4 = rock_4(self.client)
        self.rock_5 = rock_5(self.client)
        self.rock_6 = rock_6(self.client)
        self.rock_7 = rock_7(self.client)
        self.ball1 = ball1(self.client)
        self.ball2 = ball2(self.client)
        # self.ball3 = ball3(self.client)
        # self.ball4 = ball4(self.client)
        # self.apple = apple(self.client)
        # self.tree_1 = tree_1(self.client)
        # self.tree_2 = tree_2(self.client)
        self.tree_3 = tree_3(self.client)
        self.tree_4 = tree_4(self.client)

        self.done = False

        self.prev_dist_to_wall = self.car.get_distance()
        
        return np.array(self.prev_dist_to_wall, dtype=np.float32)

    def render(self, mode='human'):
        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((100, 100, 4)))

        # Base information
        car_id, client_id = self.car.get_ids()
        wall_id = self.wall.get_ids()

        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                   nearVal=0.01, farVal=100)
        pos, ori = [list(l) for l in
                    p.getBasePositionAndOrientation(car_id, client_id)]
        pos[2] = 0.2

        # Rotate camera direction
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = p.computeViewMatrix([pos[0], pos[1], pos[2]+7.5], pos + camera_vec, up_vec)

        # Display image
        frame = p.getCameraImage(400, 400, view_matrix, proj_matrix)[2]
        frame = np.reshape(frame, (400, 400, 4))
        self.rendered_img.set_data(frame)
        plt.draw()
        plt.pause(.00001)

    def close(self):
        p.disconnect(self.client)