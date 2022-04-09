#!/usr/bin/env python3
import rospy
import rospkg
import numpy as np
import argparse
import time

from graic_msgs.msg import ObstacleList, ObstacleInfo
from graic_msgs.msg import LocationInfo, WaypointInfo
from ackermann_msgs.msg import AckermannDrive
from carla_msgs.msg import CarlaEgoVehicleControl
from carla_msgs.msg import CarlaLaneInvasionEvent
from graic_msgs.msg import LaneList
from graic_msgs.msg import LaneInfo
from carla_msgs.msg import CarlaEgoVehicleControl
from std_msgs.msg import String, Float32

import sys
import copy
import os
import yaml
import logging
import argparse
from ddpg import DDPG

logger = logging.getLogger(__name__)
#logger = logging.basicConfig(filename='my.log')

def load_config(config_path="config.yml"):
    if os.path.isfile(config_path):
        f = open(config_path)
        return yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise Exception("Configuration file is not found in the path: "+config_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Train RL agent")
    parser.add_argument(
        "--agent_name",
        type=str,
        default='DDPG',
        choices=['DQN', 'DDPG', 'SAC', 'A2C', 'PPO_GAE', 'MBRL'],
    )
    parser.add_argument(
        "--port",
        type=int,
        default=2000,
    )
    args = parser.parse_args()

    return args


# parameters for the gym_carla environment
params = {
    'number_of_vehicles': 100,
    'number_of_walkers': 0,
    'display_size': 256,  # screen size of bird-eye render
    'max_past_step': 1,  # the number of past steps to draw
    'dt': 0.1,  # time interval between two frames
    'discrete': False,  # whether to use discrete control space
    'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
    'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
    'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
    'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
    'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
    'port': 2000,  # connection port
    'town': 'Town03',  # which town to simulate
    'task_mode': 'random',  # mode of the task, [random, roundabout (only for Town03)]
    'max_time_episode': 1000,  # maximum timesteps per episode
    'max_waypt': 12,  # maximum number of waypoints
    'obs_range': 32,  # observation range (meter)
    'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
    'd_behind': 12,  # distance behind the ego vehicle (meter)
    'out_lane_thres': 2.0,  # threshold for out of lane
    'desired_speed': 8,  # desired speed (m/s)
    'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    'display_route': True,  # whether to render the desired route
    'pixor_size': 64,  # size of the pixor labels
    'pixor': False,  # whether to output PIXOR observation

    'role_name': 'ego_vehicle',  # whether to output PIXOR observation
}


class VehiclePerception:
    def __init__(self, role_name='ego_vehicle', test=False):
        rospy.Subscriber("/carla/%s/location" % role_name, LocationInfo,
                         self.locationCallback)
        rospy.Subscriber("/carla/%s/obstacles" % role_name, ObstacleList,
                         self.obstacleCallback)
        rospy.Subscriber("/carla/%s/lane_markers" % role_name, LaneInfo,
                         self.lanemarkerCallback)
        rospy.Subscriber("/carla/%s/waypoints" % role_name, WaypointInfo,
                         self.waypointCallback)
        rospy.Subscriber("/carla/%s/waypoints" % role_name, WaypointInfo,
                         self.waypointCallback)
        rospy.Subscriber("/carla/%s/lane_invasion"%role_name, CarlaLaneInvasionEvent,
                         self.on_lane_invasion)
        rospy.Subscriber("/carla/%s/collision_detail"%role_name, String, self.collisionCallback)

        self.position = None
        self.rotation = None
        self.velocity = None
        self.obstacleList = None
        self.lane_marker = None
        self.waypoint = None
        self.score = 0
        self.prevScore = 0
        self.test = test
        self.collisionInfo = 0
        self.linecross = 0
        self.old = None

    def collisionCallback(self, data):
        self.collisionInfo = 1 #data.data
        print("COLLISIONNOW:", self.collisionInfo)

    def on_lane_invasion(self, data):
        text = []
        crossing_solid = 0
        crossing_other = 0
        self.prevScore = self.score
        for marking in data.crossed_lane_markings:
            self.linecross += 1
            text.append(str(self.linecross)+" ")
            if marking is CarlaLaneInvasionEvent.LANE_MARKING_OTHER:
                text.append("Other")
                crossing_other += 1
                self.score -= 5
            elif marking is CarlaLaneInvasionEvent.LANE_MARKING_BROKEN:
                text.append("Broken")
                crossing_other += 1
                self.score -= 5
            elif marking is CarlaLaneInvasionEvent.LANE_MARKING_SOLID:
                text.append("Solid")
                crossing_solid += 1
                self.score -= 20
            else:
                text.append("Unknown ")
                crossing_other += 1
                self.score -= 5
        print("CROSSEDLANE: " , text,crossing_other, crossing_solid)

    def locationCallback(self, data):
        self.position = (data.location.x, data.location.y)
        self.rotation = (np.radians(data.rotation.x),
                         np.radians(data.rotation.y),
                         np.radians(data.rotation.z))
        self.velocity = (data.velocity.x, data.velocity.y)

    def obstacleCallback(self, data):
        self.obstacleList = data.obstacles

    def lanemarkerCallback(self, data):
        self.lane_marker = data
        #old = self.old
        #self.old   = self.lane_marker.lane_markers_center.location[-1]
        #if self.old != old:
        #    print("LANEMARKER:", str(self.old))

    def waypointCallback(self, data):
        self.waypoint = data

    def ready(self):
        return (self.position
                is not None) and (self.rotation is not None) and (
                    self.velocity
                    is not None) and (self.obstacleList is not None) and (
                        self.lane_marker is not None
                    )  and (self.waypoint is not None)

    def clear(self):
        self.position = None
        self.rotation = None
        self.velocity = None
        self.obstacleList = None
        self.lane_marker = None
        self.waypoint = None

class VehicleController():

    def __init__(self, role_name='ego_vehicle'):
        # Publisher to publish the control input to the vehicle model
        self.controlPub = rospy.Publisher("/carla/%s/ackermann_control"%role_name, AckermannDrive, queue_size = 1)
        self.controlPub3 = rospy.Publisher("/carla/%s/vehicle_control_cmd"%role_name, CarlaEgoVehicleControl, queue_size=1)
        self.detect_dist = 30

    def publish_control3(self, control):
        self.controlPub3.publish(control)

    def stop(self):
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.acceleration = -20
        newAckermannCmd.speed = 0
        newAckermannCmd.steering_angle = 0
        self.controlPub.publish(newAckermannCmd)

    #return: a tuple of the distance and the waypoint orientation
    def get_lane_dis(self,waypt, x, y):
        vec = np.array([x - waypt[0], y - waypt[1]])
        lv = np.linalg.norm(np.array(vec))
        w = np.array([np.cos(waypt[2]/180*np.pi), np.sin(waypt[2]/180*np.pi)])
        cross = np.cross(w, vec/lv)
        dis = - lv * cross
        return dis, w

    def execute(self, currState, obstacleList, lane_marker, waypoint):
        # Get current position
        curr_x = currState[0][0]
        curr_y = currState[0][1]

        # Check whether any obstacles are in the front of the vehicle
        obs_front = False
        obs_left = False
        obs_right = False
        front_dist = 20

        yaw = currState[1][2]
        if obstacleList:
            for obs in obstacleList:
                for vertex in obs.vertices_locations:
                    dy = vertex.vertex_location.y - curr_y
                    dx = vertex.vertex_location.x - curr_x
                    rx = np.cos(-yaw) * dx - np.sin(-yaw) * dy 
                    ry = np.cos(-yaw) * dy + np.sin(-yaw) * dx

                    psi = np.arctan(ry/rx)
                    if rx > 0:
                        front_dist = np.sqrt(dy*dy + dx*dx)
                        # print("detected object is at {} away and {} radians".format(front_dist, psi))
                        # print("detected object is at {} away and {} radians".format(front_dist, psi))
                        if front_dist <= self.detect_dist:
                            if psi < 0.2 and psi > -0.2:
                                obs_front = True
                                #front_dis = front_dist
                            elif psi > 0.2:
                                obs_right = True
                                #right_dis = front_dist
                            elif psi < -0.2:
                                obs_left = True 
                                #left_dis = front_dist

        #last point in the centeral lane marker
        lm_pos   = lane_marker.lane_markers_center.location[-1]
        lm_state =  lane_marker.lane_state
        lateral_dis, w =  self.get_lane_dis((lm_pos.x, lm_pos.y, lm_pos.z), curr_x, curr_y)

        #print("lateral_dis=", lateral_dis, w)
        #print("position = ", curr_x, curr_y)
        #print("lm position = ",lm_pos.x, lm_pos.y, lm_pos.z)

        vx =  currState[2][0]
        vy =  currState[2][1]
        speed = np.sqrt(vx**2 + vy**2)

        #print(currState, "speed = ", speed)

        # longitudinal speed
        lspeed = np.array([vx, vy])
        lspeed_lon = np.dot(lspeed, w)

        # TODO: add this to state
        delta_yaw = np.arcsin(np.cross(w, np.array(np.array([np.cos(yaw), np.sin(yaw)]))))   
        state = np.array([lateral_dis, -delta_yaw, speed, np.float(obs_front), np.float(obs_left), np.float(obs_right)], dtype=object)

        return state, lspeed_lon

def publish_control(pub_control, control):
    #control.steering_angle = -control.steering_angle
    #control.steering_angle_velocity = -control.steering_angle_velocity
    pub_control.publish(control)

def init_model():
    args = load_config('./default_config.yaml')
    agent_name = "DDPG" 
    args_agent = args[agent_name]
    print(args_agent)
    agent = DDPG(args_agent)
    agent.model_path = "DDPG_model"
    os.makedirs(agent.model_path, exist_ok=True)

    # Setup logging
    logging.root.handlers = []
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(agent.model_path, agent.name + '.log'), mode='a', encoding=None, delay=False)
        ],
    )
    logger.setLevel(logging.INFO)

    logger.info('Finish loading environment and agent {}'.format(agent.name))

    mbrl_train_eps = 10
    random_episode = 10#0
    episodes = 10#10000
    test_episode = 10
    test_rewards = []
    return agent


def run_model(role_name, controller):
    #load DDPG model
    agent = init_model()

    #subscribe control
    perceptionModule = VehiclePerception(role_name=role_name)

    controlPub = rospy.Publisher("/carla/%s/ackermann_cmd" % role_name,
                                 AckermannDrive,
                                 queue_size=1)
    controlPub3 = rospy.Publisher("/carla/%s/vehicle_control_cmd" % role_name,
                                  CarlaEgoVehicleControl,
                                  queue_size=1)

    def shut_down():
        print("stop")
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.acceleration = -20
        newAckermannCmd.speed = 0
        newAckermannCmd.steering_angle = 0
        controlPub.publish(newAckermannCmd)

    def random_act():
        action = np.random.uniform(-1.0, 1.0, size=2)
        acc = action[0]*5
        steer = action[1]
        # Convert acceleration to throttle and brake
        if acc > 0:
            throttle = np.clip(acc / 3, 0, 1)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-acc / 8, 0, 1)
        act = CarlaEgoVehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))
        return act

    rospy.on_shutdown(shut_down)

    # newAckermannCmd = AckermannDrive()
    # newAckermannCmd.acceleration = 0
    # newAckermannCmd.speed = 0
    # newAckermannCmd.steering_angle = 0
    # publish_control(controlPub, newAckermannCmd)
    # control = controller.stop()
    # publish_control(controlPub, control)

    # start it
    import time
    time.sleep(1)
    controlPub3.publish(CarlaEgoVehicleControl())

    rate = rospy.Rate(20)  # 100 Hz 
    step = 0
    prev_state = np.array([np.float(0),np.float(0),np.float(0),np.float(0),np.float(0),np.float(0)], dtype=object)
    random_step = 1000
    action = [0, 0]

    while not rospy.is_shutdown():
        if perceptionModule.ready():
            step += 1
            #print(perceptionModule.position)
            # Get the current position and orientation of the vehicle
            ego_pos = (perceptionModule.position, perceptionModule.rotation, perceptionModule.velocity)
            state, lspeed_lon = controller.execute(ego_pos, perceptionModule.obstacleList, perceptionModule.lane_marker, perceptionModule.waypoint)

            print("============== CYCLE ===========", step)
            print("STATE:", state, "lspeed_on", lspeed_lon)

            # Score
            reward =  perceptionModule.score - perceptionModule.prevScore  -  perceptionModule.collisionInfo * 100
            print("REWARD:", reward)

            if step < random_step:
                act = random_act()
            else:
                act = agent.select_action(state, False)

            done = False

            agent.store_transition([prev_state, action, reward, state, done])
            agent.train()

            if step % 1000 == 900:
                agent.save_model()

            # Apply control
            #print("ACTION:", act.steer, act.throttle, act.brake);
            #input("press...")

            perceptionModule.clear()
            controlPub3.publish(act)
            prev_state = copy.deepcopy(state)
        rate.sleep()


if __name__ == "__main__":
    rospy.init_node("graic_agent_wrapper", anonymous=True)
    #roskpack = rospkg.RosPack()
    #config_path = roskpack.get_path('graic_config')
    #race_config = open(config_path + '/' + 'race_config', 'rb')
    #vehicle_typeid = race_config.readline().decode('ascii').strip()
    #sensing_radius = race_config.readline().decode('ascii').strip()

    role_name = 'ego_vehicle'
    controller = VehicleController()
    try:
        run_model(role_name, controller)
    except rospy.exceptions.ROSInterruptException:
        print("stop")
