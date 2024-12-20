# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations. """

from collections import OrderedDict
import math
import random
import numpy as np
import carla
from eval import box_2_polygon, caluclate_tp_fp, eval_final_results
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.local_planner import RoadOption
from agents.navigation.behavior_types import Cautious, Aggressive, Normal

from agents.tools.misc import get_speed, positive, is_within_distance, compute_distance
from detector import Detector  # pylint: disable=import-rror
from shapely.geometry import Polygon

class BehaviorAgent(BasicAgent):
    """
    BehaviorAgent implements an agent that navigates scenes to reach a given
    target destination, by computing the shortest possible path to it.
    This agent can correctly follow traffic signs, speed limitations,
    traffic lights, while also taking into account nearby vehicles. Lane changing
    decisions can be taken by analyzing the surrounding environment such as tailgating avoidance.
    Adding to these are possible behaviors, the agent can also keep safety distance
    from a car in front of it by tracking the instantaneous time to collision
    and keeping it in a certain range. Finally, different sets of behaviors
    are encoded in the agent, from cautious to a more aggressive ones.
    """

    def __init__(self, vehicle, behavior='normal', opt_dict={}, map_inst=None, grp_inst=None):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param behavior: type of agent to apply
        """

        super().__init__(vehicle, opt_dict=opt_dict, map_inst=map_inst, grp_inst=grp_inst)
        self._look_ahead_steps = 0
        # self._count = 0

        # Vehicle information
        self._speed = 0
        self._speed_limit = 0
        self._direction = None
        self._incoming_direction = None
        self._incoming_waypoint = None
        self._min_speed = 5
        self._behavior = None
        self._sampling_resolution = 4.5

        self.bound_x = vehicle.bounding_box.extent.x
        self.bound_y = vehicle.bounding_box.extent.y
        self.bound_z = vehicle.bounding_box.extent.z

        # Parameters for agent behavior
        if behavior == 'cautious':
            self._behavior = Cautious()

        elif behavior == 'normal':
            self._behavior = Normal()

        elif behavior == 'aggressive':
            self._behavior = Aggressive()

        # Initalize the detector
        self._detector = Detector()

        # Evaluate detection results
        self.result_stat = {
                            0.05: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                            0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                            0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                            0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}\
        
        # Bounding boxes
        self.bbox = {}

    def destroy(self):
        eval_final_results(self.result_stat, global_sort_detections=True)
        
    def sensors(self):  # pylint: disable=no-self-use
        sensors = self._detector.sensors()
        for s in sensors:
            s['x'] = s['x']*self.bound_x
            s['y'] = s['y']*self.bound_y
            s['z'] = s['z']*self.bound_z
        return sensors
    
    def _update_information(self):
        """
        This method updates the information regarding the ego
        vehicle based on the surrounding world.
        """
        self._speed = get_speed(self._vehicle)
        self._speed_limit = self._vehicle.get_speed_limit()
        self._local_planner.set_speed(self._speed_limit)
        self._direction = self._local_planner.target_road_option
        if self._direction is None:
            self._direction = RoadOption.LANEFOLLOW

        self._look_ahead_steps = int((self._speed_limit) / 10)

        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self._look_ahead_steps)
        if self._incoming_direction is None:
            self._incoming_direction = RoadOption.LANEFOLLOW

    def traffic_light_manager(self):
        """
        This method is in charge of behaviors for red lights.
        """
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        affected, _ = self._affected_by_traffic_light(lights_list)

        return affected

    def _tailgating(self, waypoint, vehicle_list):
        """
        This method is in charge of tailgating behaviors.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :param vehicle_list: list of all the nearby vehicles
        """

        left_turn = waypoint.left_lane_marking.lane_change
        right_turn = waypoint.right_lane_marking.lane_change

        left_wpt = waypoint.get_left_lane()
        right_wpt = waypoint.get_right_lane()

        behind_vehicle_state, behind_vehicle, _ = self._vehicle_obstacle_detected(vehicle_list, max(
            self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, low_angle_th=160)
        if behind_vehicle_state and self._speed < get_speed(behind_vehicle):
            if (right_turn == carla.LaneChange.Right or right_turn ==
                    carla.LaneChange.Both) and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the right!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         right_wpt.transform.location)
            elif left_turn == carla.LaneChange.Left and waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the left!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         left_wpt.transform.location)

    def collision_and_car_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        and managing possible tailgating chances.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
        """

        vehicle_list = self._world.get_actors().filter("*vehicle*")
        def dist(v): return v.get_location().distance(waypoint.transform.location)
        vehicle_list = [v for v in vehicle_list if dist(v) < 45 and v.id != self._vehicle.id]

        if self._direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
        else:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=30)

            # Check for tailgating
            if not vehicle_state and self._direction == RoadOption.LANEFOLLOW \
                    and not waypoint.is_junction and self._speed > 10 \
                    and self._behavior.tailgate_counter == 0:
                self._tailgating(waypoint, vehicle_list)

        return vehicle_state, vehicle, distance

    def pedestrian_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        with any pedestrian.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a walker nearby, False if not
            :return vehicle: nearby walker
            :return distance: distance to nearby walker
        """

        walker_list = self._world.get_actors().filter("*walker.pedestrian*")
        def dist(w): return w.get_location().distance(waypoint.transform.location)
        walker_list = [w for w in walker_list if dist(w) < 10]

        if self._direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=60)

        return walker_state, walker, distance

    def car_following_manager(self, vehicle, distance, debug=False):
        """
        Module in charge of car-following behaviors when there's
        someone in front of us.

            :param vehicle: car to follow
            :param distance: distance from vehicle
            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """

        vehicle_speed = get_speed(vehicle)
        delta_v = max(1, (self._speed - vehicle_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)

        # Under safety time distance, slow down.
        if self._behavior.safety_time > ttc > 0.0:
            target_speed = min([
                positive(vehicle_speed - self._behavior.speed_decrease),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Actual safety distance area, try to follow the speed of the vehicle in front.
        elif 2 * self._behavior.safety_time > ttc >= self._behavior.safety_time:
            target_speed = min([
                max(self._min_speed, vehicle_speed),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Normal behavior.
        else:
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        return control

    def gt_box_vertice_sequence(self, box):
        box = [box[1],
               box[3],
               box[7],
               box[5],
               box[0],
               box[2],
               box[6],
               box[4]]
        return np.array(box)

    def actor_detected(self, actor, detection_results, actor_class):
        """
            actor: carla.Actor
            detection_results: dict
            actor_class: 0 for vehicle, 1 for pedestrian, 2 for cyclist
        """
        # Prepare GT box
        
        gt_box = [[v.x, v.y, v.z] for v in actor.bounding_box.get_world_vertices(actor.get_transform())]
        gt_box = self.gt_box_vertice_sequence(gt_box)
        gt_polygon = box_2_polygon(gt_box)
        # Compare detection boxes with GT boxes
        if ("det_boxes" not in detection_results) or ("det_class" not in detection_results):
            return False
        det_boxes = detection_results["det_boxes"]
        classes = detection_results["det_class"]
        for obj_id in range(len(classes)):
            if classes[obj_id] != actor_class:
                continue
            polygon = box_2_polygon(det_boxes[obj_id])
            
            union = polygon.union(gt_polygon).area
            if union == 0:
                return False
            try: 
                iou = polygon.intersection(gt_polygon).area / union
                if iou > 0.5:
                    return True
            except:
                print(polygon)
        return False

    def gt_actors(self):
        """
        Get all the ground truth actors in the scene
        """
        actor_list = self._world.get_actors()
        vehicles = actor_list.filter("*vehicle*")
        walkers = actor_list.filter("*walker.pedestrian*")
        detection_results = dict()
        detection_results["det_boxes"] = []
        detection_results["det_class"] = []
        detection_results["det_score"] = []
        transform = self._vehicle.get_transform()
        def dist(l):
            return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z)**2)
        for v in vehicles:
            if dist(v.get_location()) > 50:
                continue
            if v.id == self._vehicle.id:
                continue
            bbox = [[v.x, v.y, v.z] for v in v.bounding_box.get_world_vertices(v.get_transform())]
            bbox = self.gt_box_vertice_sequence(bbox)
            detection_results["det_boxes"].append(bbox)
            detection_results["det_class"].append(0)
            detection_results["det_score"].append(1.0)
        for w in walkers:
            if dist(w.get_location()) > 50:
                continue
            bbox = [[w.x, w.y, w.z] for w in w.bounding_box.get_world_vertices(w.get_transform())]
            bbox = self.gt_box_vertice_sequence(bbox)
            detection_results["det_boxes"].append(bbox)
            detection_results["det_class"].append(1)
            detection_results["det_score"].append(1.0)
        detection_results["det_boxes"] = np.array(detection_results["det_boxes"])
        detection_results["det_class"] = np.array(detection_results["det_class"])
        detection_results["det_score"] = np.array(detection_results["det_score"])
        return detection_results

    def get_sensor_transforms(self):
        left_cam_2_world = np.array(self.sensor_interface._sensors_objects['Left'].get_transform().get_matrix())
        right_cam_2_world = np.array(self.sensor_interface._sensors_objects['Right'].get_transform().get_matrix())
        forward_cam_2_world = np.array(self.sensor_interface._sensors_objects['Forward'].get_transform().get_matrix())

        return {
            "left2world": left_cam_2_world,
            "right2world": right_cam_2_world,
            "forward2world": forward_cam_2_world,
            "LIDAR2world": np.array(self.sensor_interface._sensors_objects['LIDAR'].get_transform().get_matrix()),
            "Side-Left-LIDAR2world": np.array(self.sensor_interface._sensors_objects['Side-Left-LIDAR'].get_transform().get_matrix()),
            "Side-Right-LIDAR2world": np.array(self.sensor_interface._sensors_objects['Side-Right-LIDAR'].get_transform().get_matrix()),
            "Front-LIDAR2world": np.array(self.sensor_interface._sensors_objects['Front-LIDAR'].get_transform().get_matrix()),
            "Back-LIDAR2world": np.array(self.sensor_interface._sensors_objects['Back-LIDAR'].get_transform().get_matrix()),
        }

    def run_step(self, sensor_transforms, debug=False):
        """
        Execute one step of navigation.

            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        sensor_data = self.get_sensor_data()
        detections = self._detector.detect(sensor_data, sensor_transforms)
        gt_detections = self.gt_actors()

        # Evaluate detection results
        det_boxes = np.array([[[0,0,0],[0,1,0],[1,1,0],[1,0,0],[0,0,1],[0,1,1],[1,1,1],[1,0,1]]])
        det_score = np.array([0])
        if "det_boxes" in detections:
            det_boxes = detections["det_boxes"]
        if "det_score" in detections:
            det_score = detections["det_score"]
        
        frame_number = next(iter(sensor_data.values()))[0]

        self.bbox = {
            'frame':frame_number,
            'gt_det':gt_detections,
            'det':detections
        }

        if "det_boxes" in detections and gt_detections["det_boxes"].shape[0] > 0:
            # if self._count == 0 and gt_detections["det_boxes"].shape[0] > 0:
            print("====================================")
            print("Frame number:", frame_number)
            print("GT detections:\n", gt_detections["det_boxes"])
            print("Detections:\n", det_boxes)
            print("gt box shape:", gt_detections["det_boxes"].shape)
            print("det box shape:", det_boxes.shape)
            max_iou = float("-inf") 

            for gt_box in gt_detections["det_boxes"]:
                for det_box in det_boxes:
                    # Compute the intersection
                    ixmin = np.maximum(gt_box[:, 0].min(), det_box[:, 0].min())
                    iymin = np.maximum(gt_box[:, 1].min(), det_box[:, 1].min())
                    izmin = np.maximum(gt_box[:, 2].min(), det_box[:, 2].min())
                    ixmax = np.minimum(gt_box[:, 0].max(), det_box[:, 0].max())
                    iymax = np.minimum(gt_box[:, 1].max(), det_box[:, 1].max())
                    izmax = np.minimum(gt_box[:, 2].max(), det_box[:, 2].max())

                    iw = np.maximum(ixmax - ixmin, 0)
                    ih = np.maximum(iymax - iymin, 0)
                    id = np.maximum(izmax - izmin, 0)

                    intersection = iw * ih * id

                    # Compute the volume of each box
                    gt_volume = (gt_box[:, 0].max() - gt_box[:, 0].min()) * \
                                (gt_box[:, 1].max() - gt_box[:, 1].min()) * \
                                (gt_box[:, 2].max() - gt_box[:, 2].min())

                    det_volume = (det_box[:, 0].max() - det_box[:, 0].min()) * \
                                 (det_box[:, 1].max() - det_box[:, 1].min()) * \
                                 (det_box[:, 2].max() - det_box[:, 2].min())

                    union = gt_volume + det_volume - intersection

                    if union == 0:
                        continue

                    try:
                        iou = intersection / union
                        # if iou != 0.0:
                        #     print("Frame number:", frame_number)
                        #     print("GT detections:\n", gt_box)
                        #     print("Detections:\n", det_box)

                        max_iou = max(iou, max_iou)
                        # print(f"IoU between GT box and detection box: {iou}")
                    except Exception as e:
                        print(f"Error computing IoU: {e}")
            print("max IoU:", max_iou)
            print("====================================")
            # np.save("lidar_data.npy", sensor_data["LIDAR"][1])
            # np.save("lidar2world.npy", sensor_transforms["lidar2world"])
            # np.save("gt_det.npy", gt_detections["det_boxes"])
            # np.save("det.npy", det_boxes)
        # self._count += 1
            
        caluclate_tp_fp(det_boxes, det_score, gt_detections["det_boxes"], self.result_stat, iou_thresh=0.05)
        caluclate_tp_fp(det_boxes, det_score, gt_detections["det_boxes"], self.result_stat, iou_thresh=0.3)
        caluclate_tp_fp(det_boxes, det_score, gt_detections["det_boxes"], self.result_stat, iou_thresh=0.5)
        caluclate_tp_fp(det_boxes, det_score, gt_detections["det_boxes"], self.result_stat, iou_thresh=0.7)

        self._update_information()

        control = None
        if self._behavior.tailgate_counter > 0:
            self._behavior.tailgate_counter -= 1

        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)

        # 1: Red lights and stops behavior
        if self.traffic_light_manager():
            return self.emergency_stop()

        # 2.1: Pedestrian avoidance behaviors
        walker_state, walker, w_distance = self.pedestrian_avoid_manager(ego_vehicle_wp)

        if walker_state:
            if self.actor_detected(walker, detections, 1):
                # Distance is computed from the center of the two cars,
                # we use bounding boxes to calculate the actual distance
                distance = w_distance - max(
                    walker.bounding_box.extent.y, walker.bounding_box.extent.x) - max(
                        self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

                # Emergency brake if the car is very close.
                if distance < self._behavior.braking_distance:
                    return self.emergency_stop()

        # 2.2: Car following behaviors
        vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)

        if vehicle_state:
            if self.actor_detected(vehicle, detections, 0):
                # Distance is computed from the center of the two cars,
                # we use bounding boxes to calculate the actual distance
                distance = distance - max(
                    vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(
                        self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

                # Emergency brake if the car is very close.
                if distance < self._behavior.braking_distance:
                    return self.emergency_stop()
                else:
                    control = self.car_following_manager(vehicle, distance)

        # 3: Intersection behavior
        elif self._incoming_waypoint.is_junction and (self._incoming_direction in [RoadOption.LEFT, RoadOption.RIGHT]):
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - 5])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # 4: Normal behavior
        else:
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        return control

    def emergency_stop(self):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control
