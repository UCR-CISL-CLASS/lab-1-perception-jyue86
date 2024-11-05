from mmdet3d.apis import init_model, inference_detector
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO

class Detector:
    def __init__(self, maybe_use_camera=True):
        self._root = "lab-1-perception-jyue86/"
        if maybe_use_camera:
            self.maybe_use_camera = maybe_use_camera
            self._yolov8 = YOLO(self._root + "models/yolov8.pt").cuda()
        else:
            # Add your initialization logic here
            self._models = {
                "pointpillars": ["models/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py", "models/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth"],
            }
            self._point_pillar = init_model(
                self._root + self._models["pointpillars"][0], self._root + self._models["pointpillars"][1]
            )

    def sensors(self):  # pylint: disable=no-self-use
        """
        Define the sensor suite required by the detector. The location is defined with respect to the actor center
        -- x axis is longitudinal (forward-backward)
        -- y axis is lateral (left and right)
        -- z axis is vertical
        Unit is in meters

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]

        """
        sensors = [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 1280, 'height': 720, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 1280, 'height': 720, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
                      'range': 50, 
                      'rotation_frequency': 20, 'channels': 64,
                      'upper_fov': 4, 'lower_fov': -20, 'points_per_second': 2304000,
                      'id': 'LIDAR'},

            {'type': 'sensor.other.gnss', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'id': 'GPS'}
        ]

        return sensors

    def detect(self, sensor_data, sensor_transforms):
        """
        Add your detection logic here
            Input: sensor_data, a dictionary containing all sensor data. Key: sensor id. Value: tuple of frame id and data. For example
                'Right' : (frame_id, numpy.ndarray)
                    The RGBA image, shape (H, W, 4)
                'Left' : (frame_id, numpy.ndarray)
                    The RGBA image, shape (H, W, 4)
                'LIDAR' : (frame_id, numpy.ndarray)
                    The lidar data, shape (N, 4)
            Output: a dictionary of detected objects in global coordinates
                det_boxes : numpy.ndarray
                    The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
                det_class : numpy.ndarray
                    The object class for each predicted bounding box, shape (N, 1) corresponding to the above bounding box. 
                    0 for vehicle, 1 for pedestrian, 2 for cyclist.
                det_score : numpy.ndarray
                    The confidence score for each predicted bounding box, shape (N, 1) corresponding to the above bounding box.
        """
        if self.maybe_use_camera:
            _, left_cam_data = sensor_data["Left"]
            _, right_cam_data = sensor_data["Right"]
            left_image = Image.fromarray(left_cam_data)
            right_image = Image.fromarray(right_cam_data)
            # plt.imshow(left_image)
            results = self._yolov8(right_image)
            det_data = results[0].boxes.data.detach().cpu().numpy()
            n_detections = det_data.shape[0]
            if n_detections == 0:
                return {}

            det_classes = det_data[:,-1]
            det_scores = det_data[:,-2]
            valid_indices = np.where((det_classes == 0) | (det_classes == 1) | (det_classes == 3))
            n_detections = len(valid_indices[0])
            # print("valid indices:", valid_indices)
            det_data = det_data[valid_indices]
            assert det_data.shape[0] == len(valid_indices[0])
            if n_detections == 0:
                return {}
            det_classes = det_classes[valid_indices]
            det_classes[det_classes == 3] = 2
            # print("Det data shape:", det_data.shape)
            det_scores = det_scores[valid_indices]
            # print("classes and scores shape:", det_classes.shape, det_scores.shape)

            bboxes = []
            cam_K = sensor_transforms["camera_K"]
            left_2_world = sensor_transforms["right2world"]
            # print(sensor_transforms["left2world"])
            for i in range(n_detections):
                bbox = det_data[i]
                x1, y1, x2, y2, _, _ = bbox
                bbox = np.array([
                    [x1, y1], 
                    [x1, y2], 
                    [x2, y2], 
                    [x2, y1]]) # 4 x 2
                # print("=====")
                # print("bbox:")
                # print(bbox)
                # print("=====")
                # plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2))
                
                # Unproject the bounding box
                bbox_homogeneous = np.concatenate([bbox, np.ones((4, 1))], axis=1) # 4 x 3
                assert bbox_homogeneous.shape == (4,3)
                # print("=====")
                # print("bbox homogeneous:")
                # print(bbox_homogeneous)
                # print("=====")
                bbox_unprojected = (np.linalg.inv(cam_K) @ bbox_homogeneous.T).T # 4 x 3
                assert bbox_unprojected.shape == (4,3)
                # print("=====")
                # print("bbox unprojected:")
                # print(bbox_unprojected)
                # print("=====")

                # Transform to world coordinates
                bbox_unprojected_homogeneous = np.concatenate([bbox_unprojected, np.ones((4, 1))], axis=1) # 4 x 4
                assert bbox_unprojected_homogeneous.shape == (4,4)
                bbox_world = (left_2_world @ bbox_unprojected_homogeneous.T).T
                assert bbox_world.shape == (4,4)
                bbox_world = bbox_world[:, :3] / bbox_world[:, -1][:, np.newaxis]
                bbox_world = bbox_world[:,1:]
                assert bbox_world.shape == (4,2)
                # print("=====")
                # print("bbox world:")
                # print(bbox_world)
                # print("=====")
                # print("BBox shape:", bbox_world.shape)
                bboxes.append(bbox_world)
                # return {}
            bboxes = np.stack(bboxes)
            # if n_detections > 0:
            #     plt.savefig("output.png")
            # plt.close()

            return {
                "det_boxes": bboxes,
                "det_class": det_classes,
                "det_score": det_scores
            }
        else:
            # Initialize the output
            det_bboxes = []
            # Left, Right, # GPS, Lidar
            _, lidar_data = sensor_data['LIDAR']
            result, _ = inference_detector(self._point_pillar, lidar_data)
            bboxes3d = result.pred_instances_3d.bboxes_3d.corners.detach().cpu().numpy()
            n_detections = bboxes3d.shape[0]
            if n_detections == 0:
                return {}

            for i in range(n_detections):
                bbox3d = bboxes3d[i]
                # det_boxes = np.array([[[0,0,0],[0,1,0],[1,1,0],[1,0,0],[0,0,1],[0,1,1],[1,1,1],[1,0,1]]])
                bbox3d = np.stack([
                    bbox3d[0], bbox3d[3], bbox3d[7], bbox3d[4], bbox3d[1], bbox3d[2], bbox3d[6], bbox3d[5]
                ])
                bbox3d = np.concatenate([bbox3d, np.ones((8, 1))], axis=1) # 8 x 4
                bbox3d = (sensor_transforms["lidar2world"] @ bbox3d.T).T # 8 x 4 
                bbox3d = bbox3d[:, :3] / bbox3d[:, -1][:, np.newaxis] # 8 x 3
                det_bboxes.append(bbox3d) 
            det_bboxes = np.stack(det_bboxes)
            det_classes = result.pred_instances_3d.labels_3d.detach().cpu().numpy().flatten()
            det_scores = result.pred_instances_3d.scores_3d.detach().cpu().numpy().flatten()

            return {
                "det_boxes": det_bboxes,
                "det_class": det_classes,
                "det_score": det_scores
            }