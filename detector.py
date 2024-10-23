import numpy as np
from mmdet3d.apis import init_model, inference_detector

class Detector:
    def __init__(self):
        # Add your initialization logic here
        self._root = "lab-1-perception-jyue86/"
        self._edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]
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
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
                      'range': 50, 
                      'rotation_frequency': 20, 'channels': 64,
                      'upper_fov': 4, 'lower_fov': -20, 'points_per_second': 2304000,
                      'id': 'LIDAR'},

            {'type': 'sensor.other.gnss', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'id': 'GPS'}
        ]

        return sensors

    def detect(self, sensor_data):
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
        # Initialize the output
        det_bboxes = []

        # Left, Right, # GPS, Lidar
        _, lidar_data = sensor_data['LIDAR']
        result, _ = inference_detector(self._point_pillar, lidar_data)
        bboxes3d = result.pred_instances_3d.bboxes_3d.tensor.detach().cpu().numpy()
        n_detections = bboxes3d.shape[0]

        for i in range(n_detections):
            det_bboxes.append(self._construct_3d_bbox(bboxes3d[i])) 
        if len(det_bboxes) != 0:
            det_bboxes = np.stack(det_bboxes)
        else:
            det_bboxes = np.zeros((0, 8, 3))
        det_classes = result.pred_instances_3d.labels_3d.detach().cpu().numpy().reshape((-1, 1))
        det_scores = result.pred_instances_3d.scores_3d.detach().cpu().numpy().reshape((-1, 1))

        print(det_bboxes.shape)
        print(det_classes.shape)
        print(det_scores.shape)

        return {
            "det_boxes": det_bboxes,
            "det_class": det_classes,
            "det_score": det_scores
        }

    
    def _construct_3d_bbox(self, bbox_data):
        """
        Construct 3D bounding box from the given data
        :param bbox_data: list of 3D bounding box data
        :return: list of 3D bounding box in the format of 8 corners
        """
        tlx, tly, tlz, x_size, y_size, z_size, yaw = bbox_data
        brx, bry, brz = tlx + x_size, tly + y_size, tlz + z_size

        # 3D bounding box corners
        corners = np.array([
            [tlx, tly, tlz], [tlx, bry, tlz], [brx, bry, tlz], [brx, tly, tlz],
            [tlx, tly, brz], [tlx, bry, brz], [brx, bry, brz], [brx, tly, brz]
        ])

        return corners