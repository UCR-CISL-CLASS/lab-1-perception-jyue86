import numpy as np

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def get_image_point(loc, K, w2c):
        point = np.array([loc.x, loc.y, loc.z, 1])
        point_camera = np.dot(w2c, point)

        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        point_img = np.dot(K, point_camera)
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]