import cv2
import numpy as np
import os
from tqdm import tqdm
import yaml

from kitti import Kitti


def get_rotational_matrix(angle):
    s, c = np.sin(angle), np.cos(angle)
    r = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return r


def compute_box_3d(obj_info, proj_matrix):
    # compute rotational matrix around yaw axis
    r = get_rotational_matrix(obj_info['rotation_y'])
    # 3d bounding box dimensions
    h, w, l = obj_info['dimension']
    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners_3d = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    # rotate and translate 3d bounding box
    corners_3d = np.dot(r, corners_3d)
    corners_3d += np.array(obj_info['location'], dtype=np.float32).reshape(3, 1)
    # mirror the points behind the camera (z < 0)
    corners_3d[2, :] = np.abs(corners_3d[2, :])
    # project the 3D bounding box into the image plane
    corners_3d = corners_3d.transpose()
    corners_2d = project_to_image(corners_3d, proj_matrix)
    return corners_2d


def project_to_image(pts_3d, proj_matrix):
    # project in image
    pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
    pts_2d = np.dot(proj_matrix, pts_3d_homo.transpose()).transpose()
    # scale projected points
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
    return pts_2d


def draw_box_3d(image, corners, color, thickness, orient_mask_weight):
    # 3D bounding box faces
    #   1 -------- 0
    #  /|         /|
    # 2 -------- 3 .
    # | |        | |
    # . 5 -------- 4
    # |/         |/
    # 6 -------- 7
    face_idx = [[0, 1, 5, 4],  # front face
                [1, 2, 6, 5],  # left face
                [2, 3, 7, 6],  # back face
                [3, 0, 4, 7]]  # right face
    if corners is not None:
        corners = corners.astype(np.int32)
        for i, f in enumerate(face_idx):
            for j in (0, 1, 2, 3):
                cv2.line(image,
                         (corners[f[j], 0], corners[f[j], 1]),
                         (corners[f[(j + 1) % 4], 0], corners[f[(j + 1) % 4], 1]),
                         color,
                         thickness,
                         lineType=cv2.LINE_AA)
            # orientation
            if i == 0:
                mask = np.zeros(image.shape, dtype=np.uint8)
                cv2.fillConvexPoly(mask,
                                   np.array([(corners[f[0], 0], corners[f[0], 1]),
                                             (corners[f[1], 0], corners[f[1], 1]),
                                             (corners[f[2], 0], corners[f[2], 1]),
                                             (corners[f[3], 0], corners[f[3], 1])]),
                                   color)
                image = cv2.addWeighted(image, 1, mask, orient_mask_weight, 0)
    return image


def main():
    root_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(root_dir)
    with open(r'config.yml', 'r') as cfg_file:
        cfg = yaml.load(cfg_file.read(), Loader=yaml.Loader)
    directory = cfg['directory']
    threshold = cfg['threshold']
    color_table = cfg['color_table']
    thick_3d = cfg['thick_3d']
    thick_2d = cfg['thick_2d']
    orient_mask_weight = cfg['orient_mask_weight']
    output_dir = directory['output']
    if os.path.lexists(output_dir):
        if not os.path.isdir(output_dir):
            print("Output directory is not valid.")
            return
    else:
        os.mkdir(output_dir)
    kitti = Kitti(directory)
    for index, image, proj_matrix, obj_infos in tqdm(kitti):
        # draw objects from near to far
        for obj_info in sorted(obj_infos, key=lambda x: x['location'][2], reverse=True):
            if 'score' in obj_info and obj_info['score'] < threshold:
                continue
            obj_type = obj_info['type']
            color = color_table.get(obj_type, None)
            if color:
                # draw 3D bounding box
                color_3d = color.get('3d', None)
                if color_3d:
                    image = draw_box_3d(
                        image, compute_box_3d(obj_info, proj_matrix), color_3d, thick_3d, orient_mask_weight)
                # draw 2D bounding box
                color_2d = color.get('2d', None)
                if color_2d:
                    bbox = [int(n) for n in obj_info['bbox']]
                    image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color_2d, thick_2d)
        output_path = os.path.join(output_dir, f'{index}.png')
        cv2.imwrite(output_path, image)
    return


if __name__ == '__main__':
    main()
