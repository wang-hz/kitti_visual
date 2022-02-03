import argparse
import cv2
from glob import iglob
import numpy as np
import os
from tqdm import tqdm


def read_calibration(calib_path, cam_id):
    with open(calib_path, 'r') as calib_file:
        p = np.array(calib_file.readlines()[cam_id].split()[1:], dtype=np.float32).reshape((3, 4))
    return p


def read_labels(label_path):
    labels = list()
    with open(label_path, 'r') as label_file:
        for line in label_file.read().strip().splitlines():
            label = dict()
            line = line.split()

            # extract label, truncation, occlusion
            label['type'] = line[0]
            label['truncation'] = float(line[1])
            label['occlusion'] = int(float(line[2]))
            label['alpha'] = float(line[3])

            # extract 2D bounding box in 0-based coordinates
            label['x1'] = float(line[4])
            label['y1'] = float(line[5])
            label['x2'] = float(line[6])
            label['y2'] = float(line[7])

            # extract 3D bounding box information
            label['h'] = float(line[8])
            label['w'] = float(line[9])
            label['l'] = float(line[10])
            label['t'] = [float(s) for s in line[11:14]]
            label['ry'] = float(line[14])

            if len(line) == 16:
                label['score'] = float(line[15])
            labels.append(label)
    return labels


def compute_box_3d(label, p):
    # compute rotational matrix around yaw axis
    s, c = np.sin(label['ry']), np.cos(label['ry'])
    r = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    # 3d bounding box dimensions
    h, w, l = label['h'], label['w'], label['l']

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(r, np.array([x_corners, y_corners, z_corners], dtype=np.float32))
    corners_3d += np.array(label['t'], dtype=np.float32).reshape(3, 1)

    # mirror the points behind the camera (z < 0)
    for i in range(len(corners_3d[2, :])):
        if corners_3d[2, i] < 0:
            corners_3d[2, i] = -corners_3d[2, i]

    # project the 3D bounding box into the image plane
    corners_2d = project_to_image(corners_3d.transpose(), p)
    return corners_2d


def project_to_image(pts_3d, p):
    # project in image
    pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
    pts_2d = np.dot(p, pts_3d_homo.transpose()).transpose()

    # scale projected points
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
    return pts_2d


def draw_box_3d(image, corners, color):
    #   1 -------- 0
    #  /|         /|
    # 2 -------- 3 .
    # | |        | |
    # . 5 -------- 4
    # |/         |/
    # 6 -------- 7

    # index for 3D bounding box faces
    face_idx = [[0, 1, 5, 4],  # front face
                [1, 2, 6, 5],  # left face
                [2, 3, 7, 6],  # back face
                [3, 0, 4, 7]]  # right face

    if corners is not None:
        corners = corners.astype(np.int32)
        for i, f in enumerate(face_idx):
            for j in range(4):
                cv2.line(image,
                         (corners[f[j], 0], corners[f[j], 1]),
                         (corners[f[(j + 1) % 4], 0], corners[f[(j + 1) % 4], 1]),
                         color,
                         1,
                         lineType=cv2.LINE_AA)
            if i == 0:
                # orientation
                mask = np.zeros(image.shape, dtype=np.uint8)
                cv2.fillConvexPoly(mask,
                                   np.array([(corners[f[0], 0], corners[f[0], 1]),
                                             (corners[f[1], 0], corners[f[1], 1]),
                                             (corners[f[2], 0], corners[f[2], 1]),
                                             (corners[f[3], 0], corners[f[3], 1])]),
                                   color)
                image = cv2.addWeighted(image, 1, mask, 0.2, 0)
    return image


def visual_one_file(calib_path, image_path, label_path, output_path, threshold):
    p = read_calibration(calib_path, 2)
    image = cv2.imread(image_path)
    labels = read_labels(label_path)
    for label in sorted(labels, key=lambda k: k['t'][2], reverse=True):
        if 'score' in label and label['score'] < threshold:
            continue
        if label['type'] == 'Car':
            color = (0, 0, 255)
        elif label['type'] == 'Pedestrian':
            color = (0, 255, 0)
        elif label['type'] == 'Cyclist':
            color = (255, 0, 0)
        else:
            color = (0, 255, 255)
        image = draw_box_3d(image, compute_box_3d(label, p), color)
        # draw 2d box
        image = cv2.rectangle(image,
                              (int(label['x1']), int(label['y1'])),
                              (int(label['x2']), int(label['y2'])),
                              (255, 255, 255),
                              1)
    cv2.imwrite(output_path, image)
    return


def visual_whole_folder(calib_dir, image_dir, label_dir, index_path, output_dir, threshold):
    if index_path:
        indices = index_path.read().strip().splitlines()
        for index in tqdm(indices):
            calib_path = os.path.join(calib_dir, f'{index}.txt')
            image_path = next(iglob(os.path.join(image_dir, f'{index}.*')))
            label_path = os.path.join(label_dir, f'{index}.txt')
            output_path = os.path.join(output_dir, f'{index}.png')
            visual_one_file(calib_path, image_path, label_path, output_path, threshold)
    else:
        for image_filename in tqdm(os.listdir(image_dir)):
            index = image_filename.split('.')[0]
            calib_path = os.path.join(calib_dir, f'{index}.txt')
            image_path = os.path.join(image_dir, image_filename)
            label_path = os.path.join(label_dir, f'{index}.txt')
            output_path = os.path.join(output_dir, f'{index}.png')
            visual_one_file(calib_path, image_path, label_path, output_path, threshold)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', '-R', type=str, metavar='DATASET_DIRECTORY')
    parser.add_argument('--calib', '-c', type=str, metavar='CALIB_FILES_DIRECTORY')
    parser.add_argument('--image', '-i', type=str, metavar='IMAGES_DIRECTORY')
    parser.add_argument('--label', '-l', type=str, metavar='LABELS_DIRECTORY')
    parser.add_argument('--index', '-I', type=str, metavar='INDEX_FILE_PATH',
                        help='All images will be processed if index file is not given')
    parser.add_argument('--output', '-o', type=str, metavar='OUTPUT_DIRECTORY')
    parser.add_argument('--threshold', '-t', type=float, default=0)
    args = parser.parse_args()

    calib_dir = None
    image_dir = None
    label_dir = None
    output_dir = None

    if args.root:
        calib_dir = os.path.join(args.root, 'calib')
        image_dir = os.path.join(args.root, 'image_2')
        label_dir = os.path.join(args.root, 'label_2')
        output_dir = os.path.join(args.root, 'output')

    if args.calib:
        calib_dir = args.calib
    elif not calib_dir:
        print("Calib files directory is not specified.")
        return

    if args.image:
        image_dir = args.image
    elif not image_dir:
        print("Images directory is not specified.")
        return

    if args.label:
        label_dir = args.label
    elif not label_dir:
        print("Labels directory is not specified.")
        return

    if os.path.lexists(output_dir):
        if not os.path.isdir(output_dir):
            print("Output directory is not a folder.")
            return
    else:
        os.mkdir(output_dir)

    index_path = args.index if args.index else None

    visual_whole_folder(calib_dir, image_dir, label_dir, index_path, output_dir, args.threshold)
    return


if __name__ == '__main__':
    main()
