import logging
import os
from glob import iglob

import cv2
import numpy as np


class Kitti:
    def __init__(self, directory):
        root_dir = directory['root']
        self.calib_dir = os.path.join(root_dir, directory['calib'])
        self.image_dir = os.path.join(root_dir, directory['image'])
        self.label_dir = os.path.join(root_dir, directory['label'])
        index_path = os.path.join(root_dir, directory['index'])
        image_filenames = list()
        if os.path.isfile(index_path):
            with open(index_path) as index_file:
                for index in index_file.read().strip().splitlines():
                    # image extensions may be different
                    image_path = next(iglob(os.path.join(self.image_dir, f'{index}.*')), None)
                    if image_path and os.path.isfile(image_path):
                        image_filenames.append(os.path.basename(image_path))
                    else:
                        logging.warning(f'The image does not exist. index=f{index}')
        else:
            # if index file is not specified, image names will be indices
            image_filenames = sorted(os.listdir(self.image_dir))
        # "self.image_filenames" contains images that calib and label both exist
        # "self.image_filenames" is the index of the dataset
        self.image_filenames = list()
        for image_filename in image_filenames:
            index = image_filename.split('.', maxsplit=1)[0]
            is_image_filename_valid = True
            if not os.path.isfile(os.path.join(self.calib_dir, f'{index}.txt')):
                is_image_filename_valid = False
                logging.warning(f'The calib file does not exist. index=f{index}')
            if not os.path.isfile(os.path.join(self.label_dir, f'{index}.txt')):
                is_image_filename_valid = False
                logging.warning(f'The label file does not exist. index=f{index}')
            if is_image_filename_valid:
                self.image_filenames.append(image_filename)
        return

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, item):
        image_filename = self.image_filenames[item]
        index = image_filename.split('.', maxsplit=1)[0]
        image = cv2.imread(os.path.join(self.image_dir, image_filename))
        matrices = self.read_calib(os.path.join(self.calib_dir, f'{index}.txt'))
        proj_matrix = matrices['P2']
        obj_infos = self.read_label(os.path.join(self.label_dir, f'{index}.txt'))
        return index, image, proj_matrix, obj_infos

    @staticmethod
    def read_calib(calib_path):
        with open(calib_path, 'r') as calib_file:
            matrices = dict()
            for line in calib_file.read().strip().splitlines():
                matrix_id, matrix = line.split(maxsplit=1)
                matrix_id = matrix_id[:-1]
                matrix = matrix.split()
                matrix = np.array(matrix, dtype=np.float32)
                matrix = matrix.reshape((3, -1))
                matrices[matrix_id] = matrix
        return matrices

    @staticmethod
    def read_label(label_path):
        obj_infos = list()
        with open(label_path, 'r') as label_file:
            for line in label_file.read().strip().splitlines():
                obj_info = dict()
                line = line.split()
                obj_info['type'] = line[0]
                obj_info['truncation'] = float(line[1])
                obj_info['occlusion'] = int(float(line[2]))
                obj_info['alpha'] = float(line[3])
                obj_info['bbox'] = [float(n) for n in line[4:8]]
                obj_info['dimension'] = [float(n) for n in line[8:11]]
                obj_info['location'] = [float(n) for n in line[11:14]]
                obj_info['rotation_y'] = float(line[14])
                if len(line) > 15:
                    obj_info['score'] = float(line[15])
                obj_infos.append(obj_info)
        return obj_infos
