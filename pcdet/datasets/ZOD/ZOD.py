import os
import pickle
import copy
import numpy as np
import torch
import multiprocessing
import tqdm
from pathlib import Path

import sys
sys.path.append("../../../")
# from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
# from pcdet.utils import box_utils, common_utils
from pcdet.datasets.dataset import DatasetTemplate

from zod import ZodFrames
import zod.constants as constants
from zod.constants import Lidar, Anonymization, AnnotationProject
from zod.data_classes import LidarData
from zod.anno.object import OBJECT_CLASSES, OBJECT_SUBCLASSES
# from quaternion import as_euler_angles


class ZOD(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=dataset_cfg.ROOT_PATH,
            logger=logger
        )

        self.frames = ZodFrames(dataset_root=dataset_cfg.DATA_PATH,
                    version=dataset_cfg.VERSION)
        if training:
            self.data_ids = list(self.frames._train_ids)
        else:
            self.data_ids = list(self.frames._val_ids)
        
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        To support a custom dataset, implement this function to receive the predicted results from the model, and then
        transform the unified normative coordinate to your required coordinate, and optionally save them to disk.
        N is "num_samples"

        Args:
            batch_dict: dict of original data from the dataloader
            pred_dicts: dict of predicted results from the model
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path: if it is not None, save the results to this path
        
        Returns:
            Returns (waymo case):
            {
                name: (N) Should be an array of class names
                score: (N) Array of prediction scores
                boxes_lidar: (N,7) Array of bouding boxes
                frame_id: (N) From the input arg batch_dict
                metadata (N) From the input arg batch_dict
            }

        """

    def __len__(self):
        return self.data_ids.__len__()

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        frame = self.frames[self.data_ids[index]]
        lidar = frame.get_lidar()[0] #Get only the core frame lidar
        
        normalized_intensity = lidar.intensity / 255 # In ZOD, intensity is a measure of the reflection magnitude ranging from 0-255
        unc_points = np.column_stack((lidar.points, normalized_intensity))
    
        annotations = frame.get_annotation(AnnotationProject.OBJECT_DETECTION) #List[ObjectAnnotation]
        gt_names = []
        gt_boxes = []
        for annotation in annotations:
            if annotation.box3d is None:
                continue
            gt_names.append(annotation.subclass)
            gt_boxes.append(zod_coordinate_system_to_uniform_coordinate_system(annotation.box3d))

        # input_dict.update({
        #     'gt_names': annos['name'],
        #     'gt_boxes': gt_boxes_lidar,
        #     'num_points_in_gt': num_points_in_gt
        # })

        input_dict = {
            'points'  : unc_points, # In unified normative coordinate system
            'frame_id': frame.metadata.frame_id,
            'gt_names': np.array(gt_names),
            'gt_boxes': np.array(gt_boxes)
        }
        # data_dict = self.prepare_data(data_dict=input_dict)
        # Add metadata? 
        return input_dict

def zod_coordinate_system_to_uniform_coordinate_system(zod_box3d):
    # Desired box format: 
    # format: [xc yc zc dx dy dz heading_angle category_name]
    # print(zod_box3d)
    # print("done")
    [xc,yc,zc] = zod_box3d.center
    [length, width, height] = zod_box3d.size
    rotation = zod_box3d.orientation.yaw_pitch_roll[0]
    uniformed_box = [xc, yc, zc, length, width, height, rotation]
    return uniformed_box


# TODO: Add some tests to see that the dataset wrapper is working
if __name__ == "__main__":
    import yaml
    import open3d
    from yaml import Loader
    from pathlib import Path
    from easydict import EasyDict
    from sklearn import preprocessing
    from tools.visual_utils import open3d_vis_utils as V
    # print(dataset_cfg)
    
    dataset_cfg = EasyDict(yaml.load(open(os.path.join('tools', 'cfgs', 'dataset_configs', 'ZOD.yaml')),Loader=Loader))
    dataset = ZOD(dataset_cfg=dataset_cfg, class_names=OBJECT_SUBCLASSES)

    test_frame = dataset[0]
    le = preprocessing.LabelEncoder()
    le.fit(test_frame['gt_names'])
    test_frame['categorical_label'] = le.transform(test_frame['gt_names'])
    print(test_frame['gt_names'])
    print(test_frame['categorical_label'])
    print(len(test_frame['gt_names']))
    print(len(test_frame['categorical_label']))
    scores = [1] * len(test_frame['gt_names'])
    V.draw_scenes(points=test_frame['points'],
                  ref_boxes=test_frame['gt_boxes'],
                  ref_labels=test_frame['categorical_label'],
                  ref_scores=scores) # TODO: I want to add a legend
