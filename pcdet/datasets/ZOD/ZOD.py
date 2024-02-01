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

class ZOD(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )

        self.frames = ZodFrames(dataset_root=root_path,
                    version="mini") # mini or full, TODO: Move this to config

    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        To support a custom dataset, implement this function to receive the predicted results from the model, and then
        transform the unified normative coordinate to your required coordinate, and optionally save them to disk.

        Args:
            batch_dict: dict of original data from the dataloader
            pred_dicts: dict of predicted results from the model
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path: if it is not None, save the results to this path
        Returns:

        """

    def __len__(self):
        self.frames.__len__()

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        # item = self.prepare_data(self.frames[index]) 
        return self.frames[index]

# TODO: Add some tests to see that the dataset wrapper is working
if __name__ == "__main__":
    import yaml
    from pathlib import Path
    from easydict import EasyDict
    dataset_cfg = EasyDict(yaml.load(open('CasA\tools\cfgs\dataset_configs\ZOD.yaml')))
    print(dataset_cfg)
    # dataset = ZOD_dataset()