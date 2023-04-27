from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from .sample.multi_pose import MultiPoseDataset
from .sample.grasp_pose import GraspPoseDataset

from .dataset.ps_grasp import PSGrasp


dataset_factory = {
  'ps_grasp': PSGrasp,
}

_sample_factory = {
  'grasp_pose': GraspPoseDataset,
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  if _sample_factory[task] is None:
    pass
  return Dataset
  
