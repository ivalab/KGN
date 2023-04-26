from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from .sample.multi_pose import MultiPoseDataset
from .sample.grasp_pose import GraspPoseDataset
try:
  from .sample.vae_grasp_pose import VAEGraspPoseDataset 
except:
  print("VAE GraspNet is not ready")
  VAEGraspPoseDataset = None

from .dataset.ps_grasp import PSGrasp
from .dataset.l2g_datasets import YCB8_test


dataset_factory = {
  'ps_grasp': PSGrasp,
  'YCB8_test': YCB8_test
}

_sample_factory = {
  'grasp_pose': GraspPoseDataset,
  'vae_graspnet': VAEGraspPoseDataset, 
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  if _sample_factory[task] is None:
    if task == "vae_graspnet":
      raise ImportError("The VAE Graspnet dataset is not ready. Please prepare it first.")
  return Dataset
  
