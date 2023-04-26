from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .grasp_pose import GraspPoseDetector

detector_factory = {
  "grasp_pose": GraspPoseDetector
}
