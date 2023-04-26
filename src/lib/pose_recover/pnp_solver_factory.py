from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .cvPnPs import CVEPnPSolver, CVIPPESolver, CVP3PSolver 
from .planarPnP import PlanerPnPSolver 

PnPSolverFactory = {
  'cvEPnP': CVEPnPSolver, 
  'cvP3P': CVP3PSolver,
  'cvIPPE': CVIPPESolver,
  'planar': PlanerPnPSolver,
}

