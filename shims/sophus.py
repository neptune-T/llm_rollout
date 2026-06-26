"""Compatibility module for HomeRobot's legacy `import sophus as sp`.

The A6000 HomeRobot env currently has `sophuspy` installed, whose modern wheel
exports the module name `sophuspy` instead of `sophus`. HomeRobot only needs
the SE2/SE3/SO2/SO3 API exposed by `sophuspy`, so re-export it here and keep
the third-party HomeRobot source tree unchanged.
"""

from sophuspy import *  # noqa: F401,F403
