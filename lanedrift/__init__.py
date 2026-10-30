"""Lane-drift monitor: classical lane detection -> tracking -> lateral offset -> departure events."""
from .config import Config
from .pipeline import LaneMonitor

__all__ = ["Config", "LaneMonitor"]
__version__ = "0.1.0"
