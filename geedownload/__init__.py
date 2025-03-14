"""
init script for geedownload imports everything from geeutils and tiffutils

Joel Nicolow, Coastal Research Collaborative, March 2025
"""

# this way makes it so all the functins from both scripts are just imported dirrectly from geedownload
from .geeutils import *  # imports everything from geeutils.py under geedownload module
from .tiffutils import *  # imports everything from tiffutils.py under geedownload module

# or could do this to keep scripts as modules
# import geedownload.geeutils
# import geedownload.tiffutils