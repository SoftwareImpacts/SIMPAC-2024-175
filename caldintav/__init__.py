"""
Caldintav is a program that is developed by GCM in Python language in order to calculate the
dynamic responses of the bridges under the railway traffic loadings. The program offers:
    1. Can be used for analyzing both types of bridges:  simply-supported and continuos bridges
    2. Determine the time history of displacement and acceleration at the mid-span of bridge
    3. Determine the envelope of the maximum dynamic responses for a range of train velocities
    4. Can be used to perform a parametric calculation for various bridges and trains.
"""

from .functions import *

from .caldintav_designer import Ui_MainWindow

