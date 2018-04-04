import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
%matplotlib inline
from helper import *
from vizdoom import *

from random import choice
from time import sleep
from time import time

# Adapted from https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2
# See also: https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb
class AC_Network():
	def __init__(self,n_states, n_actions, scope, trainer):
		with tf.variable_scope(scope):
