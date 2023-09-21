#All the dependencies are here
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
import os
import random
from tqdm.notebook import tqdm
from glob import glob
from itertools import chain
from PIL import Image
from flask import Flask
from flask import request
from flask import render_template
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
from keras.losses import binary_crossentropy
from keras import backend as K

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

plt.style.use("ggplot")
matplotlib.use('agg')