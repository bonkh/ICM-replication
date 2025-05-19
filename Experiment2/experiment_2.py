import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
import pandas as pd
from sklearn.linear_model import LassoCV
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time

obs_data = pd.read_csv('observational_data.csv')
obs_data = obs_data.drop(columns=["Unnamed: 0"])

int_data = pd.read_csv('interventional_data.csv')
int_data = int_data.drop(columns=["Unnamed: 0"])

int_pos_data = pd.read_csv('interventional_position_data.csv')
edges = pd.read_csv("gene_network_edge.csv")
nodes = pd.read_csv('node.csv')

with open('top_10_predictors.jsonz', 'r') as f:
    data = json.load(f)
