
import pandas as pd
import matplotlib.pyplot as plt

#Scikit Learn Imports
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

df = pd.read_csv('Data/Volve/volve_wells.csv', 
                usecols=['WELL', 'DEPTH', 'RHOB', 'GR', 'NPHI', 'PEF', 'DT'])

