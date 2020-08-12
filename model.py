import numpy as np
from numpy import mean
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import random

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.linear_model import LogisticRegression 
from sklearn.datasets import make_classification
from sklearn import model_selection
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


data = pd.read_csv('all_data_selected_atc.level3.softmax.csv')
feature_names = list(data.columns.values)
feature_names.remove('asd_status')
X = data[feature_names]
Y = data['asd_status']


random.seed(8) 
# define pipeline
over = SMOTE(sampling_strategy=0.1, k_neighbors = 5)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)

# transform the dataset
X_smot, Y_smot = pipeline.fit_resample(X, Y)


log_model = LogisticRegression(solver = 'liblinear')
log_model = log_model.fit(X_smot, Y_smot)

model_columns = list(X_smot.columns)

pickle.dump(log_model, open('model.pkl','wb'))
pickle.dump(model_columns, open('model_columns.pkl','wb'))


