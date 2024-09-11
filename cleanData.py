import streamlit as st 
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import imblearn


# Ignore warnings
import warnings
import sys
warnings.filterwarnings('ignore')

# Settings
pd.set_option('display.max_columns', None)

np.set_printoptions(threshold=sys.maxsize)


np.set_printoptions(precision=3)
sns.set(style="darkgrid")
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


def stringOutput(y_predict):
	clust_Val=[]
	for i in y_predict:
		if i==0:
			st="Low crime Rate"
			clust_Val.append(st)
			
		elif i==1:
			st="High crime Rate"
			clust_Val.append(st)
	return clust_Val[0]

def clean_Data(y_predict):
	clust_Val=[]
	for i in y_predict:
		if i==0:
			st="Low crime Rate"
			clust_Val.append(st)
			
		elif i==1:
			st="High crime Rate"
			clust_Val.append(st)
	return clust_Val[0]

def data_preparation(test_data):
	#print(data_train.head(4))
	#print(train.head(4))

	#print("Training data has {} rows & {} columns".format(train.shape[0],train.shape[1]))

	#train.describe()
	#print(test_data.head(4))

	#print("Testing data has {} rows & {} columns".format(test_data.shape[0],test_data.shape[1]))

	#Exploration  Analysis
	# Descriptive statistics
	#train.describe()
	#print(train['num_outbound_cmds'].value_counts())
	#print(test['num_outbound_cmds'].value_counts())

	#'num_outbound_cmds' is a redundant column so remove it from both train & test datasets
		#train.drop(['num_outbound_cmds'], axis=1, inplace=True)
	test_data.drop(['num_outbound_cmds'], axis=1, inplace=True)

	# Attack Class Distribution
		#train['class'].value_counts()

	# SCALING NUMERICAL ATTRIBUTES

	from sklearn.preprocessing import StandardScaler
	scaler = StandardScaler()

	# extract numerical attributes and scale it to have zero mean and unit variance  
	#cols = train.select_dtypes(include=['float64','int64']).columns
	cols = test_data.select_dtypes(include=['float64','int64']).columns

		#sc_train = scaler.fit_transform(train.select_dtypes(include=['float64','int64']))
	sc_test = scaler.fit_transform(test_data.select_dtypes(include=['float64','int64']))

	# turn the result back to a dataframe
		#sc_traindf = pd.DataFrame(sc_train, columns = cols)
	sc_testdf = pd.DataFrame(sc_test, columns = cols)

	# ENCODING CATEGORICAL ATTRIBUTES

	from sklearn.preprocessing import LabelEncoder
	encoder = LabelEncoder()

	# extract categorical attributes from both training and test sets 
		#cattrain = train.select_dtypes(include=['object']).copy()
	cattest=test_data.select_dtypes(include=['object']).copy()

	# encode the categorical attributes
		#traincat = cattrain.apply(encoder.fit_transform)
	testcat = cattest.apply(encoder.fit_transform)

	# separate target column from encoded data 
		#enctrain = traincat.drop(['class'], axis=1)
		#cat_Ytrain = traincat[['class']].copy(	)

		#train_x = pd.concat([sc_traindf,enctrain],axis=1)
		#train_y = train['class']
		#train_x.shape

	test_df = pd.concat([sc_testdf,testcat],axis=1)
	return test_df		

