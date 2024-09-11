import numpy as np 
import pandas as pd
import streamlit as st

# for some basic operations
import numpy as np 
import pandas as pd 

# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import squarify
import time
import plotly.express as px
from cleanData import stringOutput

global numeric_columns
def encodeer(df):
	from sklearn.preprocessing import LabelEncoder
	encState=LabelEncoder()
	df['States']=encState.fit_transform(df['States'])
	return df 


def df_standardized(df):
	from sklearn import preprocessing
	df_standardized = preprocessing.scale(df)
	print(df_standardized)
	df_standardized = pd.DataFrame(df_standardized) 
	return df_standardized




def show_K_K_means(df):
	import streamlit as st
	from streamlit_yellowbrick import st_yellowbrick
	from yellowbrick.cluster import KElbowVisualizer
	
	model = KMeans()
	# k is range of number of clusters.
	visualizer = KElbowVisualizer(model, k=(1,11), timings=False)
	visualizer.fit(X)     
	st_yellowbrick(visualizer)
	kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)
	y_kmeans = kmeans.fit_predict(df_standardized)





def load_dataset(df):
	df=df[["States","Murder","Assault","UrbanPop","Rape","Year"]]
	df=df.rename({"UrbanPop":"Theft"},axis=1)
	df=df.dropna()
	numeric_columns=list(df.select_dtypes(['float','int']).columns)
	df=encodeer(df)
	df=df_standardized(df)
	return df

def Dataset_upload():

	upload_file=st.file_uploader('Upload Dataset',type='csv')
	if upload_file is not None:
		df=pd.read_csv(upload_file, index_col=0)
		st.success('Dataset Successfully uploaded')
		if st.button('Click to Train|Test dataset'):

			progress_bar = st.progress(0) 
			status_text = st.empty()
			df=df[["States","Murder","Assault","UrbanPop","Rape","Year"]]
			df=df.rename({"UrbanPop":"Theft"},axis=1)
			df=df.dropna()
			df=encodeer(df)
			df=df_standardized(df)
			status_text.text("Training dataset.......")
			for i in range(100):
				progress_bar.progress(i + 1)
				time.sleep(0.1)
			status_text.text('Done Training dataset!')
			if st.checkbox("Predict"):
				show_crime_predict()
			
			
		else:
			st.markdown('please upload a .CSV file')

	



def show_explore_page():
	

		
		
	st.title("CRIME RATE ANALYSIS AND VISUALISATION")
	
	df=pd.read_csv("NigerCrime.csv", index_col=0)
	df=df[["States","Murder","Assault","UrbanPop","Rape","Year"]]
	df=df.rename({"UrbanPop":"Theft"},axis=1)
	df=df.dropna()
	numeric_columns=list(df.select_dtypes(['float','int']).columns)
	
	

	st.sidebar.header("Crime Exploration")
	st.sidebar.markdown("### Visualising  Data")
	if st.sidebar.checkbox("Preview Dataset"):
		
		num=st.sidebar.slider("Select number of rows")
		if num!="":
			st.write(df.head(num)) 
	st.sidebar.subheader("Scatterplot settings")
	

	x_axis = st.sidebar.selectbox('X axis',options=numeric_columns)  
	y_axis = st.sidebar.selectbox('Y axis',options=numeric_columns)  
	cdict=[]
	cluster={x_axis:"red",
			y_axis:"blue"
			}
	

	# plot the value
	fig = px.scatter(df,
                x=x_axis,
                y=y_axis,
              
                title=f'Murder rate vs {x_axis}')

	st.plotly_chart(fig)
	

	











