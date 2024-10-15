import numpy as np
import pandas as pd
from pandas import plotting
# import SessionState
# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

# EDA Pkgs
import os
# Plotting Pkgs
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image,ImageFilter,ImageEnhance



# Core Pkgs
import streamlit as st 

# EDA Pkgs
import pandas as pd 
import codecs
from ydata_profiling import ProfileReport 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Components Pkgs
import streamlit.components.v1 as components
from streamlit_pandas_profiling import st_profile_report
from analysis import show_Analysis


# Custome Component Fxn
import sweetviz as sv 

# st.set_option('deprecation.showPyplotGlobalUse', False)
def st_display_sweetviz(report_html,width=1000,height=500):
	report_file = codecs.open(report_html,'r')
	page = report_file.read()
	components.html(page,width=width,height=height,scrolling=True)


def callback():
	st.session_state.button_clicked = True
	st.session_state.checked_box = True







def show_main():
		show_customer_Analysis()

		          

		












def Dataset_upload():
	# st.markdown('## Upload dataset') #Streamlit also accepts markdown
	data_file = st.file_uploader("Upload a CSV file", type="csv") #data uploader
	# st.sidebar.markdown('## Data Import') #Streamlit also accepts markdown
	# data_file = st.sidebar.file_uploader("Upload a CSV file", type="csv") #data uploader
	if data_file is not None:
		df = pd.read_csv(data_file)
		st.markdown('### Data Preview')
		st.dataframe(df)
		# st.warning("To get an overview of the dataset click the Data Info button  in the sidebar")
		return df




def explore_page():
	st.subheader("EXPLORATORY DATA ANALYSIS")
	data_file = st.sidebar.file_uploader("Upload CSV",type=['csv'])
	if data_file is not None:
		data=pd.read_csv(data_file)
		st.info("Click on DATA VISUALIZATION to Explore the dataset")

		if st.sidebar.button("DATA VISUALIZATION", on_click=callback) or st.session_state.button_clicked :
			if st.sidebar.checkbox("PIE CHART FOR GENDER DISTRIBUTION"):
				labels = ['Female', 'Male']
				size = data['Gender'].value_counts()
				colors = ['lightgreen', 'orange']
				explode = [0, 0.1]
				plt.rcParams['figure.figsize'] = (9, 9)
				fig=plt.figure(figsize= (9, 9))
				plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')
				plt.title('Gender', fontsize = 20)
				plt.axis('off')
				plt.legend()
				plt.show()
				st.pyplot(fig)
				st.subheader("Inference from the Distribution of gender from  Pie-charts")  
				st.write("It can be seen from the pie chart that distribution that the Females are in the lead with a share of 56% whereas the Males have a share of 44%, which implies that the female has higher population than the male")
			if st.sidebar.checkbox("Andrew Curve Plot with Matplotlib"):
				st.subheader("Andrew Curve to show distance between Gender Datapoint")
				fig=plt.figure(figsize= (20, 10))
				ax=plt.axes()
				# plt.rcParams['figure.figsize'] = (15, 10)
				plotting.andrews_curves(data.drop("CustomerID", axis=1), "Gender")
				plt.title('Andrew Curves for Gender', fontsize = 20)
				# plt.show()
				st.pyplot(fig)
				st.write("Andrews curves are able to preserve means, distance (up to a constant) and variances. Which means that Andrews curves that are represented by functions close together suggest that the corresponding data points will also be close together")



			if st.sidebar.checkbox("DISTRIBUTION FOR ANNUAL INCOME AND AGE"):
				st.subheader("Distribution of Annual income and Age for customer dataset")
				import warnings
				warnings.filterwarnings('ignore')
				fig=plt.figure(figsize= (20, 10))
				plt.subplot(1, 2, 1)
				sns.set(style = 'whitegrid')
				sns.distplot(data['Annual Income (k$)'])
				plt.title('Distribution of Annual Income', fontsize = 20)
				plt.xlabel('Range of Annual Income')
				plt.ylabel('Count') 
				plt.subplot(1, 2, 2)
				sns.set(style = 'whitegrid')
				sns.distplot(data['Age'], color = 'red')
				plt.title('Distribution of Age', fontsize = 20)
				plt.xlabel('Range of Age')
				plt.ylabel('Count')
				st.pyplot(fig)
				st.write("""we can infer one thing that There are few people who earn more than 100 US Dollars. Most of the people have an earning of around 50-75 US Dollars. Also, we can say that the least Income is around 20 US Dollars.
	""")
				st.subheader("Taking inferences about the Customers Age.")
				st.write("""The most regular customers for the Mall has age around 30-35 years of age. Whereas the the senior citizens age group is the least frequent visitor in the Mall. Youngsters are lesser in umber as compared to the Middle aged people.
	""")



			if st.sidebar.checkbox("DISTRIBUTION OF AGE ON COUNT PLOT"):
				fig=plt.figure(figsize= (20, 8))
				sns.countplot(data['Age'], palette = 'hsv')
				plt.title('Distribution of Age', fontsize = 24)
				st.pyplot(fig)
				st.write("From the graph, Ages from 27 to 39 are very frequent but there is no clear pattern,  the older age groups are lesser frequent. There are equal number of Visitors in the Mall for the Agee 18 and 67. People of Age 55, 56, 69, 64 are very less frequent in the Malls. People at Age 32 are the Most Frequent Visitors in the Mall.")
			if st.sidebar.checkbox("DISTRIBUTION OF ANNUAL INCOME ON COUNT PLOT"):	
				fig=plt.figure(figsize= (20, 8))
				sns.countplot(data['Annual Income (k$)'], palette = 'rainbow')
				plt.title('Distribution of Annual Income', fontsize = 20)
				st.write("This is also a chart to better explain the Distribution of Each Income level,There are customers in the mall with a very much comparable frequency with their Annual Income ranging from 15k US Dollars to 137K US Dollars. There are more Customers in the Mall who have their Annual Income as 54k US Dollars or 78k US Dollars.")
			if st.sidebar.checkbox("DISTRIBUTION OF SPENDING SCORE"):
				fig=plt.figure(figsize= (20, 8))
				sns.countplot(data['Spending Score (1-100)'], palette = 'copper')
				plt.title('Distribution of Spending Score', fontsize = 20)
				st.pyplot(fig)
				st.write("we may conclude that most of the Customers have their Spending Score in the range of 35-60. Interesting there are customers having I spending score also, and 99 Spending score also, Which shows that the mall caters to the variety of Customers with Varying needs and requirements available in the Mall.")
			if st.sidebar.checkbox("PAIR PLOT FOR THE DATASET"):
				sns.pairplot(data)
				plt.title('Pairplot for the Data', fontsize = 20)
				st.pyplot()


			if st.sidebar.checkbox("HEATMAP FOR THE DATASET"):
				fig=plt.figure(figsize= (15, 8))
				sns.heatmap(data.corr(), cmap = 'Wistia', annot = True)
				plt.title('Heatmap for the Data', fontsize = 20)
				st.pyplot()
				st.write("The Heatmap Graph for Shows the correlation between the different attributes of the  Dataset, This Heatmap reflects the most correlated features with Orange Color and least correlated features with yellow color.")







footer_temp = """

	 <!-- CSS  -->
	  <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
	  <link href="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css" type="text/css" rel="stylesheet" media="screen,projection"/>
	  <link href="static/css/style.css" type="text/css" rel="stylesheet" media="screen,projection"/>
	   <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.5.0/css/all.css" integrity="sha384-B4dIYHKNBt8Bc12p+WXckhzcICo0wtJAoU8YZTY5qE0Id1GSseTk6S+L3BlXeVIU" crossorigin="anonymous">


	 <footer class="page-footer grey darken-4">
	    <div class="container" id="aboutapp">
	      <div class="row">
	        <div class="col l6 s12">
	          <h5 class="white-text">About Customer Segmentation System</h5>
	          <p class="grey-text text-lighten-4">Using Streamlit,Python and Pandas Profile for Market customer segmentation.</p>


	        </div>
	      
	   <div class="col l3 s12">
	          <h5 class="white-text">Connect With Me</h5>
	          <ul>
	            <a href="https://facebook.com/Akinwande Alex " target="_blank" class="white-text">
	            <i class="fab fa-facebook fa-4x"></i>
	          </a>
	          <a href="https://gh.linkedin.com/in/Akinwande Alexander" target="_blank" class="white-text">
	            <i class="fab fa-linkedin fa-4x"></i>
	          </a>
	          <a href="https://www.youtube.com/channel/UC2wMHF4HBkTMGLsvZAIWzRg" target="_blank" class="white-text">
	            <i class="fab fa-youtube-square fa-4x"></i>
	          </a>
	           <a href="https://github.com/Akinwande/" target="_blank" class="white-text">
	            <i class="fab fa-github-square fa-4x"></i>
	          </a>
	          </ul>
	        </div>
	      </div>
	    </div>
	    <div class="footer-copyright">
	      <div class="container">
	      Made by <a class="white-text text-lighten-3" href="https://akinalex21@gmail.com">Fakorede Akinwande Alexander</a><br/>
	      <a class="white-text text-lighten-3" href="https://akinwandealex95@gmail.com">akinwandealex95@gmail.com</a>
	      </div>
	    </div>
	  </footer>

	"""