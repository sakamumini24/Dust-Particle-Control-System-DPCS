import numpy as np
import pandas as pd
from pandas import plotting
# import SessionState

# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')


import plotly.figure_factory as ff


from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans


import streamlit as st
# EDA Pkgs
import os

from PIL import Image,ImageFilter,ImageEnhance
# from predict_page import predictor
# from EDAappnew import show_main
# from EDAappnew import explore_page
from analysis import show_Analysis
from analysis import show_predict
# for some basic operations
import numpy as np 
import pandas as pd 
# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
import time
import plotly.express as px
# from cleanData import stringOutput
# Components Pkgs
import streamlit.components.v1 as components
from streamlit_pandas_profiling import st_profile_report
import sqlite3
import joblib

# Set background image using custom CSS
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({image_url});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )







conn=sqlite3.connect("data.db")
c=conn.cursor()



header  = st.container()
inp = st.container()
pred = st.container()

BASE_DIR = os.getcwd()


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











def create_table():
	c.execute('CREATE TABLE IF NOT EXISTS usertable(username TEXT, password TEXT)')

def add_userdata(username,password):
	c.execute('INSERT INTO  usertable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM usertable WHERE username=? AND password=?',(username, password))
	data=c.fetchall()
	return data

def view_all_users():
	c.execute('SELECT * FROM usertable')
	data=c.fetchall()
	return data




def Dataset_upload():
	# st.markdown('## Upload dataset') #Streamlit also accepts markdown
	st.markdown('## Upload CSV or Excel File')
	data_file = st.file_uploader("Upload a CSV or Excel file here", type=["csv", "xlsx"]) #data uploader
	if data_file is not None:
		# Check file type and read accordingly
		if data_file.name.endswith('.csv'):
			df = pd.read_csv(data_file)
		elif data_file.name.endswith('.xlsx'):
			df = pd.read_excel(data_file, engine='openpyxl')
		
		# Display data
		st.write("Data Preview:")
		st.dataframe(df.head())

		# Perform additional analysis or operations here
		st.write("Data Summary:")
		st.write(df.describe())

		return df


# menu=["Home" ,"Login","Signup","About","Logout"]
def callback():
	st.session_state.button_clicked = True
	st.session_state.checked_box = True

def home(home):
	st.subheader("Home")
	st.title('Dust Particle Control System')
	st.warning("Check the sidebar menu to login into the system")


# Function to predict using the model
def predict_pm25(input_data, historical_data, window=30):
    file_name=BASE_DIR+'/models/voting_reg.pkl'
    model = joblib.load(file_name)
    
    # Use historical data to calculate rolling mean dynamically
    historical_data = historical_data.append(input_data, ignore_index=True)
    # historical_data["pm25"]
    rolling_mean = calculate_rolling_mean(historical_data, window)
    input_data['pm25_rolling_mean'] = rolling_mean
    
    prediction = model.predict(input_data)
    return prediction




# Function to calculate the rolling mean for the prediction
def calculate_rolling_mean(df, window=30):
    df['pm25_rolling_mean'] = df['pm25_log'].rolling(window=window).mean().iloc[-1]
        # # Example: Rolling mean for PM2.5
    df['pm25_rolling_mean'] = df['pm25_log'].rolling(window=30).mean()
    return df['pm25_rolling_mean']



def main():

    # Sidebar menu options
    menu = ["Home", "Login", "Signup", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    # App title
    st.title('Dust Particle Control System')

    # Home Page
    if choice == "Home":
        st.subheader("Home")
        image_url =BASE_DIR+"/dust_part.jpg"
        st.image(image_url)
        st.warning("Check the sidebar menu to login into the system")

    # Login Page
    elif choice == "Login":
        if "button_clicked" not in st.session_state:
            st.session_state.button_clicked = False
        if "checked_box" not in st.session_state:
            st.session_state.checked_box = False

        st.error("Please input your login details in the sidebar")
        st.sidebar.subheader("Login Section")
        
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type='password')

        if st.sidebar.button("Login", on_click=callback) or st.session_state.button_clicked:
            create_table()
            result = login_user(username, password)

            if result:
                st.success(f"Logged In as {username}")
                st.title('Dust Particle Control System')

                task = st.sidebar.selectbox("Task", ["Homepage", "Data Modeling","Predict PM2.5", "Profiles"])

                if task == "Homepage":
                    st.write("Welcome to Dust Particle Control System")
                    st.warning("""> This system makes use of **Machine learning Models**.. 
                    To get started, go to the Data Modeling section of the task tab and upload a CSV file 
                    containing dust particle and meteorological features for modeling the control of Dust particles""")

                elif task == "Data Modeling":
                    show_Analysis()

                elif task == "Predict PM2.5":
                    show_predict()


                elif task == "Profiles":
                    st.subheader("User Profiles")
                    u_data = view_all_users()
                    clean_db = pd.DataFrame(u_data, columns=["Username", "Password"])
                    st.dataframe(clean_db)

            else:
                st.warning("Incorrect Username/Password")

    # Signup Page
    elif choice == "Signup":
        st.subheader("Create New Account")
        new_user = st.text_input('Username')
        new_password = st.text_input("Password", type="password")

        if st.button("Signup"):
            create_table()
            add_userdata(new_user, new_password)
            st.success("You have successfully created a valid account")
            st.info("Go to the Login Menu to log in")

    # About Page
    elif choice == "About":
        st.subheader("About App")
        components.html(footer_temp, height=500)

if __name__=='__main__':
	main()