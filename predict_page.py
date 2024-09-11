import streamlit as st 
import numpy as np 
import pickle
import pandas as pd
 
from explore_page import Dataset_upload


def load_model():

	with open("savecrime.pkl",'rb') as file:
		data=pickle.load(file)
	return data

data=load_model()
predictor=data['model']
encState=data['encState']



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


def show_crime_predict():
	st.title("CRIME RATE ANALYSIS AND PREDICTION SYSTEM ")
	st.subheader('Predicting crime rate of an  Area within a country')
	States=("Abia","Adamawa","Akwa Ibom","Anambra","Bauchi","Bayelsa","Benue","Borno","Cross River","Delta","Ebonyi","Edo","Ekiti","Enugu","Gombe","Imo","Jigawa","Kaduna",
	"Kano","Kastina","Kebbi","Kogi","Kwara","Lagos","Nasarawa","Niger",
	"Ogun","Ondo","Osun","Oyo","Plateau","Rivers","Sokoto","Taraba","Yobe","Zamfara","FCT Abuja")

 
	State=st.selectbox('State',States)
	Murder=st.number_input("Murder")
	Assault=st.number_input("Assault")
	Theft=st.number_input("Theft")
	Rape=st.number_input("Rape")
	Year=st.slider("Year",2010,2050,(2018))
	ok=st.button("Predict ")
	if ok:
		x=np.array([[State,Murder,Assault,Theft,Rape,Year]])
		x[:,0]=encState.fit_transform(x[:,0])
		x=x.astype(float)
		predict=stringOutput(predictor.predict(x))
		st.subheader(f"Total Crime Rate is: {predict}")

	





	
















