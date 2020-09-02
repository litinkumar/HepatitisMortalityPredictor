# Core Packages
import streamlit as st

#EDA packages
import pandas as pd
import numpy as np

#Utilities
import os 
import joblib
import hashlib

#Data Visualization 
import matplotlib.pyplot as pyplot
import matplotlib
matplotlib.use('Agg')

# ML Interpretation
import lime
import lime.lime_tabular

from manage_db import *

# Password 
def generate_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()


def verify_hashes(password,hashed_text):
	if generate_hashes(password) == hashed_text:
		return hashed_text
	return False

def get_value(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return value 

def get_key(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return key

def get_fvalue(val):
	feature_dict = {"No":1,"Yes":2}
	for key,value in feature_dict.items():
		if val == key:
			return value 

# Load ML Models
def load_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model



feature_names_best = ['age', 'sex', 'spiders', 'ascites','varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime','histology']

gender_dict = {"Male":1,"Female":2}
feature_dict = {"No":1,"Yes":2}

def main():
	""" Mortaility Prediction App"""
	st.title("Hepatitis Mortality Prediction App")

	menu = ["Home","Login","SignUp"]
	submenu = ["Exploratory Data Analysis","Prediction"]

	choice = st.sidebar.selectbox("Menu",menu)
	if choice == "Home":
		st.subheader("Home")
		st.text("What is Hepatitis?")
	elif choice == "Login":

		username = st.sidebar.text_input("Username")
		password = st.sidebar.text_input("Password",type='password')
		if st.sidebar.checkbox("Login"):
			create_usertable()
			hashed_pswd = generate_hashes(password)
			result = login_user(username,verify_hashes(password,hashed_pswd))
			if result:
				st.success("welcome {}".format(username))
				
				activity = st.selectbox("Activity",submenu)
				if activity == "Exploratory Data Analysis":
					st.subheader("Dataset")
					df = pd.read_csv("data/clean_hepatitis_dataset.csv")
					st.dataframe(df)

					st.subheader("Target Variable Distribution")
					df['class'].value_counts().plot(kind='bar')
					st.pyplot()

					st.subheader("Age Distribution")
					freq_df = pd.read_csv("data/freq_df_hepatitis_dataset.csv")
					pyplot.bar(freq_df['age'],freq_df['count'])
					pyplot.ylabel('Counts')
					st.pyplot()


				if st.checkbox("Univariate Analysis"):
					all_columns = df.columns.to_list()
					feat_choices = st.multiselect("Choose a Feature",all_columns)
					new_df = df[feat_choices]
					st.area_chart(new_df)

				if activity == "Prediction":
					st.subheader("Predictive Analytics")

					age = st.number_input("Age",7,80)
					sex = st.radio("Sex",tuple(gender_dict.keys()))
					spiders = st.radio("Presence of Spider Naeve",tuple(feature_dict.keys()))
					ascites = st.selectbox("Ascities",tuple(feature_dict.keys()))
					varices = st.selectbox("Presence of Varices",tuple(feature_dict.keys()))
					bilirubin = st.number_input("bilirubin Content",0.0,8.0)
					alk_phosphate = st.number_input("Alkaline Phosphate Content",0.0,296.0)
					sgot = st.number_input("Sgot",0.0,648.0)
					albumin = st.number_input("Albumin",0.0,6.4)
					protime = st.number_input("Prothrombin Time",0.0,100.0)
					histology = st.selectbox("Histology",tuple(feature_dict.keys()))
					malaise = st.selectbox("Malaise",tuple(feature_dict.keys()))
					liver_firm = st.selectbox("Liver Firm",tuple(feature_dict.keys()))


					feature_list = [age,get_fvalue(spiders),get_fvalue(ascites),get_fvalue(varices),bilirubin,alk_phosphate,sgot,albumin,int(protime),get_fvalue(histology),get_fvalue(malaise),get_fvalue(liver_firm)]
					st.write(len(feature_list))
					st.write(feature_list)
					pretty_result = {"age":age,"sex":sex,"spiders":spiders,"ascites":ascites,"varices":varices,"bilirubin":bilirubin,"alk_phosphate":alk_phosphate,"sgot":sgot,"albumin":albumin,"protime":protime,"histolog":histology}
					st.json(pretty_result)
					single_sample = np.array(feature_list).reshape(1,-1)

					# ML
					model_choice = st.selectbox("Select Model",["LR","KNN","DecisionTree"])
					if st.button("Predict"):
						if model_choice == "KNN":
							loaded_model = load_model("models/KNN_hepB_model.pkl")
							prediction = loaded_model.predict(single_sample)
							pred_prob = loaded_model.predict_proba(single_sample)
						elif model_choice == "DecisionTree":
							loaded_model = load_model("models/DT_hepB_model.pkl")
							prediction = loaded_model.predict(single_sample)
							pred_prob = loaded_model.predict_proba(single_sample)
						else:
							loaded_model = load_model("models/logreg_hepB_model.pkl")
							prediction = loaded_model.predict(single_sample)
							pred_prob = loaded_model.predict_proba(single_sample)

						# st.write(prediction)
						# prediction_label = {"Die":1,"Live":2}
						# final_result = get_key(prediction,prediction_label)
						if prediction == 1:
							st.warning("Patient Dies")
							pred_probability_score = {"Die":pred_prob[0][0]*100,"Live":pred_prob[0][1]*100}
							st.subheader("Prediction Probability Score using {}".format(model_choice))
							st.json(pred_probability_score)
							st.subheader("Prescriptive Analytics")
							st.markdown(prescriptive_message_temp,unsafe_allow_html=True)
							
						else:
							st.success("Patient Lives")
							pred_probability_score = {"Die":pred_prob[0][0]*100,"Live":pred_prob[0][1]*100}
							st.subheader("Prediction Probability Score using {}".format(model_choice))
							st.json(pred_probability_score)
							
			else:
				st.warning("Incorrect Username/Password")


	elif choice == "SignUp":
		new_username = st.text_input("User name")
		new_password = st.text_input("Password", type='password')

		confirm_password = st.text_input("Confirm Password",type='password')
		if new_password == confirm_password:
			st.success("Password Confirmed")
		else:
			st.warning("Passwords not the same")

		if st.button("Submit"):
			create_usertable()
			hashed_new_password = generate_hashes(new_password)
			add_userdata(new_username,hashed_new_password)
			st.success("You have successfully created a new account")
			st.info("Login to Get Started")

if __name__ == '__main__':
	main()
