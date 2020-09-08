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
	menu = ["Login","Sign up"]
	submenu = ["Exploratory Data Analysis","Prediction"]

	st.title("Hepatitis-B Analysis & Prediction")
	choice = st.selectbox("",menu)

	if choice == "Login":
		username = st.text_input("Username")
		password = st.text_input("Password", type='password')
		create_usertable()
		hashed_pswd = generate_hashes(password)
		result = login_user(username,verify_hashes(password,hashed_pswd))
		if st.checkbox("Login"):
			if 	result and (username!="" or password!=""):
				st.success("You've successfully signed in")

				st.header("Exploratory Data Analysis")

				st.subheader("Dataset")
				df = pd.read_csv("data/clean_hepatitis_dataset.csv")
				st.dataframe(df)

				st.subheader("Target Variable Distribution")
				df2=df
				df2['class'] = df['class'].replace(1, 'Patient Dies')
				df2['class'] = df['class'].replace(2, 'Patient Lives')
				df2['class'].value_counts().plot(kind='pie')
				pyplot.ylabel('')
				st.pyplot()

				st.subheader("Age Distribution")
				freq_df = pd.read_csv("data/freq_df_hepatitis_dataset.csv")
				pyplot.bar(freq_df['age'],freq_df['count'])
				pyplot.ylabel('Counts')
				st.pyplot()


				st.header("Predictive Analytics")

				age = st.number_input("Age",7,80)
				sex = st.radio("Sex",tuple(gender_dict.keys()))
				spiders = st.radio("Presence of Spider Naeve",tuple(feature_dict.keys()))
				ascites = st.radio("Ascities",tuple(feature_dict.keys()))
				varices = st.radio("Presence of Varices",tuple(feature_dict.keys()))
				bilirubin = st.slider("Bilirubin Content",0.0,8.0)
				alk_phosphate = st.slider("Alkaline Phosphate Content",0.0,296.0)
				sgot = st.slider("Sgot",0.0,648.0)
				albumin = st.slider("Albumin",0.0,6.4)
				protime = st.slider("Prothrombin Time",0.0,100.0)
				histology = st.radio("Histology",tuple(feature_dict.keys()))
				malaise = st.radio("Malaise",tuple(feature_dict.keys()))
				liver_firm = st.radio("Liver Firm",tuple(feature_dict.keys()))


				feature_list = [age,get_fvalue(spiders),get_fvalue(ascites),get_fvalue(varices),bilirubin,alk_phosphate,sgot,albumin,int(protime),get_fvalue(histology),get_fvalue(malaise),get_fvalue(liver_firm)]
#				st.write(len(feature_list))
#				st.write(feature_list)
#				pretty_result = {"age":age,"sex":sex,"spiders":spiders,"ascites":ascites,"varices":varices,"bilirubin":bilirubin,"alk_phosphate":alk_phosphate,"sgot":sgot,"albumin":albumin,"protime":protime,"histolog":histology}
#				st.json(pretty_result)
				single_sample = np.array(feature_list).reshape(1,-1)

				# ML
				model_choice = st.selectbox("Select Model",["LR","KNN","DecisionTree"])
				if st.button("Predict"):
					if model_choice == "KNN":
						loaded_model = load_model("models/KNN_hepB_Model.pkl")
						prediction = loaded_model.predict(single_sample)
						pred_prob = loaded_model.predict_proba(single_sample)
					elif model_choice == "DecisionTree":
						loaded_model = load_model("models/DT_hepB_Model.pkl")
						prediction = loaded_model.predict(single_sample)
						pred_prob = loaded_model.predict_proba(single_sample)
					else:
						loaded_model = load_model("models/LogReg_hepB_Model.pkl")
						prediction = loaded_model.predict(single_sample)
						pred_prob = loaded_model.predict_proba(single_sample)

					# st.write(prediction)
					# prediction_label = {"Die":1,"Live":2}
					# final_result = get_key(prediction,prediction_label)
					if prediction == 1:
						st.error("Patient Dies")
#						pred_probability_score = {"Die":pred_prob[0][0]*100,"Live":pred_prob[0][1]*100}
#						st.subheader("Prediction Probability Score using {}".format(model_choice))
#						st.json(pred_probability_score)
#						st.markdown(prescriptive_message_temp,unsafe_allow_html=True)
						
					else:
						st.success("Patient Lives")
#						pred_probability_score = {"Die":pred_prob[0][0]*100,"Live":pred_prob[0][1]*100}
#						st.subheader("Prediction Probability Score using {}".format(model_choice))
#						st.json(pred_probability_score)

			else :
				st.error("Incorrect Username/Password. Please Sign Up!")

	elif choice=="Sign up":
		new_username = st.text_input("Choose a Username")
		new_password = st.text_input("Enter a Password", type='password')

		confirm_password = st.text_input("Confirm Password",type='password')
		if new_password != confirm_password:
			st.warning("Passwords not the same")

		if st.button("Submit"):
			create_usertable()
			hashed_new_password = generate_hashes(new_password)
			add_userdata(new_username,hashed_new_password)
			st.success("You have successfully created a new account")
			st.info("Login to Get Started")

if __name__ == '__main__':
	main()
