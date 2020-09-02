# Core Packages
import streamlit as st

#EDA packages
import pandas as pandas
import numpy as numpy

#Utilities
import os 
import joblib

#Data Visualization 
import matplotlib.pyplot as pyplot
import matplotlib
matplotlib.use('Agg')

def main():
	""" Mortaility Prediction App"""
	st.title("Hepatitis Mortality Prediction App")

	menu = ["Home","Login","SignUp"]
	submenu = ["Plot","Prediction","Metrics"]

	choice = st.sidebar.selectbox("Menu",menu)
	if choice == "Home":
		st.subheader("Home")
		st.text("What is Hepatitis?")
	elif choice == "Login":
		username = st.sidebar.text_input("Username")
		password = st.sidebar.text_input("Password",type='password')
		if st.sidebar.checkbox("Login"):
			if password == "12345":
				st.success("welcome {}".format(username))
				
				activity = st.selectbox("Activity",submenu)
				if activity == "Plot":
					st.subheader("Data Visualization Plot")

				if activity == "Prediction":
					st.subheader("Predictive Analytics")
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
				pass

if __name__ == '__main__':
	main()
