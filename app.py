import numpy as np
import pandas as pd
import pickle 
import streamlit as st

pickle_a=open("linear.pkl","rb")
regressor=pickle.load(pickle_a) # our model

def predict_chance(GREScore,TOEFLScore,UniversityRating,CGPA):
    prediction=regressor.predict([[GREScore,TOEFLScore,UniversityRating,CGPA]]) #predictions using our model
    return prediction 


def main():
    st.title("Admission prediction APP using ML") #simple title for the app
    html_temp="""
        <div>
        <h2>Admission Prediction ML app</h2>
        </div>
        """
    st.markdown(html_temp,unsafe_allow_html=True) #a simple html 
    GREScore=st.text_input("GRE Score + Range 1 - 340")
    TOEFLScore=st.text_input("TOEFL Score + Range 1 - 120")
    UniversityRating=st.text_input("University Rating - Range 1 - 5 ")
    CGPA=st.text_input("CGPA Range + 1 - 10") #giving inputs as used in building the model
    result=""
    if st.button("Predict"):
        result=predict_chance(GREScore, TOEFLScore, UniversityRating, CGPA) #result will be displayed if button is pressed
    st.success("The chance of admission in that university is{}".format(result))
        
if __name__=='__main__':
    main()
