import pickle
import pandas as pd
import numpy as np
import streamlit as st

st.title('Diabetes Predictor')

pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))


pregnancies = st.number_input('Enter pregnancies', min_value=0)

glucsose = st.number_input('Enter Glucose amount',min_value=0)

Bp = st.number_input('Enter Blood pressure',min_value=0)

skinthick = st.number_input('Enter skinthickness',min_value=0)

insulin = st.number_input('Enter insulin ',min_value=0)

bmi = st.number_input("Enter bmi",min_value=0.0)

dpf = st.number_input('Enter DiabetesPedigreeFunction',min_value=0.00)

age = st.number_input("Enter Age",min_value=0,max_value=100)

input = np.array([[pregnancies,glucsose,Bp,skinthick,insulin,bmi,dpf,age]])

def prediction(input):
    a = pipe.predict(input)
    if a[0] == 1:
         st.error('Person has diabetes.')
    else:
        st.success("Person doesn't have diabetes.")


if st.button('Predict'):
    prediction(input)

