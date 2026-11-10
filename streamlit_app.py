import streamlit as st
import pandas as pd
from typing import Any, Dict, Union


st.title('Machine Learning App')

st.info("this app builds a machine learning model !")

with st.expander('Data'):
  st.write('**Row Data**')
  df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv")
  df
with st.expander('X/y'):
  st.write('**X**')
  X_row = df.drop('species', axis=1)
  X_row

  st.write('**y**')
  y_row = df.species
  y_row

with st.expander('Data Visualization'):
  st.scatter_chart(data=df, x='bill_length_mm',y='body_mass_g',color = 'species')

 

with st.sidebar:
  st.header("Input Features")
  island = st.selectbox(("island"),("Torgersen","Biscoe","Dream"))
  gender = st.selectbox(("gender"),("Male","Female"))
  bill_length_mm = st.slider(("Bill length (mm)"),32.1,59.6,43.9)
  bill_depth_mm = st.slider(("Bill depth (mm)"),13.1,21.5,17.2)
  flipper_length_mm = st.slider(("Flipper length (mm)"),172.0,231.0,201.0)
  body_mass_g = st.slider(("Body mass (mm)"),2700.0,6300.0,4207.0)


  #create DataFrame for the Input Features
  data = {'island':island,
          'bill_length_mm': bill_length_mm,
          'bill_depth_mm': bill_depth_mm,
          'flipper_length_mm': flipper_length_mm,
          'body_mass_g': body_mass_g,
          'gender': gender
           }
  input_df = pd.DataFrame(df, index = [0])
  input_penguins = pd.concat([input_df,X_row],axis = 0)

#encode X
encode = ['island','gender','sex']
df_penguins = pd.get_dummies(input_penguins, prefix = encode)
input_row = df_penguins[:1]

#encode y
target_mapper = {'Adelie' : 0,
                 'Chinstrap' : 1,
                 'Gentoo' : 2}


def target_encode(val) :
  return target_mapper(val)

y = y_row.apply(target_encode)
y
  
with st.expander("Input Features"):
  st.write('**Input Penguin**')
  input_df
  st.write('**Combined penguins data**')
  input_penguins
  st.write("**Input Encoded**")
  input_row











  
  









