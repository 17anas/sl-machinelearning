import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier



st.title('Machine Learning App') 

st.info("this app builds a machine learning model !")

with st.expander('Data'):
  st.write('**Row Data**')
  df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv")
  df
with st.expander('X/y'):
  st.write('**X**')
  X_raw = df.drop('species', axis=1)
  X_raw

  st.write('**y**')
  y_raw = df.species
  y_raw

with st.expander('Data Visualization'):
  st.scatter_chart(data=df, x='bill_length_mm',y='body_mass_g',color = 'species')

 
#Input Features
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
  input_penguins = pd.concat([input_df,X_raw],axis = 0)

  #Data preparation
  #encode X
  encode = ['island','gender','sex']
  df_penguins = pd.get_dummies(input_penguins, prefix = encode)

  X = df_penguins[1:]
  input_row = df_penguins[:1]

  #encode y
  target_mapper = {'Adelie' : 0,
                 'Chinstrap' : 1,
                 'Gentoo' : 2}


  def target_encode(val) :
    return target_mapper[val]

  y = y_raw.apply(target_encode)
  y
  y_raw
  


with st.expander("Data preparation"):
  st.write("**Encoded X (Input penguins)**")
  input_row
  st.write("**Encoded y** ")
  y


# model training and inference
# train the ML model
clf = RandomForestClassifier()
clf.fit(X,y)

# apply model to make predictions
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ["Adelie","Chinstrap","Gentoo"]



# Display predicted species

st.subheader("**Predicted Species**")
df_prediction_proba
penguins_species = np.array(["Adelie","Chinstrap","Gentoo"])
st.success(str(penguins_species[prediction][0]))





  
  









