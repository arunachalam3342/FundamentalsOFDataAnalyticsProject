import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import base64
@st.cache(allow_output_mutation=True)
def preprocess():
    pass

data=pd.read_csv("Housing_Data.csv")
X=data.drop('medianHouseValue',axis=1)
y=data['medianHouseValue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
sc_x=StandardScaler()
sc_y=StandardScaler()   
X_train=sc_x.fit_transform(X_train)
y_train=np.array(y_train).reshape(y_train.shape[0],1)
y_train=sc_y.fit_transform(y_train)


#streamlit section

app_mode=st.sidebar.selectbox('Select Page',['Welcome','Home','Prediction'])
if app_mode=='Welcome':
    page_bg_img='''
    <style>
    .stApp{
    background-image: url('https://wallpapers.com/images/hd/real-estate-white-graphics-design-qr9394ynnogt96or.jpg');
    background-size:cover;
    }
    </style>
    '''
    st.markdown(page_bg_img,unsafe_allow_html=True)
    st.title("Welcome to Real Estate Housing Price Prediction Web App")
    st.subheader("Home Page : Information about website and dataset")
    st.subheader("Prediction Page: Predict the real estate housing price with user input feature values")
elif app_mode=='Home':
    st.snow()
    st.title("Real Estate Hosing Price Prediction ")
    st.header("Want to know the land prices for a housing plot move on to Prediction Page")
    st.image('real_estate_image.jpg')
    st.markdown('Used Dataset Sample:')
    st.write(data.head())
    st.markdown('Dataset statistical information is :')
    st.write(data.describe())
elif app_mode=='Prediction':
    st.title('Welcome to prediction page')
    st.image('img.jpg')
    st.header('Fill all necessary information in order to get the real estate housing price estimated amount')
    pred=st.button("Predict")
    st.sidebar.header("Information about the housing land")
    latitude=st.sidebar.number_input("Latitude of the Location")
    longitude=st.sidebar.number_input("Longitude of the Location")
    age=st.sidebar.number_input("Median Age group of the people in Housing Area",10,100)
    room=st.sidebar.number_input("Total rooms in the Housing Area")
    bedroom=st.sidebar.number_input("Total Bedroom in the Housing Area")
    population=st.sidebar.number_input("Population strength in Housing Area")
    house=st.sidebar.number_input("Total Number of Houses in Housing Area")
    income=st.sidebar.number_input("Median Income of the people in Housing Area($)")
    val=[latitude,longitude,age,room,bedroom,population,house,income]
    val=list(map(lambda x:float(x),val))
    val=np.array(val)
    val=val.reshape(1,8)
    val=pd.DataFrame(val,columns=data.columns[:8])
    val_new=sc_x.transform(val)
    if pred:
        st.balloons()
        file=open("price.gif","rb")
        contents=file.read()
        data_url=base64.b64encode(contents).decode("utf-8")
        file.close()
        loaded_model=pickle.load(open('real_estate_model.sav','rb'))
        prediction=loaded_model.predict(val_new)
        st.success("The Estimated Price for Housing Area is " + str(sc_y.inverse_transform([prediction])[0][0]) + "$")
        st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="price gif">',unsafe_allow_html=True,)
        
    
    
    
    