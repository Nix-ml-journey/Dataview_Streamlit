import streamlit as st 
import pandas as pd 
import plotly.express as px 
from Data_loader import DataLoader
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def streamlit_design():
    st.title("Forbes 2000 Dataset")
    st.write(df.head())
    st.write(df.info())
    st.write(df.describe())
    st.write(df.columns)

    st.set_page_config(page_title="Forbes 2000 Dataset", layout="wide")

df = DataLoader('config.yml').load_data()

if __name__ == "__main__":
    data_loader = DataLoader('config.yml')
    df = data_loader.load_data 
    logging.info(f"Successfully loaded data")
    
