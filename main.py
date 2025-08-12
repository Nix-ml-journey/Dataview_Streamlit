from Data_loader import DataLoader
import logging
import sys 
import streamlit as st 
from UI import streamlit_design
import pandas as pd

def main():
    try:
        df = DataLoader('config.yml').load_data()
        logging.info(f"Successfully loaded data")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        sys.exit(1)

    st.title("Forbes 2000 Dataset")
    st.write(df.head())
    st.write(df.info())
    st.write(df.columns)

if __name__ == "__main__":
    main()