import streamlit as st 
import logging
from UI import streamlit_design

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        streamlit_design()
    except Exception as e:
        st.error(f"An Error Occured for the UI: {e}")
        logging.error(f"Streamlit Error: {e} ")


if __name__ == "__main__":
    main()