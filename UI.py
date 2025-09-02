import streamlit as st 
import pandas as pd 
import plotly.express as px 
import plotly.graph_objects as go 
from Data_loader import DataLoader
import logging
import json
import yaml
import os
from Model import split_data_and_train_model, get_latest_result_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    try: 
        # Fix: Use the correct config path
        config_path = os.path.join(os.path.dirname(__file__), 'config.yml')
        data_loader = DataLoader(config_path)
        df = data_loader.load_data()

        df_cleaned = df.drop(columns=['Percentage_Change_Clean'])
        logging.info(f"Successfully loaded data and removed duplicated columns")
        return df_cleaned
    except Exception as e: 
        logging.error(f"Error loading data: {e}")
        return None 
    
def load_model_results():
    try:
        latest_file = get_latest_result_file()
        if latest_file and os.path.exists(latest_file):
            with open(latest_file, 'r') as file:
                results = json.load(file)
                results['result_file'] = latest_file
                return results
        else:
            return None

    except Exception as e:
        logging.error(f"Error loading model results: {e}")
        return None

def create_visualizations(df):
    # Age distribution
    fig_age = px.histogram(df, x='Age', nbins=30, title='Age Distribution of Billionaires')
    st.plotly_chart(fig_age, use_container_width=True)

    # Number of billionaires by country
    fig_country = px.histogram(df, x='Country/Territory', title='Number of Billionaires by Country')
    st.plotly_chart(fig_country, use_container_width=True)

    # Top 10 countries by net worth
    country_worth = df.groupby('Country/Territory')['Net Worth_numeric(Billions)'].sum().sort_values(ascending=False).head(10)
    fig_country_worth = px.bar(x=country_worth.index, y=country_worth.values, title='Top 10 Countries by Net Worth')
    st.plotly_chart(fig_country_worth, use_container_width=True)

    # Age vs Net Worth scatter plot
    fig_scatter = px.scatter(df, x='Age', y='Net Worth_numeric(Billions)', 
                            color='Country/Territory', size='Net Worth_numeric(Billions)', 
                            title='Age vs Net Worth')
    st.plotly_chart(fig_scatter, use_container_width=True)

def model_selection_interface():
    st.header("Model Selection for Prediction")

    # Fix: Use the correct config path
    config_path = os.path.join(os.path.dirname(__file__), 'config.yml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    available_models = config['Model_type'] 
    selected_model = st.selectbox("Select a model to use for prediction", available_models, index=0)

    st.info(f"Currently selected model is {selected_model}")

    if st.button("Train Selected Model"):
        with st.spinner("Training model... This may take a while..."):

            progress_bar = st.progress(0)
            status_text = st.empty()
        
            try:
                config['Selected_Models'] = selected_model
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)

                progress_bar.progress(0)
                status_text.text("Loading data...")

                progress_bar.progress(40)
                status_text.text("Training model...")

                model, X_test, y_test, y_pred, mse, r2, results, result_file = split_data_and_train_model()

                progress_bar.progress(100)
                status_text.text("Model training completed successfully!")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Squared Error", f"{mse:.4f}")
                with col2:
                    st.metric("R-squared", f"{r2:.4f}")

                st.subheader("Detailed Results")
                st.json(results)
                
                # Show where results were saved
                st.info(f"Results saved to: {result_file}")

            except Exception as e:
                st.error(f"Error training model: {e}")

            finally:
                progress_bar.empty()
                status_text.empty()

def display_dataset_info(df):
    st.header("Dataset Overview and Statistics Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Billionaires", len(df))
    with col2:
        st.metric("Total Countries", df['Country/Territory'].nunique())
    with col3:
        st.metric("Total Net Worth", f"{df['Net Worth_numeric(Billions)'].sum():.2f} Billion")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

    st.subheader("Dataset Statistics")
    st.write(df.describe())

def project_status_dashboard(df):
    st.header("Project Status Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Dataset Health", "Good", "No issues")
    with col2:
        missing_data = df.isnull().sum().sum()
        st.metric("Missing Values", missing_data, "0" if missing_data == 0 else f"+{missing_data}")
    with col3:
        results_count = len([f for f in os.listdir('results') if f.endswith('.json')]) if os.path.exists('results') else 0
        st.metric("Models Trained", results_count, "0" if results_count == 0 else f"+{results_count}")
    with col4:
        latest_results = load_model_results()
        status = "Ready" if latest_results else "No Model"
        st.metric("System Status", status)
    
    if missing_data > 0:
        st.warning("Dataset has missing values that will affect model performance")
    elif results_count == 0:
        st.info("Ready to train the first model! Go to 'Model Training' page")
    else:
        st.success("System is running optimally!")

def streamlit_design():
    st.set_page_config(page_title="Forbes Billionaires Dataset", layout="wide")  # Fixed typo
    st.title("Forbes Billionaires Dataset Analysis and Prediction")
    st.markdown("---")

    df = load_data()
    if df is None:
        st.error("Error loading data. Please check the data source and configuration.")
        return 

    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a page", ["Home", "Data Overview", "Model Training", "Visualizations", "Results"])  # Fixed missing comma
    
    if page == "Home":
        if df is not None:
            project_status_dashboard(df)
            st.markdown("---")
            
        st.header("Welcome to the Forbes Billionaires Dataset Analysis and Prediction")
        st.write("""
        This application analyzes the Forbes Billionaires dataset and provides:
        - **Data Exploration**: Understand the dataset structure and statistics
        - **Machine Learning**: Train various regression models to predict net worth
        - **Visualizations**: Interactive charts and graphs
        - **Results**: View model performance and predictions
        """)

        if df is not None:
            st.subheader("Quick Statistics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Billionaires", len(df))
            with col2:
                st.metric("Total Countries", df['Country/Territory'].nunique())
            with col3:
               st.metric("Average Age", f"{df['Age'].mean():.1f}")
            with col4:
                st.metric("Average Net Worth", f"{df['Net Worth_numeric(Billions)'].mean():.2f} Billion")
            
    elif page == "Data Overview":
        display_dataset_info(df)

    elif page == "Model Training":
        model_selection_interface()

    elif page == "Visualizations":
        st.header("Interactive Visualizations")
        create_visualizations(df)

    elif page == "Results":
        st.header("Model Results")
        results = load_model_results()
        if results:
            st.json(results)

            if 'result_file' in results:
                st.info(f"Results loaded from: {results['result_file']}")
        else:
            st.warning("No model results found. Please train a model first.")

def show_model_history():
    results_dir = 'results'
    if os.path.exists(results_dir):
        json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
        st.subheader("Previous Model Results")
        for file in sorted(json_files, reverse=True):
            st.write(f"• {file}") 

if __name__ == "__main__":
    streamlit_design()