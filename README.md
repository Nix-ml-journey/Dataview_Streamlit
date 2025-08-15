# Dataview_Streamlit

I perform data cleaning and EDA on Kaggle's "The World's Billionaires by Forbes" dataset, then train regression models to generate predictions. The results are presented in an interactive Streamlit app for intuitive analysis and visualization.

I also use Cursor to help me with coding. The project structure will evolve over time as development continues.

## Update for the things that I made

I have updated the code so that Streamlit now functions properly when running "streamlit run main.py". However, there are still some improvements needed: the results page in the Streamlit UI is only reading the old results.json file from when I ran the XGBoost model, which needs to be fixed. Also, the dataset I am using has duplicate columns that need to be removed from the view.