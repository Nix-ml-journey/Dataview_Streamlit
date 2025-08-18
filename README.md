# Dataview_Streamlit

I perform data cleaning and EDA on Kaggle's "The World's Billionaires by Forbes" dataset, then train regression models to generate predictions. The results are presented in an interactive Streamlit app for intuitive analysis and visualization.

I also use Cursor to help me with coding. The project structure will evolve over time as development continues.

## Update for the things that I made

I have removed the duplicated column in the dataset while displaying in the Streamlit app, and I have changed the method of reading the dataset so that now the selected model is working properly. I also added a folder called "results" so that no matter which model is used, it will create a JSON output named "R1_{MODEL_NAME}.json".