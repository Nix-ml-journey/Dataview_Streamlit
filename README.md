# Dataview_Streamlit

I perform data cleaning and EDA on Kaggle's "The World's Billionaires by Forbes" dataset, then train regression models to generate predictions. The results are presented in an interactive Streamlit app for intuitive analysis and visualization.

I also use Cursor to help me with coding. The project structure will evolve over time as development continues.

## Update for the things that I made

I have updated the config.yml with several models that we can choose from, and I have set the model scaler and the metrics that will be used for evaluation during training and testing. I have created a new Python code called Model.py so that when we run it, it will use the model that we choose (we set the model in the config), and after that we will get a new result file named results.json. It's not that good yet - I have only tested with Linear Regression so far.