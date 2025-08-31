# Dataview_Streamlit

I perform data cleaning and EDA on Kaggle's "The World's Billionaires by Forbes" dataset, then train regression models to generate predictions. The results are presented in an interactive Streamlit app for intuitive analysis and visualization.

I also use Cursor to help me with coding. The project structure will evolve over time as development continues.

## Update for the things that I update. 

I have added a progress bar to the Streamlit app that will display during the model training process. I also have fixed the model selection issue so that now when you choose a different model, it will actually use that model instead of always using the same one. 

I have created a results folder that will save all the model training results with naming like "R1_ModelName.json" so that you can track all your previous model runs. The app now also shows training time for each model so you can compare which models are faster or slower.

I have removed the duplicate columns from the dataset display and fixed the config file path issues so that the app works properly in both the root directory and vir_env folder.