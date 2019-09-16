# Disaster Response Pipeline Project


### About The Project:

The Disaster Response Pipeline is designed to be able to take in any set of text and recognize if it is related to disasters and then also categorize what topic is the text related to within the categories of the model. The major files include:
	
1. process_data.py
	- process data is used to complete a full ETL process on the dataset used to build our model. The data is processed by merging the 2 different datasets, creating the required category columns, and removing duplicates and non binary data from the categorical data. After cleaning the data it is stored into a sqlite database.

2. train_classifer.py
	- The classifier is trained here using a pipeline with a count vectorizer and tfidf transformer on the tokens first, then fitted on a MultiOutputClassifer using a RandomForestClassifer as its base estimator. Grid Search was used to find the optimal model, and the results were calculated with Sklearn's Classification report. 

3. run.py
	- Run.py contains information of the flask app that is used to show informative graphics of the data that was processed, while also show casing the model's classification ability on a string of the user's choice. 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://Localhost:3001/


