## NLP-Machine-Learining-Pipelines

### Project Description

The project is about classifying messages that are sent during disaster into 36 different categories. The aim is to ensure the message are forwarded to the relevant disaster relief agent for quick action.

A multi-label classification model is used to predict the category each message belongs to. In addition, a web app is developed that displays the different category counts and with an option of inputting a message and displaying the category the message belongs to.


### Installation
Must runing with Python 3 with libraries of numpy, pandas, sqlalchemy, re, NLTK, pickle, Sklearn, plotly and flask libraries.

### File Descriptions
- App folder contains the templates folder and "run.py" for the web application
- Data folder contains  "DisasterResponse.db", "disaster_categories.csv", "disaster_messages.csv" and "process_data.py" for data cleaning and transfering.
- Models folder contains "classifier.pkl" and "train_classifier.py" for the Machine Learning model.


### Instructions
Run the following commands in the project's root directory to set up your database and model.

- To run ETL pipeline that cleans data and stores in database `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

- To run ML pipeline that trains classifier and saves `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

- Run the following command in the app's directory to run your web app. `python run.py`

Go to http://0.0.0.0:3000/

