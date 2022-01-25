<<<<<<< HEAD
# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
=======
## NLP-Machine-Learining-Pipelines

File Structure

app
- | - template
- | |- master.html # main page of web app
- | |- go.html # classification result page of web app
- |- run.py # Flask file that runs app

data
|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py
|- CleanDatabase.db # database to save clean data to

models
|- train_classifier.py
|- cv_AdaBoostr.pkl # saved model

README.md

Instructions:
Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/cv_AdaBoost.pkl

Run the following command in the app's directory to run your web app. python run.py

Go to http://0.0.0.0:3001/

https://view6914b2f4-3001.udacity-student-workspaces.com
>>>>>>> e90f50a8b585c50c03d8ca779f354b6be27fec3d