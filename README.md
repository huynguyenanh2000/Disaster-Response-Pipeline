# **Disaster Response Pipeline Project**


## **Table of contents**

- [Environment Setup](#environment-setup)
- [Project Descriptions](#project-descriptions)
- [File Structure](#file-structure)
- [Usage](#usage)


## **Environment Setup**

**Environment**
- OS: Windows 10

- Interpreter: Visual Studio Code

- Python version: Python 3.7

**Libraries**
- Install all packages using requirements.txt file using `pip install -r requirements.txt` command.

**Link to GitHub Repository**

`https://github.com/huynguyenanh2000/Disaster-Response-Pipeline.git`


## **Project Descriptions**

This project is a part of Udacity Data Scientist Nanodegree program. It helps us to categorize the message of a disaster event so that we can react in a timely manner.

## **File Structure**

~~~~~~~
disaster_response_pipeline
    |-- app
        |-- templates
                |-- go.html
                |-- master.html
        |-- run.py
    |-- data
        |-- DisasterResponseETL.db
        |-- disaster_message.csv
        |-- disaster_categories.csv
        |-- process_data.py
    |-- models
        |-- DisasterResponseModel.pkl
        |-- train_classifier.py
    |-- Preparation
        |-- ETL Pipeline Preparation.ipynb
        |-- ML Pipeline Preparation.ipynb
    |-- README
    |-- requirements.txt
~~~~~~~


## **Usage**

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponseETL.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponseETL.db models/DisasterResponseModel.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
