# Disaster Response Pipeline Project

The project was done as part of [Udacity Data Science Nanodegree program](https://www.udacity.com/course/data-scientist-nanodegree--nd025). The objective is be able to accurately predict if a given message (that could eventually come from different sources) is related to a disaster category or not. To achieve this, a webpage has been created where any user can input a message and in the backend a machine learning model will do the prediction. In the same webpage, the user can visualize some descriptive graphics about the dataset that was used for training the model. 

The learning objective is to be able to apply an ETL pipeline as was as a ML pipeline combined with GridSearch to automatically look for the best parameters to fit the training data. 

## About the dataset

The dataset was provided by Figure Eight (now [Appen](https://appen.com/)). It consist of two CSV files: 

- **disaster_messages.csv**: Containing the messages 
- **disaster_categories.csv**: Containing the categories of those messages (0 or 1 for each category) 

It originally consists of 32 categories, however as exposed in the discussion section, some categories were merged together. 

## How to run this project

The project consists on three main scripts:

- **process_data.py**: This script reads the csv files and, performs some transformation on the data and saves it into a database. Here is an example on how to run the script: 

    ```bash
    python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db'
    ```
- **train_classifier.py**: This script reads the database, performs cleaning operations on the dataset, trains a machine learning pipeline (using GridSearch) and stores the model into a pickle file. Here is an example on how to run the script: 

    ```bash
    python train_classifier.py ../data/DisasterMessages.db classifier.pkl
    ``` 

- **run.py**: This script runs a Flask application allowing us to visualize a website. It will read the database and create some figures to be loaded to the website, and will also load the model in order to perform predictions based on the website input. From the python/Anaconda terminal you have to run:

    ```bash
    python run.py
    ``` 
  
    From your local browser just go to: **127.0.0.1:3001**. Be sure to run the two previous scripts before running website. 

## Discussion 

### Issues with the dataset 

One of the main issues encountered while training the model with the dataset, is that the data is **not balanced** i.e. the number of negative labels (0) in the dataset greatly outnumbered the number of positive labels (1) for almost all categories. As we can see in the following graphic (also included in the website):

![Image1](https://github.com/jagutier75/Disaster_Response_Pipeline/blob/master/images/Pos_neg_per_category.PNG)

Some of the categories showed less than 2% of positive labels in the entire dataset. This caused some issues when evaluating the model as not having enough data to calculate recall or precision. To try to minimize this, it was decided to merge those categories with similar categories (this is done in the cleaning phase of the dataset): 

    `col_merge_pairs = {'offer': 'aid_related',
                           'fire': 'other_aid',
                           'shops': 'other_infrastructure',
                           'hospitals': 'other_infrastructure',
                           'tools': 'aid_related',
                           'missing_people': 'search_and_rescue',
                           'clothing': 'aid_related',
                           'aid_centers': 'aid_related',
                           'security': 'other_aid',
                           }`

### Evaluation metrics

As we have very few positive labels per category. It was decided to use *recall-micro* as the scoring function while using GridSearch. What this means is that we are trying to rise the recall, thus decreasing the number of false negatives made in our predictions (we want to catch as much real situations as possible!)
