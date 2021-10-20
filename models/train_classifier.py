import sys
import pandas as pd
import numpy as np
import re
import pickle
import nltk
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# NLTK package download
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def load_data(database_filepath):
    """
    Reads mySQL database and loads the information into a DataFrame

    :param database_filepath: Path to the database
    :return:
    - X - df containing the features data
    - Y - df containing the class/category data
    - categories_names - Names of the target classes
    """
    # Load data from database
    file_path = 'sqlite:///' + database_filepath
    engine = create_engine(file_path)
    df = pd.read_sql_table('Messages', con=engine)
    X = df['message']
    target = [col for col in list(df.columns) if col not in ['id', 'message', 'original', 'genre']]
    Y = df[target]
    categories_names = Y.columns
    return X, Y, categories_names


def tokenize(text):
    """
    Takes the text input and performs the following actions:
    - Remove all punctuation and numbers
    - It applies "stemmizing"
     It removes stopwords

    :param text: Input textx
    :return: clean_tokens: List of clean words extracted from text
    """
    # Removing punctuation and numbers
    text = re.sub(r"[^a-zA-Z]", " ", text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Removing stop words
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    # Initiate lemmatizer
    # lemmatizer = WordNetLemmatizer()
    stemmizer = PorterStemmer()
    # Iterate through each token
    clean_tokens = []
    for tok in tokens:
        # Stemmizing, normalize case, and remove leading/trailing white space
        clean_tok = stemmizer.stem(tok.lower().strip())
        # Lemmatize, normalize case, and remove leading/trailing white space
        # clean_tok = lemmatizer.lemmatize(tok.lower().strip(), pos='n')  # Lemmatizing nouns
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model(scoring='precision_weighted'):
    """
    Creates a GridSearch object based on a TfidVectorizer - RandomForestClassifier pipeline

    :param scoring: metric to be used as scoring input for the GridSearch object
    :return: cv: GridSearch object to be used as model
    """
    pipeline = Pipeline([('tfidf_vect', TfidfVectorizer(tokenizer=tokenize)),
                         ('clf', MultiOutputClassifier(RandomForestClassifier())),
                         ])
    parameters = {
        "tfidf_vect__use_idf": [True, False],
        "clf__estimator__n_estimators": [10, 20]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring=scoring, refit=True)
    return cv


def evaluate_model(model, X_test, Y_test):
    """
    Creates a classification report and a  confusion matrix for the model to be evaluated.

    :param model: model to be evaluated
    :param X_test: df containing the features from the test set data
    :param Y_test: df containing the labels/targets from the test set data
    """
    Y_pred = model.predict(X_test)
    for i, col in enumerate(Y_test):
        y_true = Y_test[col]
        y_pred = Y_pred[:, i]
        print("\n---------------------------------------------\n" + col.upper())
        print(classification_report(y_true, y_pred))
        print("\nConfusion matrix:\n{}".format(confusion_matrix(y_true, y_pred)))
        print("\nAccuracy: {}\n".format(accuracy_score(y_true, y_pred, normalize=True)))


def save_model(model, model_filepath):
    """
    Saves ML model into a pickle file

    :param model: model to be saved
    :param model_filepath: filepath for the output pickle file
    """
    filename = model_filepath
    # Saving only the best estimator from GridSearch
    pickle.dump(model.best_estimator_, open(filename, 'wb'))


def main():
    """
    This script runs a ML pipeline running the following steps:
    - Loading cleaned data from a mySQL database
    - Builds a ML model
    - Separates the input data into training and test sets
    - Fits the model into the training set  (choosing the best estimator via GridSearch)
    - Saving the most performing model into a pickle file

    Example on how to run the script:
        python train_classifier.py ../data/DisasterResponse.db classifier.pkl

    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
