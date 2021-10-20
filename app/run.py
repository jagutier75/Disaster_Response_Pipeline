import json
import plotly
import numpy as np
import pandas as pd
import re

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie, Figure, Heatmap

import joblib
from sqlalchemy import create_engine


def figures_rendering(df):
    """
    Creates a list of figures based on the input DataFrame data:

    - Message quantity by genre category
    - Category with the highest number of messages
    - Number of positive/negative messages by category in the dataset
    - Heatmap showing the most related categories based on positive labels

    :param df: DataFrame to extract the information from
    :return: list of figures to be plotted
    """
    # Extract data needed for visuals

    # ----- MESSAGE BY GENRE ----- #
    # Bar chart
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # ----- CATEGORY WITH THE HIGHEST NUMBER OF MESSAGES -----#
    # Pie chart / bar
    cat_col = set(df.columns).difference({'id', 'message', 'genre', 'original'})
    # test = df[cat_col].sum(axis=0).sort_values(ascending=False) * 100 / (df[cat_col].sum().sum())
    test = df[cat_col].sum(axis=0).sort_values(ascending=False)
    msg_cat_name = test.index.tolist()  # Category names
    msg_count = test.values.round(2)  # Percentage of messages per category

    # ----- NUMBER OF POSITIVE/NEGATIVE MESSAGES PER CATEGORY IN THE DATASET -----#
    cat_col = set(df.columns).difference({'id', 'message', 'genre', 'original'})
    test = df[cat_col]
    neg_label_count, pos_label_count = {}, {}
    x, y_0, y_1 = [], [], []
    for col in test:
        neg_label_count[col] = df[col].value_counts()[0]
        pos_label_count[col] = df[col].value_counts()[1]
    # Sorting the values
    for key, value in sorted(neg_label_count.items(), key=lambda item: item[1], reverse=True):
        x.append(key)
        y_0.append(value)
        y_1.append(pos_label_count[key])
    pos_neg_msg_cat = x
    pos_neg_y0 = y_0
    pos_neg_y1 = y_1

    # ----- HEAT MAP ----- #
    cat_col = set(df.columns).difference({'id', 'message', 'genre', 'original', 'related', 'direct_report'})
    test = df[cat_col]

    ord_col = test.columns.sort_values().to_list()
    hm_dict = {}
    for col in ord_col:
        # Recovering labels most related to the current category
        tmp_df = test.groupby(col).sum().loc[1, :]
        col_max_value = test[col].sum()
        tmp_dict = {col: col_max_value}
        for idx in tmp_df.index:
            tmp_dict[idx] = tmp_df[idx]
        hm_dict[col] = tmp_dict

    arr_len = len(hm_dict.keys())
    hm_array = np.zeros(shape=(arr_len, arr_len))

    for i, col_i in enumerate(ord_col):
        for j, col_j in enumerate(ord_col):
            hm_array[i][j] = hm_dict[col_i][col_j]

    hm_array_norm = np.zeros(shape=(arr_len, arr_len))
    for i, row in enumerate(hm_array):
        norm = np.linalg.norm(row)
        norm_row = row / norm
        hm_array_norm[i, :] = norm_row

    # Create visuals
    graph_1 = [Bar(x=genre_names,
                   y=genre_counts
                   )]
    layout_1 = dict(title='Message source',
                    xaxis=dict(title='Source', ),
                    yaxis=dict(title='# of messages'),
                    )

    graph_2 = [Pie(labels=msg_cat_name,
                   values=msg_count,
                   hole=.3,
                   )]
    layout_2 = dict(title='Percentage of positive messages by category',
                    )

    graph_3 = Figure(data=[Bar(name='Negative labels (0)', x=pos_neg_msg_cat, y=pos_neg_y0),
                           Bar(name='Positive labels (1)', x=pos_neg_msg_cat, y=pos_neg_y1)])
    graph_3.update_layout(title='Number of positive and negative labels per category',
                          title_x=0.5,
                          barmode='stack')  # Change the bar mode; other option barmode='group'

    graph_4 = Figure(data=Heatmap(z=hm_array_norm,
                                  x=ord_col,
                                  y=ord_col,
                                  hoverongaps=False))

    # Append all charts to the figures list
    figures = []
    figures.append(dict(data=graph_1, layout=layout_1))
    figures.append(dict(data=graph_2, layout=layout_2))
    figures.append(dict(data=graph_3, layout=None))
    figures.append(dict(data=graph_4, layout=None))

    return figures


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


# Initializing Flask app
app = Flask(__name__)
# NLTK package download
nltk.download('stopwords')
# Load data
engine = create_engine('sqlite:///../data/DisasterMessages.db')
df = pd.read_sql_table('Messages', engine)
# Load model
model = joblib.load("../models/model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    Main webpage. It sends the figures and idx to be used for plotting using Plotly
    :return: None
    """
    # ----- VISUALS RENDERING ----- #
    figures = figures_rendering(df)
    # Plot ids for the html id tag
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]
    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('master.html',
                           ids=ids,
                           figuresJSON=figuresJSON)


# Web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    Receives a query from the /index page containing the message to be classified. it performs a prediction
    and returns the prediction labels to be rendered through the /go.html page
    :return:
    """
    # Save user input in query
    query = request.args.get('query', '') 

    # Use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
