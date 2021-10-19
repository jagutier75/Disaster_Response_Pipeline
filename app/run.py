import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar

import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# Load data
engine = create_engine('sqlite:///../data/DisasterMessages.db')
df = pd.read_sql_table('Messages', engine)

# Load model
model = joblib.load("../models/model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # ----- VISUALS RENDERING ----- #

    # Extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Create visuals
    # TODO: Below is an example - modify to create your own visuals
    graph = []
    graph.append(Bar(x=genre_names,
                     y=genre_counts
                     )
                 )
    layout = dict(title='Chart Two',
                  xaxis=dict(title='x-axis label',),
                  yaxis=dict(title='y-axis label'),
                  )

    # Append all charts to the figures list
    figures = [dict(data=graph, layout=layout)]
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
