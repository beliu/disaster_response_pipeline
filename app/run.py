import json
import plotly
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar, Layout
from sklearn.externals import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin
nltk.download('stopwords')


class WaterFoodExtractor(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X = X[0]
        X_water = 'water' in X
        X_food = 'food' in X

        return [X_water, X_food]


app = Flask(__name__)


def tokenize(text):
    '''
                Breakup text into individual words(tokens),
                remove stopwords, and lemmatize them.
                ARGUMENTS:
                text - a string that represents one or more sentences.

                OUTPUT:
                tokens - a list of word tokens derived from the text.
    '''

    import re
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Make text lower case
    text = text.lower().strip()
    # Tokenize the sentence
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [
        word for word in tokens if word not in
        nltk.corpus.stopwords.words("english")
    ]
    # Lemmatize each word
    tokens = [WordNetLemmatizer().lemmatize(word, pos='v') for word in tokens]

    return tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponseTable', engine)

# load model
model = joblib.load("../models/final_model.sav")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # Get the average number of food or water-related emergencies by
    # message genre
    genre_mean_df = df.groupby('genre')[['water', 'food']].mean()

    # Create the plotly data trace for each genre
    data_one = []
    for genre in genre_mean_df.index:
        data_one.append(
            Bar(
                x=genre_mean_df.columns.tolist(),
                y=genre_mean_df.loc[genre].values,
                name=genre
            )
        )

    layout_one = Layout(
        title=' Rate of Water or Food Emergencies <br> by Message Genre',
        xaxis=dict(title='Type of Emergency'),
        yaxis=dict(title='Rate of Occurrence'),
        barmode='group'
    )

    # See what are the most common types of aid being offered by people to help
    # those in emergencies
    df_aid = df[df.offer == 1].sum()
    df_aid = df_aid[['medical_help', 'medical_products', 'search_and_rescue',
                     'water', 'food', 'shelter', 'clothing',
                     'money']].sort_values(ascending=False)
    data_two = [
        Bar(
            x=df_aid.index.tolist(),
            y=df_aid.values.tolist()
        )
    ]

    layout_two = Layout(
        title='Number of Offers of Help <br> by Type of Help',
        xaxis=dict(title='Type of Help'),
        yaxis=dict(title='Number of Offers')
    )

    # append all charts to the graphs list
    graphs = []
    graphs.append(dict(data=data_one, layout=layout_one))
    graphs.append(dict(data=data_two, layout=layout_two))

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
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
