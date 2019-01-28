# import libraries
import pandas as pd
import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

nltk.download('words')
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')

# This estimator creates two features. One feature is an
# indicator for the word water and the other is an
# indicator for the word food


class WaterFoodExtractor(BaseEstimator, TransformerMixin):
    from sklearn.base import BaseEstimator, TransformerMixin

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_water = X.str.contains('water')
        X_water.name = 'water'
        X_food = X.str.contains('food')
        X_food.name = 'food'

        return pd.concat([X_water, X_food], axis=1)


def load_data(database_filepath):
    '''
        Load dadta from a SQL database.
        ARGUMENTS:
        database_filepath - The path to the filename of
                            the data database

        OUTPUT:
        X - The data features
        Y - The target variables
        category_names - The names of the target variables
    '''

    from sqlalchemy import create_engine

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse.db', engine)
    X = df.message
    Y = df.iloc[:, 4:]
    category_names = Y.columns.values

    return X, Y, category_names


def tokenize(text):
    '''
        Breakup text into individual words(tokens), remove stopwords,
        and lemmatize.
        ARGUMENTS:
        text - a string that represents one or more sentences.

        RETURNS:
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


def build_model():
    '''
        Create a model to train on word token data and
        multi-class targets.
        ARGUMENTS:
        None

        RETURNS:
        model - the machine learning model to train word token data
    '''

    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.pipeline import FeatureUnion
    from sklearn.feature_extraction.text import (CountVectorizer,
                                                 TfidfTransformer)

    # Combine NLP features and train an AdaBoost Classifier
    model = Pipeline([
        ('features', FeatureUnion([
            ('nlp_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize,
                                         max_df=0.75, ngram_range=(1, 2))),
                ('tfidf', TfidfTransformer(norm='l2',
                                           smooth_idf=True,
                                           sublinear_tf=True)),
            ])),
            ('water_food_extractor', WaterFoodExtractor())
        ])),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(learning_rate=0.5)))
    ])

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
        Evaluate the model on test data.
        ARGUMENTS:
        model - the trained model
        X_test - the test features
        Y_test - the test target values
        category_names - the names of the target variables

        RETURNS
        None
    '''

    from sklearn.metrics import classification_report

    Y_pred = model.predict(X_test)

    # Convert prediction to a dataframe for use in the classification report
    Y_pred_df = pd.DataFrame(Y_pred, columns=category_names)

    # Evaluate the score of each category column
    for col_pred, col_true in zip(Y_pred_df, Y_test):
        print(col_pred)
        print(classification_report(Y_test[col_true], Y_pred_df[col_pred]))


def save_model(model, model_filepath):
    '''
        Save a model to disk.
        ARGUMENTS:
        model - the model to save
        model_filepath - the name of the data file to which the model was saved

        RETURNS:
        None
    '''

    import pickle

    # save the model to disk
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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
