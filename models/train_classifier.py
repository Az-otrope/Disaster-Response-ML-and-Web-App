import sys
import os
import re
import pandas as pd
from sqlalchemy import create_engine

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
nltk.download(['punkt', 'wordnet', 'stopwords'])

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report

import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the data
def load_data(database_filepath):
    """
    Load the clean data from the SQL database

    Args:
    database_filepath

    Returns:
    X: features dataframe
    y: target dataframe
    """

    engine = create_engine('sqlite:///' + database_filepath)
    table_name = os.path.basename(database_filepath).split('.')[0]
    df = pd.read_sql_table(table_name,con=engine)

    X = df['message']
    y = df.iloc[:,3:]
    category_names = y.columns

    return X, y, category_names

# text processing
def tokenize(text):
    """
    Process the raw texts includes:
        1. replace any urls with the string 'urlplaceholder'
        2. remove punctuation
        3. tokenize texts
        4. remove stop words
        5. normalize and lemmatize texts

    Args:
    text (str): raw texts

    Return: a list of clean words in their roots form
    """
    # check if there are urls within the text
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex,text)
    for url in detected_urls:
        text = text.replace(url,"urlplaceholder")

    # remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]"," ",text)

    # tokenize the text
    tokens = word_tokenize(text)

    # remove stop words
    tokens = [tok for tok in tokens if tok not in stopwords.words("english")]

    # lemmatization
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok, pos='v').lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# A custom transformer which will identify buzzwords signaling disaster
class DisasterWordExtractor(BaseEstimator, TransformerMixin):

    def disaster_words(self, text):
        """
        INPUT: text - string, raw text data
        OUTPUT: bool -bool object, True or False
        """
        # list of words that are commonly used during a disaster event
        disaster_words = ['food','hunger','hungry','starving','water','drink','eat','thirsty',
                 'need','hospital','medicine','medical','ill','pain','disease','injured','falling',
                 'wound','blood','dying','death','dead','aid','help','assistance','cloth','cold','wet','shelter',
                 'hurricane','earthquake','flood','whirlpool','live','alive','child','people','shortage','blocked',
                 'trap','rob','gas','pregnant','baby','cry','fire','blizard','freezing','blackout','drought',
                 'hailstorm','heat','pressure','lightning','tornado','tsunami']

        # lemmatize the buzzwords
        lemmatized_words = [WordNetLemmatizer().lemmatize(w, pos='v') for w in disaster_words]
        # Get the stem words of each word in lemmatized_words
        stem_disaster_words = [PorterStemmer().stem(w) for w in lemmatized_words]

        # tokenize the input text
        clean_tokens = tokenize(text)
        for token in clean_tokens:
            if token in stem_disaster_words:
                return True
        return False

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        X_disaster_words = pd.Series(X).apply(self.disaster_words)
        return pd.DataFrame(X_disaster_words)


def build_model():
    """
    A pipeline that includes text processing steps and a classifier (random forest).

    Note: GridSearch can be implemented to tune the pipeline parameters.
    However, this final model doesn't include GridSearch due to accuracy scoring 
    """
    # instantiate the pipeline
    model = Pipeline([
    ('features',FeatureUnion([
        ('text_pipeline',Pipeline([
            ('vect',CountVectorizer(tokenizer=tokenize)),
            ('tfidf',TfidfTransformer())
            ])),
        ('disaster_words',DisasterWordExtractor())
        ])),
    ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])

    # specify parameters for grid search
    #params = {'clf__estimator__n_estimators':[100,200],'clf__estimator__max_depth':[5]}

    # create grid search object
    #model = GridSearchCV(pipeline, param_grid=params, cv=3, verbose=3)
    return model

def evaluate_model(model, X_test, y_test, category_names):
    """
    This function evaluates the model performance for each category of

    Args:
        model: the classification returned with optimized parameters
        X_test: feature variable from test set
        y_test: target variable from test set

    OUTPUT
        Classification report and accuracy score
    """
    # predict
    y_pred = model.predict(X_test)

    # classification report
    print(classification_report(y_test.values, y_pred, target_names=category_names))

    # accuracy score
    accuracy = (y_pred == y_test.values).mean()
    print('The model accuracy score is {:.3f}'.format(accuracy))

def save_model(model, model_filepath):
    """ This function saves the pipeline to local disk """

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names= load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
