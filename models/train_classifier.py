import sys
import pandas as pd 
import re
from sqlalchemy import create_engine
import nltk
nltk.download(['stopwords','punkt','wordnet'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pickle


def load_data(database_filepath):
    """
    Load data from database
    
    Input: sqlite database file
    Output: X feature and y target variables
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(engine.table_names()[0], engine)
    
    X = df['message']
    y = Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    
    return X, y


def tokenize(text):
    text = re.sub('[^a-zA-Z0-9]',' ',text) # Remove characters that not in A-Z, a-z, 0-9
    
    text = text.lower() # Convert all characters to lowercase
    
    words = word_tokenize(text) # Tokenize text into list of words 
    
    enstopwords = stopwords.words("english")
    words = [w for w in words if not w in enstopwords] # Remove stopwords in English
    
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words] # Lemmatize words
    return lemmed


def build_model():
    pipeline = Pipeline([
                   ('vect', CountVectorizer(tokenizer=tokenize)),
                   ('tfidf', TfidfTransformer()),
                   ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators': [1],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2) 

    return cv



def evaluate_model(model, X_test, Y_test):
    pred = model.predict(X_test)

    num_categories = Y_test.shape[1]
    for index in range(num_categories):
        col = Y_test.columns[index]
        print(col)
        print(classification_report(Y_test[col], pred[:, index]))



def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()