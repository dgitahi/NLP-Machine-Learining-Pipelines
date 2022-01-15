# import libraries
import pandas as pd
import numpy as np
from sqlalchemy  import create_engine
import sqlite3
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import f1_score,confusion_matrix,classification_report

from sklearn.model_selection import GridSearchCV








def load_data(directory_file_path):
    engine = create_engine(directory_file_path)
    df = pd.read_sql_table('df_project',con= engine)
    df =df[(df['related']== 1) & (df['related']==0)]
    X = df['message']
    category_names = ['related', 'request', 'offer',
           'aid_related', 'medical_help', 'medical_products',
           'search_and_rescue', 'security', 'military', 'child_alone', 'water',
           'food', 'shelter', 'clothing', 'money', 'missing_people',
           'refugees', 'death', 'other_aid', 'infrastructure_related',
           'transport', 'buildings', 'electricity', 'tools', 'hospitals',
           'shops', 'aid_centers', 'other_infrastructure', 'weather_related',
           'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather',
           'direct_report']
    y = df[category_names]

    return X,y,category_names





def tokenize(text):
    
    tokenizer = nltk.RegexpTokenizer(r"\w+") 
    #tokens = word_tokenize(text)
    tokens= tokenizer.tokenize(text)
    lemmatizer = WordNetLemmatizer()
    #lemm

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok,pos= 'v').lower().strip()
        clean_tok = lemmatizer.lemmatize(clean_tok,pos= 'n').lower().strip()
        #clean_tok = PorterStemmer().stem(clean_tok)
        if clean_tok not in stopwords.words('english'):
            clean_tokens.append(clean_tok)
    

    return clean_tokens





def build_model():

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),

        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ]) 


    parameters ={
        'clf__estimator__n_estimators':[50,100,200],
        'clf__estimator__criterion':['gini', 'entropy']
    
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred)
    #col_names = y_test.columns.values
    y_pred.columns = category_names

target_names = ['class 0', 'class 1','class 2']
for col in category_names:
    print(col)
    print(classification_report(Y_test[col], y_pred[col]
                            , target_names=target_names))



def save_model(model, model_filepath):
    """ Saving model's best_estimator_ using pickle
    """
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))
    


#`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
    
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data('sqlite:///ProjectMLPipelines.db')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

