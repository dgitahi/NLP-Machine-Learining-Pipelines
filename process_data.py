import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    The function loads the two dataset and outputs a data frame
    """
    #load the data set
    messages = pd.read_csv(messages_filepath)

    categories = pd.read_csv(categories_filepath)
    #merge dataset
    df = pd.merge(messages,categories,on = 'id')
    return df






def clean_data(df):
    """
    The functions takes a dataframe. 
    Cleans the data and returns a dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories =[pd.Series(str(df['categories'][i])).str.split(pat=';',expand = True) for i in range(len(df))]
    categories= pd.concat(categories,ignore_index=True)
    # select the first row of the categories dataframe and use it to extract a list of new column names for categories.
    row = list(categories.iloc[0])
    category_colnames = [cat.split('-')[0] for cat in row]
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str.get(1)
    
    # convert column from string to numeric
    categories[column] = categories[column].astype('float')

    # drop the original categories column from `df`
    df.drop(columns= 'categories',inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = df.join(categories)

    df.drop_duplicates(inplace=True)

    return df



def save_data(df, database_filepath):
    engine = create_engine('sqlite:///'+ str (database_filepath))
    df.to_sql('Categories_df', engine, index=False, if_exists = 'replace')






def main():
    if len(sys.argv) == 4:
      
        messages_filepath, categories_filepath,database_filepath  = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

