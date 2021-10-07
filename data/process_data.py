import sys
import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import warnings
warnings.filterwarnings("ignore")


def load_data(messages_filepath, categories_filepath):
    """
    Load the messages and categories datasets and merge them into a dataframe
    INPUT: messages_filepath (.csv file), catgeories_filepath (.csv file)
    OUTPUT: a merged df
    """
    messages = pd.read_csv(messages_filepath,dtype=str)
    categories = pd.read_csv(categories_filepath,dtype=str)
    df = pd.merge(messages,categories,how='inner',on='id')
    return df

def clean_data(df):
    """
    Split the categories into 36 columns each represents a category.
    Each meassage receives a value of 1 for the category its belong to, and 0 for others
    INPUT: merged df
    TASK:
        1. Split categories into seperate category columns
        2. Convert category values to just numbers 0 or 1
        3. Replace categories column in df with new category columns
        4. Remove duplicates
    OUTPUT: a dataframe in which each unique meassage is labeled with a category
    """

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # categories column names
    category_colnames = list(map(lambda i:row[i][:-2], range(len(row))))
    # rename the columns of `categories`
    categories.columns = category_colnames

    # interate through each column to extract the label value (1,0)
    for column in categories.columns:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.split('-').str[1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        # convert any values different than 0 and 1 to 1
        categories[column].loc[(categories[column]!=0)&(categories[column]!=1)]=1

    # drop the original `categories` column and `original` from `df`
    df.drop(['categories','original'],axis=1,inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    # drop the `child_alone` column from `df` - this column only exists after concatenation
    df.drop(['child_alone'],axis=1,inplace=True)
    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):
    """
    Save the clean dataset into a squlite databaset

    Args:
    df: clean data return from clean_data() function
    database_filename (str): filename.db of SQL database in which the clean dataset will be stored

    Returns:
    None
    """
    engine = create_engine('sqlite:///'+database_filename)
    table_name = os.path.basename(database_filename).split('.')[0]
    df.to_sql(table_name, engine, index=False,if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

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
