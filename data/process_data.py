import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

"""
Takes in 2 CSV files, converts them to Pandas Dataframes, and then merges them

input: filepath of both CSV files
output: merged DF
"""
def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    #merge datasets together for 1 df
    df = messages.merge(categories, on='id')
    return df

"""
Cleans data by splitting categories and making new columns, then recombining into a new Dataframe. Drops duplicates and illogical data

input: df to be cleaned
output: Cleaned df
"""
def clean_data(df):
	#split categories into a new df
    categories = df['categories'].str.split(';', expand = True)

    #Rename Column Headers to Different Categories
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    #remove category names from data
    for column in categories:
    	# set each value to be the last character of the string
    	categories[column] = categories[column].astype(str).str[-1:]
    	# convert column from string to numeric
    	categories[column] = categories[column].astype(int)

    #merge categories with df and remove duplicates
    df = df[['id','message', 'original', 'genre']]
    df2 = df.join(categories, how='left')
    df2.drop_duplicates(inplace=True)
    df2 = df2[df2.related != 2]
    return df2

"""
Saves Dataframe to a sqlite database file

input: dataframe to be saved, Database filename
"""
def save_data(df, database_filename):
	engine = create_engine('sqlite:///'+database_filename)
	df.to_sql('Disaster Response Data', engine, index=False, if_exists="replace")
      

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df= clean_data(df)
        
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