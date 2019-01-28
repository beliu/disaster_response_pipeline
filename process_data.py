# import libraries
import pandas as pd
import sys


def load_data(messages_filepath, categories_filepath):
    '''
        Loads message and category data from files and combines both
        into one dataframe
        ARGUMENTS:
        messages_filepath - filepath for the message data
        categories_filepath - filepath for the categories data

        RETURNS:
        df - a pandas dataframe that combines the two datasets
    '''

    # Read in the data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv('categories.csv')

    # merge datasets
    df = pd.merge(messages, categories, on='id')

    return df


def clean_data(df):
    '''
        Cleans up the category values and converts them to int values of [0, 1]
        ARGUMENTS:
        df - a pandas dataframe that contains message and categories data

        RETURNS:
        df - a cleaned-up version of the dataframe
    '''

    # create a dataframe of the 36 individual category columns
    categories_split = df.categories.str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories_split.iloc[0, :]

    # use this row to extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x[:-2])
    categories_split.columns = category_colnames

    # Convert each category value to be just 0 or 1
    for column in categories_split:
        # set each value to be the last character of the string
        categories_split[column] = categories_split[column].apply(
            lambda x: x[-1])

        # convert column from string to numeric
        categories_split[column] = categories_split[column].astype('int64')

    # Set ceiling of all dataframe values to 1
    categories_split = categories_split.clip(upper=1)

    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat((df, categories_split), axis=1)

    # drop duplicates
    df = df.drop_duplicates('id')

    return df


def save_data(df, database_filename):
    '''
        Save a dataframe into a SQLite database.
        ARGUMENTS:
        df - a pandas dataframe
        database_filename - the filename of the SQLite database

        RETURNS:
        None
    '''

    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponseTable', engine, index=False,
              if_exists='replace')


def main():
    if len(sys.argv) == 4:

        (messages_filepath, categories_filepath,
         database_filepath) = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
