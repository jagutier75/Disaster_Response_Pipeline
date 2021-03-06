import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# Remove chain assignment warning
pd.options.mode.chained_assignment = None  # default='warn'


def positive_label_count(df, threshold=2.0, col_to_ignore=None):
    """
    This function counts the number of positive labels for each class (considering the cardinality of each class = 2),
    and prints the name of the classes that are below a certain threshold.

    :param df: DataFrame containing the columns to be counted
    :param threshold: Columns containing less positive labels than this threshold will be printed
    :param col_to_ignore: Columns not to be considered in the analysis
    :return: None
    """
    # Counting the number of positive labels in each target column
    if col_to_ignore is None:
        col_to_ignore = ['id', 'message', 'original', 'genre']
    few_data_points_col = []
    print('Total number of messages: {}\n'.format(df.shape[0]))
    for i, col in enumerate(df):
        if col not in col_to_ignore:
            pct_positive_labels = df[df[col] == 1][col].sum()*100/df.shape[0]
            print("{}: {}/ {:.4f}%".format(col, df[df[col] == 1][col].sum(), pct_positive_labels))
            if pct_positive_labels < threshold:
                few_data_points_col.append(col)
    print('\n Columns with less than {}% of positive labels remaining:\n{}\n'.format(threshold, few_data_points_col))
    return


def merge_categories(df, col_merge_pairs):
    """
    This function helps to copy the positive labels in the column specified as key in the col_merge_paris dict
    to the column specified as value in the same dictionary in a dataframe. Once the copy is done, the column specified
    in key will be dropped from the original dataframe.

    :param df: Original dataframe
    :param col_merge_pairs: dictionary containing column names. The key contains the column to remove once the positive labels
    are copied to the column specified as value in the dictionary.
    :return: Original dataframe with the columns removed
    """
    # Merging related columns (columns with low number of positive labels with similar columns)
    for key, value in col_merge_pairs.items():
        df.loc[df[key] == 1, value] = 1
        df = df.drop(key, axis=1)
    return df


def load_data(messages_filepath, categories_filepath):
    """
    Merge data from two CSV files to a pandas dataframe

    :param messages_filepath: CSV containing the messages
    :param categories_filepath: CSV containing the categories (targets) for each message
    :return: DataFrame with the merged data
    """
    # Load messages dataset
    messages = pd.read_csv(messages_filepath)
    # Load categories dataset
    categories = pd.read_csv(categories_filepath)
    # Merge datasets
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """
    Performs cleaning algorithms to the DataFrame df:

    - Creates individual columns for each category in the "categories" column
    - Verifies that each target column has a cardinality of 2 (binary)
    - Removes duplicates
    - Removes target columns containing just a single label
    - Merge target columns for those columns containing few positive labels with similar target columns

    :param df: Original DataFrame
    :return: df: Cleaned DataFrame
    """
    # Create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # Select the first row of the categories dataframe
    row = categories.iloc[0, :]

    # Extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2])

    # Rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1]).astype('int64')

    # Cleaning the extra class in the "related" category
    misclassed_idx = categories[categories['related'] == 2].index
    categories['related'].iloc[misclassed_idx] = 1
    assert (categories['related'] == 2).sum() == 0, "ERROR: There are more than 2 classes in the related column"

    # Drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)
    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # Remove "original" column that is useless
    df = df.drop(columns=['original'])

    # Drop duplicates
    if df.duplicated(subset=['message']).sum() > 0:
        df = df.drop_duplicates(subset=['message'])
    assert df.duplicated(subset=['message']).sum() == 0, "ERROR: There are still duplicates in the data"

    # Automatically removing columns that contain just a single label
    single_label_col = []
    for i, col in enumerate(df):
        if col not in ['id', 'message', 'original', 'genre']:
            if df[col].unique().shape[0] == 1:
                single_label_col.append(col)
    df.drop(columns=single_label_col, inplace=True)

    # Merging related columns (columns with low number of positive labels with similar columns)
    col_merge_pairs = {'offer': 'aid_related',
                       'fire': 'other_aid',
                       'shops': 'other_infrastructure',
                       'hospitals': 'other_infrastructure',
                       'tools': 'aid_related',
                       'missing_people': 'search_and_rescue',
                       'clothing': 'aid_related',
                       'aid_centers': 'aid_related',
                       'security': 'other_aid',
                       }
    # positive_label_count(df)
    df = merge_categories(df, col_merge_pairs)

    return df


def save_data(df, database_filename):
    """
    Saves a DataFrame into a mySQL database

    :param df: DataFrame to save
    :param database_filename: Path to the database
    """
    filepath = 'sqlite:///' + database_filename
    engine = create_engine(filepath)
    df.to_sql('Messages', engine, index=False, if_exists='replace')


def main():
    """
    The main function performs the complete ETL pipeline:

    - Extract: Read from CSV files both messages and categories
    - Transform: Cleans the resulting DataFrame
    - Load: Saves the resulting DataFrame into a mySQL database

    Example for calling this script:
    python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db'

    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        # python process_data.py disaster_messages.csv disaster_categories.csv DisasterMessages.db

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the file paths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
