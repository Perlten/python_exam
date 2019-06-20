import pandas as pd

FILE_NAME = 'fruit_database.csv'

def save_fruit(label, percent):
    df_csv = pd.read_csv(FILE_NAME)
    #Create test data2
    df2 = pd.DataFrame({'fruit': [label],
        'certainty_level %': [percent]})

    #Concat/appending
    concat_df = pd.concat([df_csv, df2], sort=True, ignore_index=True)

    #Saving after concat
    concat_df.to_csv(FILE_NAME, index=False)
                