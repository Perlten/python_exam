import pandas as pd

def save_fruit(label, percent):
    print(label)
    print(percent)
    df_csv = pd.read_csv('fruit_database.csv')
    #Create test data2
    df2 = pd.DataFrame({'fruit': [label],
        'certainty_level %': [percent]})

    #Concat/appending
    concat = pd.concat([df_csv, df2], sort=True, ignore_index=True)

    #Saving after concat
    concat.to_csv('fruit_database.csv', index=False)
    df_csv = pd.read_csv('fruit_database.csv')
                