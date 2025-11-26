import pandas as pd

def read_stata_file(filename):
    # reading stata file
    df = pd.read_stata(filename)
    print(df.head())
    return df

def read_large_stata_file(filename):
    # reading large files
    # create iterator
    itr = pd.read_stata(filename, iterator=True)

    # Read data in chunks or loop through chunks
    chunk = itr.get_chunk(1000) # Reads the next 1000 rows

    # Read specific number of rows at a time
    chunk = itr.get_chunk(1000) # Reads the next 1000 rows

    for chunk in itr:
        # Process each chunk of data
        print(chunk.head())
    return df

read_stata_file('data/replication files/Data/PovertyTraps_replication_data.dta')
