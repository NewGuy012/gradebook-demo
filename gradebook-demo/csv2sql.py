import os
import glob
import pandas as pd
from pathlib import Path
from getpass import getpass
from mysql.connector import connect, Error
from sqlalchemy import create_engine

# Import csv as pandas
file_path = Path(
    r'C:\Users\Church\Desktop\Visual Studio Code\Gradebook\Data')
files = file_path.glob('*.csv')

for file in files:
    print(file)

    data = pd.read_csv(file, index_col=False)

    # Connect to MySQL
    db_user = os.environ.get('DB_USER')
    db_pass = os.environ.get('DB_PASS')
    db_name = 'gradebook_db'

    try:
        with connect(
                host="localhost",
                user=db_user,  # input("Enter username: "),
                password=db_pass,  # getpass("Enter password: "),
                database=db_name) as connection:
            print(connection)
    except Error as e:
        print(e)

    # Convert csv to sql
    engine = create_engine("mysql+mysqlconnector://{user}:{pw}@localhost/{db}"
                           .format(
                               user=db_user,
                               pw=db_pass,
                               db=db_name))
    file_name = file.stem
    data.to_sql(file_name, con=engine, if_exists='replace',
                chunksize=100, index=False)
