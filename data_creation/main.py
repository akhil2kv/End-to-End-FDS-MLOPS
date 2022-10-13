import argparse
import pandas as pd
from cassandra.cluster import Cluster

from transaction_data_simulation import gen_and_save

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser(description='Fraud detection model')

    # Add the arguments
    my_parser.add_argument('-g',
                       '--gen-data',
                       action='store_true',
                       help='generate data')

    # Execute the parse_args() method
    args = my_parser.parse_args()
    if args.gen_data:
        gen_and_save()
    cluster = Cluster(['127.0.0.1'], port=9042)
    session = cluster.connect()
    ## TODO: Add all columns
    # create keyspace "event" and table "blobtest" if not exist
    session.execute("CREATE KEYSPACE IF NOT EXISTS fraudkeyspace WITH replication = {'class': 'SimpleStrategy', 'replication_factor': '1'}  AND durable_writes = true;")
    session.execute("CREATE TABLE IF NOT EXISTS fraudkeyspace.fraudtable1 ( TRANSACTION_ID int PRIMARY KEY, TX_DATETIME datetime,CUSTOMER_ID int)")

    session.execute('USE fraudkeyspace')

    strCQL = "INSERT INTO fraudtable1 (TRANSACTION_ID,TX_DATETIME ,CUSTOMER_ID) VALUES (?,?,?)"
    pStatement = session.prepare(strCQL)

    ## TODO: Read all csv files 
    df = pd.read_csv("simulated-data-raw/2018-04-02.csv")
    df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])
    for index, row in df.iterrows():
        session.execute(pStatement,[row.TRANSACTION_ID,row.TX_DATETIME,row.CUSTOMER_ID])
    print(session.execute("SELECT * FROM fraudtable1").one())
