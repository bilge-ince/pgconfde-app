
import os
from utils.db_connection import create_db_connection
import pandas as pd
import psycopg2
import os

from typing import Tuple, List

from psycopg2.extras import execute_batch
from psycopg2 import sql

def create_img_embedding_table(conn: psycopg2.extensions.connection)-> None:
    with conn.cursor() as cur:
        cur.execute("""CREATE TABLE IF NOT EXISTS products_embeddings_ollama (img_id INTEGER PRIMARY KEY REFERENCES products_pgconf(img_id) ON DELETE CASCADE,
            embedding vector(4096));""")
    
def create_generic_tables(conn: psycopg2.extensions.connection)-> None:
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS products_pgconf;")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS products_pgconf (
                img_id INTEGER PRIMARY KEY,
                gender VARCHAR(50),
                masterCategory VARCHAR(100),
                subCategory VARCHAR(100),
                articleType VARCHAR(100),
                baseColour VARCHAR(50),
                season TEXT,
                year INTEGER,
                usage TEXT NULL,
                productDisplayName TEXT NULL
            );"""
        )


def populate_product_data(conn: psycopg2.extensions.connection, csv_file: str
) -> None:
    # Create a string buffer
    # Read the train.csv file into a pandas dataframe, skipping bad lines
    df = pd.read_csv(csv_file, on_bad_lines="skip")
    df_copy = df.copy()
    # Drop rows where any column value is empty
    df_copy = df_copy.dropna()
    # Convert year to integer if it's not already
    df_copy["year"] = df_copy["year"].astype("Int64")

    # Replace NaN with None for proper NULL handling in PostgreSQL
    df_copy = df_copy.replace({pd.NA: None, pd.NaT: None})
    df_copy = df_copy.where(pd.notnull(df_copy), None)
    print("Starting to populate products_pgconf table")
    # Convert DataFrame to csv format in memory

    tuples: List[Tuple] = [tuple(x) for x in df_copy.to_numpy()]
    cols_list: List[str] = list(df_copy.columns)
    cols: str = ",".join(cols_list)
    placeholders: str = ",".join(
        ["%s"] * len(cols_list)
    )  # Create the correct number of placeholders
    # Create a parameterized query
    query: sql.SQL = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
        sql.Identifier("products_pgconf"), sql.SQL(cols), sql.SQL(placeholders)
    )
    cursor: psycopg2.extensions.cursor = conn.cursor()
    try:
        execute_batch(cursor, query, tuples)
        conn.commit()
    except (Exception, psycopg2.Error) as error:
        print(f"Error while inserting data into PostgreSQL: {error}")
        conn.rollback()

    # Commit and close
    conn.commit()
    print("Finished populating products table")

if __name__ == "__main__":
    conn = create_db_connection()
    create_generic_tables(conn)
    populate_product_data(conn, 'dataset/updated_stylesc.csv')
    conn.commit()
    conn.close()