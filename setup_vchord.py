import pandas as pd
import psycopg2
import time
import os

from io import StringIO
from PIL import Image
# from transformers import CLIPModel, CLIPProcessor, AutoTokenizer, AutoModel
import sys
# Add the parent directory of 'code' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("Python search paths:", sys.path)  # Debugging line
from utils.db_connection import create_db_connection
from utils.generate_embeddings import generate_ollama_embeddings


def initialize_database(conn):
    """Initialize the database with required extensions and tables."""
    with conn.cursor() as cur:
        _create_extensions(cur)
        _create_tables(cur)
        # _populate_test_images_data(cur, '/dataset/images')


def _create_extensions(cur):
    """Create required extensions if they do not exist."""

def create_vchord_table(cur):
    cur.execute("""CREATE TABLE IF NOT EXISTS products_embeddings_ollama (img_id INTEGER PRIMARY KEY REFERENCES products(img_id) ON DELETE CASCADE,
            embedding vector(4096));""")

def _create_tables(cur):
    """Create required tables."""
    cur.execute("DROP TABLE IF EXISTS products;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS products (
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
        );""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS products_embeddings(
            img_id INTEGER PRIMARY KEY REFERENCES products(img_id) ON DELETE CASCADE,
            embedding vector(384),
            image_embedding vector(512));
    """)

def _populate_product_data(conn, csv_file):
    # Create a string buffer
    # Read the train.csv file into a pandas dataframe, skipping bad lines
    df = pd.read_csv(csv_file, on_bad_lines="skip")  
    output = StringIO()
    df_copy = df.copy()

    # Convert year to integer if it's not already
    df_copy['year'] = df_copy['year'].astype('Int64')

    # Replace NaN with None for proper NULL handling in PostgreSQL
    df_copy = df_copy.replace({pd.NA: None, pd.NaT: None})
    df_copy = df_copy.where(pd.notnull(df_copy), None)
    
    # Convert DataFrame to csv format in memory
    df_copy.to_csv(output, sep='\t', header=False, index=False, na_rep='\\N')
    output.seek(0)
    # Copy the data to the products table
    with conn.cursor() as cur:
        # Use COPY to insert data
            cur.copy_from(
                file=output,
                table='products',
                null='\\N'
            )

    # Commit and close
    conn.commit()

def create_vchord_indexes(conn):
    cur = conn.cursor()
    # cosine for text embedding
    cur.execute("""CREATE INDEX ON products_embeddings USING vchordrq (embedding vector_cosine_ops) WITH (options = $$
    residual_quantization = false
    [build.internal]
    lists = [1000]
    spherical_centroids = true
    $$);""")
    #L2 for image embedding
    cur.execute("""CREATE INDEX ON products_embeddings USING vchordrq (image_embedding vector_l2_ops) WITH (options = $$
    residual_quantization = true
    [build.internal]
    lists = [1000]
    spherical_centroids = false
    $$);""")

def generate_store_embeddings(conn, base_path, batch=10):
    """
    This function is a specific implementation for vchord semantic search capability
    """
    function_start_time = time.time()
    # Run for S3 bucket
    # The idea is to create a retriever for the images bucket so the image search can run over it.
    # Load the model and processor with timing
    fetch_start = time.time()
    cursor = conn.cursor()
    cursor.execute("SELECT img_id, productdisplayname FROM products;")
    result = cursor.fetchall()
    fetch_end = time.time()
    total_rows_inserted = 0
    for i in range(3676, len(result)):
        batch_text = result[i][1]
        if batch_text:
            embedding_output = generate_ollama_embeddings(batch_text)
            cursor.execute(
                                "INSERT INTO products_embeddings_ollama (img_id, embedding) "
                                "VALUES (%s, %s)",
                                (result[i][0], embedding_output)
                            )
    
    function_end_time = time.time()
    total_time = function_end_time - function_start_time
    print(f"Total Rows: {total_rows_inserted}")
    print(f"Total function execution time: {total_time} seconds")
    print(f"Fetching time: {fetch_end - fetch_start} seconds")

def load_images_batch(batch_ids, base_path, processor):
    images, valid_paths = [], []
    for image_id in batch_ids:
        image_path = f"{base_path}/{image_id}.jpg"
        try:
            img = Image.open(image_path)
            img.verify()  # Verify the image integrity
            img = Image.open(image_path)  # Reopen to reset file pointer
            images.append(img)
            # valid_paths.append(image_path)
        except OSError as e:
            print(f"Failed to process image {image_path}: {e}")
            continue  # Skip problematic images
    if images:
        return processor(text=["dummy text"] * len(images), images=images, return_tensors="pt", padding=True)
    else:
        return None, []

def main():
        conn=None
        try:
            conn = create_db_connection() # Connect to the database
            conn.autocommit = True  # Enable autocommit for creating the database
            start_time = time.time()
            initialize_database(conn) # Initialize the db with aidb, pgfs extensions and necessary tables
            _populate_product_data(conn, 'dataset/stylesc.csv') # Populate the products table with the stylesc.csv data
            generate_store_embeddings(conn, 'dataset/images', 25) # Create and refresh the retriever for the products table and images bucket
            vector_time = time.time() - start_time
            print(f"Total process time: {vector_time:.4f} seconds.")
        except (Exception, psycopg2.DatabaseError) as error:
            print(f"Error: {error}")
        finally:
            if conn:
                    conn.close()

if __name__ == "__main__":
    main()