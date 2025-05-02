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
from utils.generate_embeddings import generate_text_embeddings


def initialize_database(conn):
    """Initialize the database with required extensions and tables."""
    with conn.cursor() as cur:
        _create_tables(cur)


def _create_tables(cur):
    """Create required tables."""
    cur.execute("""
        CREATE TABLE IF NOT EXISTS products_embeddings_vchord(
            img_id INTEGER PRIMARY KEY REFERENCES products_pgconf(img_id) ON DELETE CASCADE,
            embedding vector(384),
            image_embedding vector(512));
    """)

def create_vchord_indexes(conn):
    cur = conn.cursor()
    # cosine for text embedding
    cur.execute("""CREATE INDEX ON products_embeddings_vchord USING vchordrq (embedding vector_cosine_ops) WITH (options = $$
    residual_quantization = false
    [build.internal]
    lists = [1000]
    spherical_centroids = true
    $$);""")
    #L2 for image embedding
    cur.execute("""CREATE INDEX ON products_embeddings_vchord USING vchordrq (image_embedding vector_l2_ops) WITH (options = $$
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
    cursor.execute("SELECT img_id, productdisplayname FROM products_pgconf;")
    result = cursor.fetchall()
    fetch_end = time.time()
    total_rows_inserted = 0
    for i in range(3676, len(result)):
        batch_text = result[i][1]
        if batch_text:
            embedding_output = generate_text_embeddings(batch_text)
            insert_query = """INSERT INTO products_embeddings_vchord (img_id, embedding)
                                VALUES (%s, %s)"""
            cursor.execute(insert_query, (result[i][0], embedding_output))
    
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
            create_vchord_indexes(conn) # Create the indexes for vchord
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