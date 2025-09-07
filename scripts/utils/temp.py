import pandas as pd; 

pd.read_parquet("/home/naren/Code/map-of-goodreads/final/books.parquet",  columns=["book_id", "title", "author_names", "author_ids", "publisher", "publication_year", "average_rating", "description"]).to_csv("books.csv", index=False)
