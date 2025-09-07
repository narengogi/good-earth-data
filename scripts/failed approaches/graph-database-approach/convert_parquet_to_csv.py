import pandas as pd
import os

base_dir = "/home/naren/Code/map-of-goodreads/data-crunching-final/data"

def convert_parquet_to_csv(parquet_file, csv_file):
    df = pd.read_parquet(parquet_file, columns=['user_id','book_id'])
    # df['title'] = [title.replace('"', '').replace("\\", "") for title in df['title']]
    # df['count'] = 0
    df.to_csv(csv_file, index=False)

def append_counts():
    for file in os.listdir(f"{base_dir}/behaviour"):
        print(f"Processing {file}")
        df_behavioral = pd.read_csv(f"{base_dir}/behaviour/{file}", index_col=None)
        print("number of rows in df_behavioral", len(df_behavioral))
        # update number of times book_id appears in df_behavioral
        #group by book_id and count the number of times it appears
        book_counts = df_behavioral.groupby('book_id').size()
        # update count in df
        with open(f"{base_dir}/temp/counts_{file}.csv", 'w') as f:
            f.write("book_id,title,count\n")
            for book_id, count in book_counts.items():
                f.write(f"{book_id},'',{count}\n")
    # reconcile data
    final_df = pd.read_csv(f"{base_dir}/books/merged.csv")
    for file in os.listdir(f"{base_dir}/temp"):
        df = pd.read_csv(f"{base_dir}/temp/{file}")
        final_df = pd.merge(final_df, df, on='book_id', how='outer', suffixes=('_1', '_2'))
        final_df['count_1'] = final_df['count_1'].fillna(0)
        final_df['count_2'] = final_df['count_2'].fillna(0)
        final_df['count'] = final_df['count_1'].astype(int) + final_df['count_2'].astype(int)
        final_df['title'] = final_df['title_1']
        final_df = final_df.drop(columns=['count_1', 'count_2', 'title_1', 'title_2'])
        # df_counts = pd.concat([df_counts, df])
        # df_counts = df_counts.groupby('book_id', as_index=False)['count'].sum()
    final_df.to_csv(f"{base_dir}/books/merged_final.csv", index=False)
        

def reset_book_counts():
    df = pd.read_csv(f"{base_dir}/books/merged.csv")
    df['count'] = 0
    df.to_csv(f"{base_dir}/books/merged.csv", index=False)

def merge_duplicates():
    df = pd.read_csv(f"{base_dir}/books/merged.csv")
    df = df.drop_duplicates(subset="book_id", keep="first")
    df.to_csv(f"{base_dir}/books/merged.csv", index=False)

def merge_behaviour_files():
    for file in os.listdir(f"{base_dir}/behaviour"):
        print(f"Processing {file}")
        df = pd.read_csv(f"{base_dir}/behaviour/{file}", index_col=None)
        print("number of rows in df", len(df))
        df.to_csv(f"{base_dir}/behaviour_merged.csv", index=False, mode='a')

# def merge_books_files():
#     for file in os.listdir(f"/home/naren/Code/map-of-goodreads/data/books"

def convert_csv_to_parquet(file_path):
    df = pd.read_csv(file_path, usecols=['book_id', 'count'])
    df.to_parquet(file_path.replace('.csv', '.parquet'))

def merge_books_parquets():
    df = pd.DataFrame()
    for file in os.listdir(f"/home/naren/Code/map-of-goodreads/data/books"):
        df = pd.concat([df, pd.read_parquet(f"/home/naren/Code/map-of-goodreads/data/books/{file}")])
    counts = pd.read_csv(f"/home/naren/Code/map-of-goodreads/data-crunching-final/data/books/merged_final.csv", usecols=['book_id', 'count'])
    #fix dtype of book_id
    counts['book_id'] = counts['book_id'].astype(str)
    df = pd.merge(df, counts, on='book_id', how='left')
    df.to_parquet(f"../data/books/merged.parquet")

def explore_books_parquet():
    df = pd.read_parquet(f"/home/naren/Code/map-of-goodreads/data/books/merged.parquet")
    print(df.head(10))
    print(df.info())
    print(df.describe())
    print(df.columns)
    print(df.dtypes)

if __name__ == "__main__":
    # for file in os.listdir(f"{base_dir}/behaviour"):
        # convert_parquet_to_csv(f"{base_dir}/behaviour/{file}", f"{base_dir}/behaviour/{file.replace('.parquet', '.csv')}")
    # convert_parquet_to_csv(f"{base_dir}/books/1.parquet", f"{base_dir}/books/1.csv")
    # convert_parquet_to_csv(f"{base_dir}/books/2.parquet", f"{base_dir}/books/2.csv")
    # reset_book_counts()
    # append_counts()
    # merge_behaviour_files()
    # merge_duplicates()
    # convert_csv_to_parquet(f"{base_dir}/books/merged_final.csv")
    merge_books_parquets()
    # explore_books_parquet()