from prep_data import prep_book_data, update_collection_to_database



if __name__ == "__main__":
   scv_file_path = r'TraceTalk\vector-db-persist-directory\book dada\book data.csv'
   prep_book_data(scv_file_path)
   update_collection_to_database(scv_file_path)