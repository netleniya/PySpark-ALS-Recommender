import os
import pandas as pd


user_recs = pd.read_parquet(os.path.join("outputs", "recommend_books"))
book_recs = pd.read_parquet(os.path.join("outputs", "recommend_readers"))

database = pd.read_parquet(os.path.join("outputs", "work_df")).reset_index(drop=True)


def recommend_books_to_readers(user_id: int) -> pd.DataFrame:
    df = database.merge(user_recs, on=["bookId"], how="left")
    print(f"Book recommendations for user with id {user_id}:")
    return df[df["targetId"] == user_id][["title", "isbn13", "language"]].drop_duplicates(keep='first')


def recommend_readers_for_book(isbn: int) -> pd.DataFrame:
    df = database[database["isbn13"] == isbn].merge(book_recs, on=["bookId"]).drop_duplicates(["targetId"])
    print("Recommend: ",df.title.drop_duplicates().to_string(index=False, header=False), "to the following users")
    return df[["targetId"]]


def main() -> None:
    print(recommend_books_to_readers(user_id=26))
    print(recommend_readers_for_book(isbn=9780020427858))


if __name__ == "__main__":
    main()
