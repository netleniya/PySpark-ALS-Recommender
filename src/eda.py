from pathlib import Path

import dask.dataframe as ddf
import pandas as pd


class DaskEDA:
    def __int__(self) -> None: ...

    @staticmethod
    def read_files(file_path, **kwargs):
        return ddf.read_csv(file_path, **kwargs).compute()


def clean_data() -> None:
    root_path = Path().cwd().parent
    books_path = root_path / "data/raw/dossier/"
    rates_path = root_path / "data/raw/Ratings.csv"

    book_files = list(books_path.glob(pattern="*.csv"))

    dask_obj = DaskEDA()

    books_raw = dask_obj.read_files(
        book_files,
        usecols=[
            "isbn",
            "isbn13",
            "title",
            "authors",
            "language",
            "publisher",
            "image",
        ],
        dtype={"isbn": "object"},
    )

    ratings = dask_obj.read_files(rates_path)

    books_raw = books_raw.dropna(subset=["publisher", "authors", "language", "title"])
    ratings.columns = ratings.columns.str.replace("-", "_").str.lower()

    ratings_books = ratings.query("book_rating !=0").merge(books_raw, on=["isbn"])
    ratings_books = ratings_books.loc[
        :,
        [
            "user_id",
            "isbn13",
            "title",
            "authors",
            "publisher",
            "language",
            "image",
            "book_rating",
        ],
    ]

    freq = ratings_books["title"].value_counts()
    freq_review = freq[freq > 10].index

    books_df = ratings_books[ratings_books["title"].isin(freq_review)]
    books_df.loc[books_df["language"] == "en_US", "language"] = "en"

    minor_lang = books_df.loc[
        (books_df["language"] == "ru") | (books_df["language"] == "hi")
    ].index
    books_df = books_df.drop(minor_lang, axis=0)

    usr_freq = books_df["user_id"].value_counts()
    freq_usr = usr_freq[usr_freq > 5].index

    books_df = books_df[books_df["user_id"].isin(freq_usr)]

    grouped = books_df.groupby("title")[["book_rating"]].agg(
        {"book_rating": [("num_ratings", "count")]}
    )
    grouped.columns = grouped.columns.droplevel()
    grouped = grouped.reset_index().sort_values(by="num_ratings", ascending=False)
    grouped = grouped.query("num_ratings >= 10")

    books_df = books_df[books_df["title"].isin(grouped["title"])]

    books_df.user_id = pd.Categorical(books_df.user_id)
    books_df["userId"] = books_df.user_id.cat.codes

    books_df.isbn13 = pd.Categorical(books_df.isbn13)
    books_df["bookId"] = books_df.isbn13.cat.codes

    books_df = books_df.rename(columns={"book_rating": "rating"})
    clean_df = books_df.loc[
        :, ["userId", "bookId", "isbn13", "title", "language", "image", "rating"]
    ]

    filepath = Path().cwd().parent.joinpath("data", "processed", "clean_df")
    clean_df.to_parquet(filepath, index=False)
    print("Cleaned data saved to disk")


clean_data()
