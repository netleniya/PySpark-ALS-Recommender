import dask.dataframe as ddf
import pandas as pd

from pathlib import Path


class DaskEDA:
    def __int__(self) -> None:
        ...

    @staticmethod
    def read_files(file_path, **kwargs):
        return ddf.read_csv(file_path, **kwargs).compute()


def main() -> None:
    root_path = Path().cwd()
    books_path = root_path / "dossier"
    rates_path = root_path / "archive/Ratings.csv"

    book_files = list(books_path.glob(pattern="*.csv"))

    dask_obj = DaskEDA()

    books_raw = dask_obj.read_files(
        book_files,
        usecols=["isbn", "isbn13", "title", "authors", "language", "publisher"],
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

    books_df.user_id = pd.Categorical(books_df.user_id)
    books_df["userId"] = books_df.user_id.cat.codes

    books_df.isbn13 = pd.Categorical(books_df.isbn13)
    books_df["bookId"] = books_df.isbn13.cat.codes

    books_df = books_df.rename(columns={"book_rating": "rating"})
    clean_df = books_df.loc[
        :, ["userId", "bookId", "isbn13", "title", "language", "rating"]
    ]
    clean_df.to_parquet("outputs/work_df", index=True)


if __name__ == "__main__":
    main()
