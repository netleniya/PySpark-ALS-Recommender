import pandas as pd

from recommend import BookRecommender, UserRecommender
from generate import BooksDataFrameGenerator
from pathlib import Path


def main() -> None:

    book_libray = Path().cwd().parent.joinpath("data", "processed", "clean_df")
    book_libray = pd.read_parquet(book_libray)

    user_recs = UserRecommender().recommend(num_recommendations=10)

    book_gen = BooksDataFrameGenerator(database=book_libray, user_recs=user_recs)

    print(book_gen.generate_dataframe(user_id=5291))


if __name__ == "__main__":
    main()
