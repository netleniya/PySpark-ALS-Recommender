from pandas import DataFrame
from abc import ABC, abstractmethod


class Recommender(ABC):
    def __init__(self, database: DataFrame) -> None:
        self.database = database

    @abstractmethod
    def generate_recommendations(self, model) -> None: ...

    @abstractmethod
    def recommend(self, id: int) -> DataFrame: ...


class BookRecommender(Recommender):
    def __init__(self, database: DataFrame, user_recs) -> None:
        super().__init__(database)
        self.user_recs = user_recs

    def recommend(self, id: int) -> DataFrame:
        df = self.database.merge(self.user_recs, on=["bookId"], how="left")
        print(f"Book recommendations for user with id {id}:")
        return df[df["targetId"] == id][
            ["title", "isbn13", "language", "image"]
        ].drop_duplicates(keep="first")


class ReaderRecommender(Recommender):
    def __init__(self, database: DataFrame, book_recs) -> None:
        super().__init__(database)
        self.book_recs = book_recs

    def recommend(self, isbn: int) -> DataFrame:
        df = (
            self.database[self.database["isbn13"] == isbn]
            .merge(self.book_recs, on=["bookId"])
            .drop_duplicates(["targetId"])
        )
        print(
            "Recommend: ",
            df.title.drop_duplicates().to_string(index=False, header=False),  # type: ignore
            "to the following users",
        )
        return df[["targetId"]]
