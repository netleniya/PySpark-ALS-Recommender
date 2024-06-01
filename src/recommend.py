from pandas import DataFrame
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession
from abc import ABC, abstractmethod

spark = (
    SparkSession.builder.appName("Book Recommender")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)


class Recommender(ABC):
    def __init__(self, database: DataFrame) -> None:
        self.database = database

    @abstractmethod
    def generate_recommendations(self, model) -> None: ...

    @abstractmethod
    def recommend(self, id: int) -> DataFrame: ...


class BookRecommender(Recommender):
    def __init__(self, database: DataFrame, model: ALSModel, num_recs: int) -> None:
        super().__init__(database)
        self.model = model
        self.user_recs = model.recommendForAllUsers(num_recs)

        self.user_recs.createTempView("ALS_recs_temp")
        query = """
        SELECT
            userId AS targetId,
            bookIds_and_ratings.bookId,
            bookIds_and_ratings.rating AS prediction
        FROM ALS_recs_temp
        LATERAL VIEW explode(recommendations) exploded_table
        AS bookIds_and_ratings
        """

    def recommend(self, id: int) -> DataFrame:
        df = self.database.merge(self.user_recs, on=["bookId"], how="left")
        print(f"Book recommendations for user with id {id}:")
        return df[df["targetId"] == id][
            ["title", "isbn13", "language", "image"]
        ].drop_duplicates(keep="first")


class ReaderRecommender(Recommender):
    def __init__(self, database: DataFrame) -> None:
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
