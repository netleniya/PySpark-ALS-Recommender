from abc import ABC, abstractmethod
from pandas import DataFrame
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel


class Recommender(ABC):
    spark = (
        SparkSession.builder.appName("Group7 Recommender")
        .config("spark.driver.memory", "8g")
        .getOrCreate()
    )

    model = ALSModel.load("../models/recommender.model")

    def __init__(self, local_model: ALSModel, spark_session: SparkSession) -> None:
        self.model = local_model
        self.spark = spark_session

    @abstractmethod
    def recommend(self, num_recommendations: int) -> DataFrame: ...


class BookRecommender(Recommender):

    def __init__(self) -> None:
        super().__init__(local_model=self.model, spark_session=self.spark)

    def recommend(self, num_recommendations: int) -> DataFrame:
        user_recs = self.model.recommendForAllUsers(num_recommendations)
        user_recs.createTempView("ALS_recs_temp")

        query = """
        SELECT
            userId AS targetId,
            bookIds_and_ratings.bookId,
            bookIds_and_ratings.rating AS prediction
        FROM ALS_recs_temp
        LATERAL VIEW explode(recommendations) exploded_table
        AS bookIds_and_ratings
        """
        clean_recs = self.spark.sql(query)
        return clean_recs


class UserRecommender(Recommender):

    def __init__(self) -> None:
        super().__init__(local_model=self.model, spark_session=self.spark)

    def recommend(self, num_recommendations: int) -> DataFrame:
        book_recs = self.model.recommendForAllItems(num_recommendations)
        book_recs.createTempView("ALS_books_temp")

        query = """
        SELECT
            bookId AS bookId,
            bookIds_and_userIds.userId AS targetId,
            bookIds_and_userIds.rating AS predicted_rating
        FROM ALS_books_temp
        LATERAL VIEW explode(recommendations) exploded_table
        AS bookIds_and_userIds
        """
        clean_recs = self.spark.sql(query)
        return clean_recs.toPandas()
