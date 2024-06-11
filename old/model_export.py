import os
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = (
    SparkSession.builder.appName("Group7")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)


def generate_book_recommendations(model):
    user_recs = model.recommendForAllUsers(5)
    user_recs.createTempView("ALS_recs_temp")

    # noinspection SqlNoDataSourceInspection
    query = """
    SELECT
        userId AS targetId,
        bookIds_and_ratings.bookId,
        bookIds_and_ratings.rating AS prediction
    FROM ALS_recs_temp
    LATERAL VIEW explode(recommendations) exploded_table
    AS bookIds_and_ratings
    """
    clean_recs = spark.sql(query)
    return clean_recs.toPandas()


def generate_user_recommendations(model):
    book_recs = model.recommendForAllItems(5)
    book_recs.createTempView("ALS_books_temp")

    # noinspection SqlNoDataSourceInspection
    query = """
    SELECT
        bookId AS bookId,
        bookIds_and_userIds.userId AS targetId,
        bookIds_and_userIds.rating AS predicted_rating
    FROM ALS_books_temp
    LATERAL VIEW explode(recommendations) exploded_table
    AS bookIds_and_userIds
    """
    clean_recs = spark.sql(query)
    return clean_recs.toPandas()


def main():
    file_path = os.path.join("", "alsrecommend.model")
    users_rec = os.path.join("outputs", "recommend_books")
    books_rec = os.path.join("outputs", "recommend_readers")

    model = ALSModel.load(file_path)

    generate_book_recommendations(model=model).to_parquet(users_rec)
    generate_user_recommendations(model=model).to_parquet(books_rec)

    spark.stop()


if __name__ == "__main__":
    main()
