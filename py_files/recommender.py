from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.recommendation import ALSModel


def load_model() -> ALSModel:
    return ALSModel.load("alsrecommend.model")


def generate_user_recommendations(num_books: int) -> DataFrame:
    model = load_model()
    return model.recommendForAllUsers(num_books)


def main() -> None:
    spark = (
        SparkSession.builder.appName("Recommend")
        .config("spark.sql.repl.eagerEval.enabled", True)
        .config("spark.sql.repl.eagerEval.maxNumRows", 10)
        .config("spark.driver.memory", "3g")
        .getOrCreate()
    )

    user_recs = generate_user_recommendations(num_books=3)
    user_recs.createTempView("ALS_recs_temp")

    query = """
    SELECT
        userId AS target,
        bookIds_and_ratings.bookId,
        bookIds_and_ratings.rating AS prediction
    FROM ALS_recs_temp
    LATERAL VIEW explode(recommendations) exploded_table
    AS bookIds_and_ratings
    """
    clean_recs = spark.sql(query)
    clean_recs.show()


if __name__ == "__main__":
    main()
