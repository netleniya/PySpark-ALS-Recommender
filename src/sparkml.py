import os
import shutil

from pathlib import Path

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

spark = (
    SparkSession.builder.appName("Book Recommender")
    .config("spark.driver.memory", "8g")
    .getOrCreate()
)


def create_dataframe() -> DataFrame:

    filepath = Path().cwd().joinpath("data", "processed", "clean_df")
    file = os.path.realpath(filename=filepath)
    spark_df = spark.read.parquet(file)
    colMap = {
        "userId": col("userId").cast("int"),
        "bookId": col("bookId").cast("int"),
        "rating": col("rating").cast("int"),
    }
    spark_df = spark_df.withColumns(colMap)

    ratings_df = spark_df.select("userId", "bookId", "rating")
    return ratings_df


def create_model(evaluator) -> CrossValidator:

    als = ALS(
        userCol="userId",
        itemCol="bookId",
        ratingCol="rating",
        nonnegative=True,
        coldStartStrategy="drop",
        seed=42,
    )

    param_grid = (
        ParamGridBuilder()
        .addGrid(als.maxIter, [50])
        .addGrid(als.rank, [10, 50, 100])
        .addGrid(als.regParam, [0.01, 0.1, 1.0])
        .build()
    )

    cv = CrossValidator(
        estimator=als,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=5,
        parallelism=3,
    )

    return cv


def main() -> None:

    book_ratings = create_dataframe()
    (train, test) = book_ratings.randomSplit([0.8, 0.2], seed=42)

    evaluator = RegressionEvaluator(
        metricName="rmse", labelCol="rating", predictionCol="prediction"
    )
    model = create_model(evaluator=evaluator).fit(train)
    best_model = model.bestModel

    model_path = "../models/recommender.model"

    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    best_model.save(model_path)

    spark.stop()


if __name__ == "__main__":
    main()
