import pandas as pd

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType
from pyspark.ml.recommendation import ALS, ALSModel




def main() -> None:
    spark = SparkSession.builder \
        .appName("Recommend") \
        .config("spark.sql.repl.eagerEval.enabled", True) \
        .config("spark.sql.repl.eagerEval.maxNumRows", 10) \
        .config("spark.driver.memory", "3g") \
        .getOrCreate()



if __name__ == "__main__":
    main()