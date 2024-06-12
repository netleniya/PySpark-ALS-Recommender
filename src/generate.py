from pandas import DataFrame
from abc import ABC, abstractmethod


class DataFrameGenerator(ABC):
    def __init__(self, database: DataFrame) -> None:
        self.database = database

    @abstractmethod
    def generate_dataframe(self, field_id: int) -> DataFrame: ...


class GenerateBookList(DataFrameGenerator):
    def __init__(self, database: DataFrame, user_recs) -> None:
        super().__init__(database)
        self.user_recs = user_recs

    def generate_dataframe(self, user_id: int) -> DataFrame:
        df = self.database.merge(self.user_recs, on=["bookId"], how="left")
        # print(f"Book recommendations for user with id {user_id}:")
        return df[df["targetId"] == user_id][
            ["title", "isbn13", "language", "image"]
        ].drop_duplicates(keep="first")


class GenerateUsersList(DataFrameGenerator, ABC):
    def __init__(self, database: DataFrame, book_recs) -> None:
        super().__init__(database)
        self.book_recs = book_recs

    def generate_dataframe(self, book_id: int) -> DataFrame:
        df = (
            self.database[self.database["bookId"] == book_id]
            .merge(self.book_recs, on=["bookId"])
            .drop_duplicates(["targetId"])
        )
        # print(
        #     "Recommend: ",
        #     df.title.drop_duplicates().to_string(index=False, header=False),  # type: ignore
        #     "to the following users",
        # )
        return df[["targetId"]]
