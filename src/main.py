import pandas as pd
import gradio as gr

from recommend import BookRecommender, UserRecommender
from generate import GenerateBookList, GenerateUsersList
from pathlib import Path


def load_data() -> pd.DataFrame:
    biblio = Path(__file__).parents[1].joinpath("data", "processed", "clean_df")
    biblio = pd.read_parquet(biblio)
    return biblio


book_libray = load_data()


def get_book_recommendations(user_id, num_recs) -> pd.DataFrame:

    try:
        rec_obj = BookRecommender().recommend(num_recommendations=num_recs)
        gen_obj = GenerateBookList(book_libray, rec_obj).generate_dataframe(
            user_id=user_id
        )
        return gen_obj
    except ValueError as e:
        print(f"No recommendations for {user_id} or User not found")


def get_user_recommendations(book_id, num_recs) -> pd.DataFrame:

    try:
        rec_obj = UserRecommender().recommend(num_recommendations=num_recs)
        gen_obj = GenerateUsersList(book_libray, rec_obj).generate_dataframe(
            book_id=book_id
        )
        return gen_obj
    except ValueError as e:
        print(f"No recommendations for {book_id} or Book not found")


def get_book_details(book_id) -> pd.DataFrame:
    try:
        return book_libray[book_libray["bookId"] == book_id][
            ["isbn13", "title", "language", "image"]
        ].drop_duplicates(keep="first")
    except ValueError as e:
        print(f"No recommendations for {book_id} or Book not found")


def main() -> None:

    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # <h1 style="text-align: center;">Book Recommender in PySpark </h1>

            Welcome to Collaborative filtering Book Recommender. 

            You can use the two tabs to get recommendations for a user or a book.
            - Use the `Book Recommender` tab to generate a list recommendations for a user.
            - Use the `User Recommender` tab to generate a list of users that may be interested in a book (NB: This is buggy and slow ATM).
            """
        )

        with gr.Tab("Book Recommender"):
            gr.Markdown("Enter the user ID and number of recommendations")
            usr_inp = gr.Number(label="User ID")
            num_inp = gr.Slider(
                label="Number of recommendations",
                minimum=1,
                maximum=10,
                step=1,
                interactive=True,
            )
            usr_df = gr.DataFrame(
                label="Recommended Books",
                headers=["Book ID", "Title", "Language"],
                col_count=(3, "fixed"),
            )
            usr_btn = gr.Button("Recommend")

        with gr.Tab("User Recommender"):
            gr.Markdown(
                "Enter the Book ID and number of users to recommend the book to"
            )
            book_inp = gr.Number(label="Book ID")
            num_recs = gr.Slider(
                label="Number of recommendations",
                minimum=1,
                maximum=10,
                step=1,
                interactive=True,
            )
            book_details = gr.DataFrame(
                label="Book Details",
                headers=["ISBN", "Title", "Language"],
                col_count=(3, "fixed"),
            )
            details_btn = gr.Button("Get Book Details")
            book_out = gr.DataFrame(
                label="Recommended Users",
                headers=["User ID"],
                col_count=(1, "fixed"),
            )
            book_btn = gr.Button("Recommend")

        usr_btn.click(
            get_book_recommendations, inputs=[usr_inp, num_inp], outputs=usr_df
        )
        details_btn.click(get_book_details, inputs=[book_inp], outputs=book_details)

        book_btn.click(
            get_user_recommendations,
            inputs=[book_inp, num_recs],
            outputs=[book_out],
        )

    demo.launch()


if __name__ == "__main__":
    main()
