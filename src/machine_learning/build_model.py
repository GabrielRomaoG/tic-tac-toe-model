from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.etl.process_tic_tac_toe_results import TicTacToeResultsProcessor


class TicTacToeModelBuilder:
    MODEL_PATH = Path("resources/tic_tac_model.z")

    @classmethod
    def build(cls):
        """
        Builds the machine learning model using the processed tic-tac-toe dataset,
        trains it, and saves it to a file.

        The model is a logistic regression with elastic net penalty and saga solver.
        The hyperparameters are set to the best values found in the model selection process.
        """
        processed_tic_tac_df = pd.read_csv(
            TicTacToeResultsProcessor.PROCESSED_DATASET_CSV_PATH
        )

        X = processed_tic_tac_df.drop(columns=["CLASS"])
        y = processed_tic_tac_df["CLASS"]

        logreg = LogisticRegression(
            penalty="elasticnet", solver="saga", max_iter=10000, C=1.73, l1_ratio=1.0
        )

        logreg.fit(X, y)

        joblib.dump(logreg, cls.MODEL_PATH)
