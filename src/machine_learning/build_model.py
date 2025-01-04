from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.etl.process_tic_tac_toe_results import TicTacToeResultsProcessor


class TicTacToeModelBuilder:
    MODEL_PATH = Path("src/machine_learning/tic_tac_model.z")

    @classmethod
    def build(cls):
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
