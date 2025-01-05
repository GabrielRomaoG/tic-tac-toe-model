from pathlib import Path
from numpy import int8
import pandas as pd
from imblearn.over_sampling import RandomOverSampler


class TicTacToeResultsProcessor:

    DATASET_CSV_PATH = Path("src/data/tic_tac_results.csv")
    PROCESSED_DATASET_CSV_PATH = Path("src/data/processed_tic_tac_results.csv")

    @classmethod
    def process(cls) -> pd.DataFrame:
        """
        Processes the tic-tac-toe results dataset by loading the data, balancing the class distribution,
        encoding categorical variables using one-hot encoding, and saving the processed dataset to a CSV file.

        Returns:
            pd.DataFrame: The processed and encoded DataFrame containing the balanced tic-tac-toe results.
        """

        results_df = pd.read_csv(cls.DATASET_CSV_PATH).astype("category")

        balanced_results_df = cls._balance(results_df)

        encoded_results_df = pd.get_dummies(
            data=balanced_results_df,
            columns=balanced_results_df.columns.drop("CLASS"),
            dtype=int8,
        )

        encoded_results_df.to_csv(cls.PROCESSED_DATASET_CSV_PATH, index=False)

        return encoded_results_df

    @staticmethod
    def _balance(results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Balances the class distribution in the tic-tac-toe results dataset.

        Args:
            results_df (pd.DataFrame): The DataFrame containing the tic-tac-toe results.

        Returns:
            pd.DataFrame: The DataFrame with the balanced class distribution.
        """
        ros = RandomOverSampler()
        X_resampled, y_resampled = ros.fit_resample(
            X=results_df.drop(columns=["CLASS"]), y=results_df["CLASS"]
        )

        X_resampled["CLASS"] = y_resampled

        return X_resampled
