import pandas as pd
from pathlib import Path


class DataLoader:
    """
    Loads and performs basic preprocessing on the delivery dataset.
    """

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def load_data(self) -> pd.DataFrame:
        """
        Load the CSV file.
        """

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.file_path}"
            )

        df = pd.read_csv(self.file_path)

        print("=" * 50)
        print("Dataset Loaded Successfully")
        print("=" * 50)
        print(f"Rows    : {df.shape[0]}")
        print(f"Columns : {df.shape[1]}")
        print()

        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic cleaning.
        """

        df = df.copy()

        # Remove duplicates
        df = df.drop_duplicates()

        # Remove missing values
        df = df.dropna()

        # Remove impossible ratings
        df = df[
            (df["Agent_Rating"] >= 0)
            & (df["Agent_Rating"] <= 5)
        ]

        # Remove impossible ages
        df = df[
            (df["Agent_Age"] >= 18)
            & (df["Agent_Age"] <= 65)
        ]

        df.reset_index(drop=True, inplace=True)

        print("=" * 50)
        print("Cleaning Complete")
        print("=" * 50)
        print(f"Remaining Rows : {len(df)}")
        print()

        return df


if __name__ == "__main__":

    loader = DataLoader("data/raw/delivery_data.csv")

    data = loader.load_data()

    cleaned = loader.clean_data(data)

    print(cleaned.head())