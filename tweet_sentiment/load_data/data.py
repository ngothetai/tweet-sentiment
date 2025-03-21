import pandas as pd
from tweet_sentiment.model.data import Tweet


class DataLoader:
    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
        self.data = None
        # Load data
        self.__preprocess_data()

    def __preprocess_data(self) -> None:
        try:
            self.data = pd.read_csv(self.data_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {data_path}")
        except pd.errors.EmptyDataError:
            raise pd.errors.EmptyDataError(f"Data file is empty at {data_path}")
        except pd.errors.ParserError:
            raise pd.errors.ParserError(f"Data file is not a valid CSV at {data_path}")
        except Exception as e:
            raise Exception(f"An error occurred while loading data: {str(e)}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> Tweet:
        return Tweet(
            target=self.data.iloc[i]["target"], tweet=self.data.iloc[i]["text"]
        )
