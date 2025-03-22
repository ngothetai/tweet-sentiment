import pandas as pd
from typing import List
from tweet_sentiment.model.data import Tweet
from dataclasses import dataclass
from loguru import logger


class DataLoader:
    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
        self.data = None
        # Load data
        self.__preprocess_data()

    def __preprocess_data(self) -> None:
        try:
            self.data = pd.read_csv(
                self.data_path,
                encoding="latin-1",
                names=["target", "ids", "date", "flag", "user", "text"],
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        except pd.errors.EmptyDataError:
            raise pd.errors.EmptyDataError(f"Data file is empty at {self.data_path}")
        except pd.errors.ParserError:
            raise pd.errors.ParserError(
                f"Data file is not a valid CSV at {self.data_path}"
            )
        except Exception as e:
            raise Exception(f"An error occurred while loading data: {str(e)}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i) -> List[Tweet]:
        res = []
        logger.info(f"Getting item {i}")
        for target, tweet in zip(
            pd.Series(self.data.iloc[i]["target"]).values,
            pd.Series(self.data.iloc[i]["text"]).values,
        ):
            res.append(Tweet(target=target, tweet=tweet))
        return res

    @classmethod
    def split(
        cls, data: "DataLoader", train_size: float = 0.8
    ) -> Tuple["DataLoader", "DataLoader"]:
        index = int(len(data) * train_size)
        return (data[:index], data[index:])


# Speed up crawl -> multiprocessing, multi-threading, asyncio
# Storage -> x csv -> database (NoSQL, SQL)
# 