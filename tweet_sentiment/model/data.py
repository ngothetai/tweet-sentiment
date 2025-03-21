from pydantic import BaseModel
from typing import Literal


class Tweet(BaseModel):
    target: Literal[0, 2, 4] # type hint python
    tweet: str


if __name__ == "__main__":
    tweet = Tweet(target=2, tweet="This is a tweet")
    print(tweet.model_dump_json())