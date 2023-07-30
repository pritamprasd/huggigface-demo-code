from transformers import pipeline
from datetime import datetime
import argparse

def sentiment(msg):
    generator = pipeline("zero-shot-classification")
    start = datetime.now()
    res = generator(
        msg,
        candidate_labels=["education", "religion", "games"]
    )
    print(f"⏰: {(datetime.now() - start).microseconds // 1000} ms")
    print(res)
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('query', type=str, help="Sentence to classify")
    args = parser.parse_args()
    sentiment(args.query)