from transformers import pipeline
from datetime import datetime
import argparse

def sentiment(msg):
    generator = pipeline("sentiment-analysis")
    start = datetime.now()
    res = generator(msg)
    print(f"⏰: {(datetime.now() - start).microseconds // 1000} ms")
    print(res)
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('query', type=str, help="Sentence to find Sentiment")
    args = parser.parse_args()
    sentiment(args.query)