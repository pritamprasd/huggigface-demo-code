from transformers import pipeline
from datetime import datetime
import argparse

def sentiment(msg):
    generator = pipeline("text-generation", model="distilgpt2")
    start = datetime.now()
    res = generator(
        msg,
        max_length=30,
        num_return_sequences=2
    )
    print(f"⏰: {(datetime.now() - start).microseconds // 1000} ms")
    print(res)
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('query', type=str, help="Sentence to Generate text")
    args = parser.parse_args()
    sentiment(args.query)