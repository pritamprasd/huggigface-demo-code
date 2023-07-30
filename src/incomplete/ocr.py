from transformers import pipeline
from datetime import datetime

def query():    
    start = datetime.now()
    document_qa = pipeline(model="impira/layoutlm-document-qa", framework="pt")    
    res = document_qa(
        image="http://im.rediff.com/getahead/2012/dec/04cheque.jpg",
        question="What is the name?",
    )    
    print(f"⏰: {(datetime.now() - start).microseconds // 1000} ms")
    print(res)
    return res

if __name__ == "__main__":
    query()