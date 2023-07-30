from transformers import pipeline, Conversation
from datetime import datetime

def query(chatbot, msg):    
    start = datetime.now()
    conversation = Conversation(msg)
    conversation = chatbot(conversation)
    conversation.generated_responses
    print(f"⏰: {(datetime.now() - start).microseconds // 1000} ms")
    print(conversation.generated_responses[-1])
    return conversation.generated_responses

if __name__ == "__main__":
    chatbot = pipeline(model="microsoft/DialoGPT-medium", framework="pt")
    while True:
        text = input().strip()
        query(chatbot, text)