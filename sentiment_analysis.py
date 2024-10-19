from main import RAGAgent

def analyze_chat_sentiment(chat_content):
    agent = RAGAgent()
    sentiment = agent.analyze_sentiment(chat_content)
    print(f"Chat content: {chat_content}")
    print(f"Sentiment: {sentiment}")
    print(f"Recommended lawyer category: {sentiment}")

# Test the function with a sample chat content
sample_chat = "I want a criminal lawyer assistance!"
analyze_chat_sentiment(sample_chat)
