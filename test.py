import unittest
from main import RAGAgent

# test_test.py

class TestRAGAgentSentimentAnalysis(unittest.TestCase):
    def setUp(self):
        self.agent = RAGAgent()

    def test_sentiment_analysis(self):
        user_query = "I am very happy with the service!"
        sentiment = self.agent.analyze_sentiment(user_query)
        self.assertIn(sentiment, ["positive", "negative", "neutral"])

if __name__ == '__main__':
    unittest.main()