import re
import random
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from textblob import TextBlob

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class SimpleChatbot:
    def __init__(self):
        # Initialize knowledge base
        self.responses = {
            'greeting': [
                "Hello! How can I help you today?",
                "Hi there! What would you like to know?",
                "Hey! I'm here to assist you.",
                "Hello! Feel free to ask me anything."
            ],
            'goodbye': [
                "Goodbye! Have a great day!",
                "See you later!",
                "Take care! Feel free to come back anytime.",
                "Bye! It was nice talking with you."
            ],
            'thanks': [
                "You're welcome!",
                "Happy to help!",
                "No problem at all!",
                "Glad I could assist you!"
            ],
            'default': [
                "I'm not sure I understand. Could you rephrase that?",
                "That's interesting! Tell me more.",
                "I'm still learning. Could you explain differently?",
                "Hmm, let me think about that..."
            ]
        }
        
        # Pattern matching for basic intents
        self.patterns = {
            'greeting': r'hello|hi|hey|good morning|good afternoon|good evening',
            'goodbye': r'bye|goodbye|see you|farewell|take care',
            'thanks': r'thank you|thanks|appreciate',
            'name': r'what.*your.*name|who.*are.*you',
            'weather': r'weather|temperature|rain|sunny|cloudy',
            'time': r'what.*time|current.*time',
        }
        
        # Knowledge base for more complex responses
        self.knowledge_base = [
            "I am a simple chatbot created to demonstrate basic NLP concepts.",
            "Natural Language Processing (NLP) helps computers understand human language.",
            "Machine learning algorithms can be used to improve chatbot responses.",
            "Text preprocessing includes tokenization, stemming, and removing stop words.",
            "TF-IDF stands for Term Frequency-Inverse Document Frequency.",
            "Cosine similarity measures the similarity between two text vectors.",
            "Python is a great language for building NLP applications.",
            "NLTK is a popular library for natural language processing in Python.",
            "Chatbots can use rule-based or machine learning approaches.",
            "Sentiment analysis helps determine the emotional tone of text."
        ]
        
        # Initialize TF-IDF vectorizer for similarity matching
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
        self.knowledge_vectors = self.vectorizer.fit_transform(self.knowledge_base)
    
    def preprocess_text(self, text):
        """Clean and preprocess input text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using TextBlob"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return "positive"
        elif polarity < -0.1:
            return "negative"
        else:
            return "neutral"
    
    def detect_intent(self, text):
        """Detect user intent using pattern matching"""
        text_clean = self.preprocess_text(text)
        
        for intent, pattern in self.patterns.items():
            if re.search(pattern, text_clean, re.IGNORECASE):
                return intent
        
        return None
    
    def find_best_match(self, user_input):
        """Find the best matching response from knowledge base using TF-IDF"""
        # Transform user input to vector
        user_vector = self.vectorizer.transform([user_input])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(user_vector, self.knowledge_vectors)
        
        # Find the best match
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[0][best_match_idx]
        
        # Return match if similarity is above threshold
        if best_similarity > 0.1:
            return self.knowledge_base[best_match_idx]
        
        return None
    
    def generate_response(self, user_input):
        """Generate response based on user input"""
        # Detect intent first
        intent = self.detect_intent(user_input)
        
        # Handle specific intents
        if intent == 'greeting':
            return random.choice(self.responses['greeting'])
        elif intent == 'goodbye':
            return random.choice(self.responses['goodbye'])
        elif intent == 'thanks':
            return random.choice(self.responses['thanks'])
        elif intent == 'name':
            return "I'm a simple chatbot built with Python and basic NLP techniques!"
        elif intent == 'weather':
            return "I don't have access to weather data, but you can check a weather app or website!"
        elif intent == 'time':
            import datetime
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            return f"The current time is {current_time}"
        
        # Try to find a match in knowledge base
        knowledge_response = self.find_best_match(user_input)
        if knowledge_response:
            return knowledge_response
        
        # Analyze sentiment and respond accordingly
        sentiment = self.analyze_sentiment(user_input)
        if sentiment == "positive":
            return "That sounds great! " + random.choice(self.responses['default'])
        elif sentiment == "negative":
            return "I'm sorry to hear that. " + random.choice(self.responses['default'])
        
        # Default response
        return random.choice(self.responses['default'])
    
    def chat(self):
        """Main chat loop"""
        print("🤖 Chatbot: Hello! I'm a simple chatbot. Type 'quit' to exit.")
        print("=" * 50)
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("🤖 Chatbot:", random.choice(self.responses['goodbye']))
                break
            
            if not user_input:
                continue
            
            response = self.generate_response(user_input)
            print("🤖 Chatbot:", response)
            print("-" * 50)

# Advanced chatbot with learning capabilities
class LearningChatbot(SimpleChatbot):
    def __init__(self):
        super().__init__()
        self.conversation_history = []
        self.learned_responses = {}
    
    def learn_from_conversation(self, user_input, bot_response, user_feedback=None):
        """Learn from conversation history"""
        self.conversation_history.append({
            'user_input': user_input,
            'bot_response': bot_response,
            'feedback': user_feedback,
            'timestamp': pd.Timestamp.now()
        })
    
    def get_conversation_stats(self):
        """Get statistics about conversations"""
        if not self.conversation_history:
            return "No conversations yet!"
        
        df = pd.DataFrame(self.conversation_history)
        stats = {
            'total_conversations': len(df),
            'avg_response_length': df['bot_response'].str.len().mean(),
            'most_common_intents': self.get_common_patterns()
        }
        
        return stats
    
    def get_common_patterns(self):
        """Identify common patterns in user inputs"""
        patterns = {}
        for conv in self.conversation_history:
            intent = self.detect_intent(conv['user_input'])
            if intent:
                patterns[intent] = patterns.get(intent, 0) + 1
        
        return sorted(patterns.items(), key=lambda x: x[1], reverse=True)
    
    def generate_response(self, user_input):
        """Enhanced response generation with learning"""
        response = super().generate_response(user_input)
        
        # Store the conversation
        self.learn_from_conversation(user_input, response)
        
        return response

if __name__ == "__main__":
    # Create and run the chatbot
    print("Initializing chatbot...")
    
    # You can choose between simple or learning chatbot
    chatbot_type = input("Choose chatbot type (1 for Simple, 2 for Learning): ").strip()
    
    if chatbot_type == "2":
        bot = LearningChatbot()
        print("Learning Chatbot initialized!")
    else:
        bot = SimpleChatbot()
        print("Simple Chatbot initialized!")
    
    bot.chat()
