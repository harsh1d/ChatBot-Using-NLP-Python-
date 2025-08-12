import re
import random
import json
import pickle
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import Counter
import math

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.util import ngrams
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not available. Using basic text processing.")

class TFIDFVectorizer:
    """Simple TF-IDF vectorizer implementation"""
    
    def __init__(self):
        self.vocabulary = {}
        self.idf_values = {}
        self.documents = []
    
    def fit(self, documents: List[str]):
        """Fit the vectorizer on documents"""
        self.documents = documents
        
        # Build vocabulary
        all_words = set()
        for doc in documents:
            words = doc.lower().split()
            all_words.update(words)
        
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(all_words))}
        
        # Calculate IDF values
        doc_count = len(documents)
        for word in self.vocabulary:
            containing_docs = sum(1 for doc in documents if word in doc.lower())
            self.idf_values[word] = math.log(doc_count / (containing_docs + 1))
    
    def transform(self, documents: List[str]) -> List[List[float]]:
        """Transform documents to TF-IDF vectors"""
        vectors = []
        
        for doc in documents:
            words = doc.lower().split()
            word_count = Counter(words)
            doc_length = len(words)
            
            vector = [0.0] * len(self.vocabulary)
            
            for word, count in word_count.items():
                if word in self.vocabulary:
                    tf = count / doc_length
                    idf = self.idf_values[word]
                    tfidf = tf * idf
                    vector[self.vocabulary[word]] = tfidf
            
            vectors.append(vector)
        
        return vectors
    
    def fit_transform(self, documents: List[str]) -> List[List[float]]:
        """Fit and transform in one step"""
        self.fit(documents)
        return self.transform(documents)

class IntentClassifier:
    """Simple intent classifier using cosine similarity"""
    
    def __init__(self):
        self.vectorizer = TFIDFVectorizer()
        self.intent_vectors = {}
        self.training_data = {}
    
    def train(self, training_data: Dict[str, List[str]]):
        """Train the classifier with intent examples"""
        self.training_data = training_data
        
        # Prepare training documents
        all_examples = []
        intent_labels = []
        
        for intent, examples in training_data.items():
            for example in examples:
                all_examples.append(example)
                intent_labels.append(intent)
        
        # Vectorize examples
        vectors = self.vectorizer.fit_transform(all_examples)
        
        # Calculate average vector for each intent
        intent_vectors = {}
        for intent in training_data.keys():
            intent_examples = [vectors[i] for i, label in enumerate(intent_labels) if label == intent]
            if intent_examples:
                avg_vector = [sum(dim) / len(intent_examples) for dim in zip(*intent_examples)]
                intent_vectors[intent] = avg_vector
        
        self.intent_vectors = intent_vectors
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict intent for given text"""
        vector = self.vectorizer.transform([text])[0]
        
        best_intent = 'default'
        best_similarity = 0.0
        
        for intent, intent_vector in self.intent_vectors.items():
            similarity = self.cosine_similarity(vector, intent_vector)
            if similarity > best_similarity:
                best_similarity = similarity
                best_intent = intent
        
        return best_intent, best_similarity

class AdvancedChatbot:
    """Advanced chatbot with ML-based intent classification"""
    
    def __init__(self):
        self.stemmer = None
        self.stop_words = set()
        self.conversation_history = []
        self.context = {}
        self.user_preferences = {}
        
        # Initialize NLP components
        self.setup_nlp()
        
        # Initialize intent classifier
        self.intent_classifier = IntentClassifier()
        
        # Load training data and responses
        self.training_data, self.responses = self.load_training_data()
        
        # Train the classifier
        self.intent_classifier.train(self.training_data)
        
        # Initialize conversation context
        self.conversation_context = {
            'last_intent': None,
            'entity_memory': {},
            'conversation_flow': []
        }
    
    def setup_nlp(self):
        """Setup NLP components"""
        if NLTK_AVAILABLE:
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                self.stemmer = PorterStemmer()
                self.stop_words = set(stopwords.words('english'))
            except:
                pass
    
    def load_training_data(self) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """Load training data and responses"""
        training_data = {
            'greeting': [
                'hello', 'hi', 'hey', 'good morning', 'good afternoon',
                'greetings', 'how are you', 'what\'s up', 'howdy'
            ],
            'goodbye': [
                'bye', 'goodbye', 'see you later', 'farewell',
                'talk to you later', 'catch you later', 'until next time'
            ],
            'question_name': [
                'what is your name', 'who are you', 'what should I call you',
                'tell me your name', 'your name please'
            ],
            'question_help': [
                'help me', 'what can you do', 'how can you help',
                'what are your capabilities', 'help', 'assist me'
            ],
            'request_joke': [
                'tell me a joke', 'make me laugh', 'say something funny',
                'joke please', 'humor me', 'be funny'
            ],
            'question_time': [
                'what time is it', 'current time', 'time please',
                'tell me the time', 'what\'s the time'
            ],
            'compliment': [
                'you are great', 'you are awesome', 'you are helpful',
                'I like you', 'you are amazing', 'good job'
            ],
            'question_weather': [
                'how is the weather', 'weather report', 'is it raining',
                'temperature outside', 'weather forecast'
            ],
            'personal_info': [
                'tell me about yourself', 'what do you like',
                'your hobbies', 'personal information', 'about you'
            ],
            'math_question': [
                'calculate', 'what is', 'solve', 'math problem',
                'addition', 'subtraction', 'multiplication', 'division'
            ]
        }
        
        responses = {
            'greeting': [
                "Hello! I'm excited to chat with you today!",
                "Hi there! How can I brighten your day?",
                "Greetings! I'm here and ready to help!",
                "Hello! What interesting topic shall we explore today?"
            ],
            'goodbye': [
                "Goodbye! It was wonderful chatting with you!",
                "See you later! Hope to talk again soon!",
                "Farewell! Take care and have an amazing day!",
                "Until next time! Thanks for the great conversation!"
            ],
            'question_name': [
                "I'm AdvancedBot, your AI companion with enhanced capabilities!",
                "You can call me AdvancedBot. I'm here to help with various tasks!",
                "I'm AdvancedBot, equipped with machine learning for better conversations!"
            ],
            'question_help': [
                "I can help with conversations, answer questions, tell jokes, do simple math, and remember our chat context!",
                "My capabilities include natural language understanding, joke telling, basic calculations, and contextual responses!",
                "I'm designed to have engaging conversations, provide information, and adapt to your preferences!"
            ],
            'request_joke': [
                "Why don't scientists trust atoms? Because they make up everything! 😄",
                "I'm reading a book about anti-gravity. It's impossible to put down! 📚",
                "Why did the math book look so sad? Because it was full of problems! 📖",
                "What do you call a bear with no teeth? A gummy bear! 🐻",
                "Why don't eggs tell jokes? They'd crack each other up! 🥚"
            ],
            'question_time': [
                f"The current time is {datetime.now().strftime('%I:%M %p')} ⏰",
                f"Right now it's {datetime.now().strftime('%I:%M %p')}",
                f"The time is currently {datetime.now().strftime('%I:%M %p')}"
            ],
            'compliment': [
                "Thank you so much! Your kind words really motivate me! 😊",
                "That's incredibly sweet of you to say! I appreciate it!",
                "You're too kind! I'm just doing my best to help you!",
                "Aww, thank you! You're pretty awesome yourself! ✨"
            ],
            'question_weather': [
                "I don't have access to real-time weather data, but I hope it's beautiful where you are! 🌞",
                "I wish I could check the weather for you! Try a weather app for accurate information! 🌤️",
                "Weather updates aren't in my toolkit yet, but I hope you're enjoying pleasant conditions! ⛅"
            ],
            'personal_info': [
                "I'm an AI created to be helpful, harmless, and honest! I enjoy learning from our conversations! 🤖",
                "I'm passionate about helping people and learning new things through our chats! What about you?",
                "I love having meaningful conversations and helping solve problems! I'm always eager to learn!"
            ],
            'math_question': [
                "I can help with basic math! Try asking me something like 'what is 15 + 27?' 🔢",
                "Math questions are fun! I can handle addition, subtraction, multiplication, and division!",
                "Give me a math problem and I'll solve it for you! Keep it to basic operations for now! ➕"
            ],
            'default': [
                "That's interesting! Could you tell me more about that? 🤔",
                "I find that fascinating! What else would you like to explore?",
                "Hmm, that's thought-provoking! Care to elaborate?",
                "I'm intrigued! What's your perspective on that?",
                "That's a great point! What made you think of that?"
            ]
        }
        
        return training_data, responses
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text (basic implementation)"""
        entities = {
            'numbers': re.findall(r'\b\d+(?:\.\d+)?\b', text),
            'math_operations': re.findall(r'\b(plus|minus|times|divided by|add|subtract|multiply|divide)\b', text.lower()),
            'names': re.findall(r'\bmy name is (\w+)\b', text.lower())
        }
        return entities
    
    def handle_math_question(self, text: str) -> Optional[str]:
        """Handle basic math questions"""
        # Simple calculator for basic operations
        math_patterns = [
            (r'what\s+is\s+(\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)', lambda m: float(m.group(1)) + float(m.group(2))),
            (r'what\s+is\s+(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)', lambda m: float(m.group(1)) - float(m.group(2))),
            (r'what\s+is\s+(\d+(?:\.\d+)?)\s*\*\s*(\d+(?:\.\d+)?)', lambda m: float(m.group(1)) * float(m.group(2))),
            (r'what\s+is\s+(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)', lambda m: float(m.group(1)) / float(m.group(2)) if float(m.group(2)) != 0 else None),
        ]
        
        for pattern, operation in math_patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    result = operation(match)
                    if result is not None:
                        return f"The answer is {result}! 🧮"
                    else:
                        return "I can't divide by zero! Try a different problem! 🚫"
                except:
                    return "I had trouble with that calculation. Could you rephrase it? 🤔"
        
        return None
    
    def update_context(self, user_input: str, intent: str, entities: Dict):
        """Update conversation context"""
        self.conversation_context['last_intent'] = intent
        self.conversation_context['conversation_flow'].append(intent)
        
        # Remember user's name if mentioned
        if entities.get('names'):
            self.conversation_context['entity_memory']['user_name'] = entities['names'][0]
        
        # Keep only last 10 intents for context
        if len(self.conversation_context['conversation_flow']) > 10:
            self.conversation_context['conversation_flow'] = self.conversation_context['conversation_flow'][-10:]
    
    def generate_contextual_response(self, intent: str, user_input: str) -> str:
        """Generate response considering context"""
        # Handle math questions specially
        if intent == 'math_question':
            math_result = self.handle_math_question(user_input)
            if math_result:
                return math_result
        
        # Get base response
        responses = self.responses.get(intent, self.responses['default'])
        base_response = random.choice(responses)
        
        # Add contextual elements
        user_name = self.conversation_context['entity_memory'].get('user_name')
        if user_name and intent == 'greeting':
            base_response = f"Hello, {user_name.title()}! " + base_response
        
        # Add conversation flow awareness
        recent_intents = self.conversation_context['conversation_flow'][-3:]
        if recent_intents.count('compliment') >= 2:
            if intent != 'compliment':
                base_response += " You're really making my day with all the kind words! 😊"
        
        return base_response
    
    def generate_response(self, user_input: str) -> str:
        """Generate response using ML-based intent classification"""
        # Store conversation
        self.conversation_history.append({
            'timestamp': datetime.now(),
            'user': user_input,
            'bot': None
        })
        
        # Extract entities
        entities = self.extract_entities(user_input)
        
        # Classify intent
        intent, confidence = self.intent_classifier.predict(user_input)
        
        # Use rule-based fallback if confidence is too low
        if confidence < 0.1:
            intent = 'default'
        
        # Update context
        self.update_context(user_input, intent, entities)
        
        # Generate contextual response
        response = self.generate_contextual_response(intent, user_input)
        
        # Update conversation history
        self.conversation_history[-1]['bot'] = response
        
        return response
    
    def chat(self):
        """Main chat interface"""
        print("🤖 AdvancedBot: Hello! I'm your advanced AI chatbot with ML capabilities!")
        print("🧠 I can understand context, remember our conversation, and learn from our interactions.")
        print("💬 Try asking me questions, telling jokes, or doing math problems!")
        print("=" * 80)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Check for exit
                if re.search(r'\b(quit|exit|bye|goodbye)\b', user_input.lower()):
                    goodbye_response = random.choice(self.responses['goodbye'])
                    print(f"\n🤖 AdvancedBot: {goodbye_response}")
                    break
                
                # Generate and display response
                response = self.generate_response(user_input)
                print(f"\n🤖 AdvancedBot: {response}")
                
            except KeyboardInterrupt:
                print(f"\n\n🤖 AdvancedBot: {random.choice(self.responses['goodbye'])}")
                break
            except Exception as e:
                print(f"\n🤖 AdvancedBot: I encountered an error: {e}")
                print("Let's keep chatting though! 😊")
    
    def save_model(self, filename: str = "chatbot_model.pkl"):
        """Save the trained model"""
        model_data = {
            'intent_classifier': self.intent_classifier,
            'conversation_context': self.conversation_context,
            'user_preferences': self.user_preferences
        }
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model saved to {filename}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, filename: str = "chatbot_model.pkl"):
        """Load a trained model"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.intent_classifier = model_data.get('intent_classifier', self.intent_classifier)
            self.conversation_context = model_data.get('conversation_context', self.conversation_context)
            self.user_preferences = model_data.get('user_preferences', {})
            
            print(f"Model loaded from {filename}")
        except FileNotFoundError:
            print(f"Model file {filename} not found. Starting with fresh model.")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def save_conversation(self, filename: str = "advanced_conversation_log.json"):
        """Save conversation with enhanced metadata"""
        try:
            conversation_data = {
                'metadata': {
                    'total_messages': len(self.conversation_history),
                    'conversation_context': self.conversation_context,
                    'user_preferences': self.user_preferences,
                    'export_timestamp': datetime.now().isoformat()
                },
                'conversation': []
            }
            
            for entry in self.conversation_history:
                conversation_data['conversation'].append({
                    'timestamp': entry['timestamp'].isoformat(),
                    'user': entry['user'],
                    'bot': entry['bot']
                })
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            
            print(f"Enhanced conversation log saved to {filename}")
        except Exception as e:
            print(f"Error saving conversation: {e}")

def main():
    """Main function"""
    print("🚀 Initializing Advanced ChatBot...")
    print("🔄 Loading ML models and NLP components...")
    
    chatbot = AdvancedChatbot()
    
    # Try to load existing model
    chatbot.load_model()
    
    # Start chatting
    chatbot.chat()
    
    # Save options
    print("\n" + "="*50)
    save_conv = input("Save this conversation? (y/n): ").lower()
    if save_conv in ['y', 'yes']:
        chatbot.save_conversation()
    
    save_model = input("Save the updated model? (y/n): ").lower()
    if save_model in ['y', 'yes']:
        chatbot.save_model()
    
    print("Thanks for chatting! 👋")

if __name__ == "__main__":
    main()
