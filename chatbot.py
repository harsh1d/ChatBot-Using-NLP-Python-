import re
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import json
from datetime import datetime
from typing import List, Dict, Tuple

class SimpleChatbot:
    """A simple rule-based chatbot with basic NLP capabilities"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set()
        self.conversation_history = []
        
        # Download required NLTK data
        self.download_nltk_data()
        
        # Load patterns and responses
        self.patterns = self.load_patterns()
        
        # Initialize context variables
        self.context = {}
        
    def download_nltk_data(self):
        """Download necessary NLTK data"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            print(f"Warning: Could not download NLTK data: {e}")
            print("Some features may not work properly.")
    
    def load_patterns(self) -> Dict:
        """Load conversation patterns and responses"""
        return {
            'greeting': {
                'patterns': [
                    r'\b(hello|hi|hey|greetings|good morning|good afternoon|good evening)\b',
                    r'\bhow are you\b',
                    r'\bwhat\'s up\b'
                ],
                'responses': [
                    "Hello! How can I help you today?",
                    "Hi there! What's on your mind?",
                    "Greetings! I'm here to chat with you.",
                    "Hello! Nice to meet you. How are you doing?"
                ]
            },
            'goodbye': {
                'patterns': [
                    r'\b(bye|goodbye|see you|farewell|talk to you later|ttyl)\b',
                    r'\b(quit|exit|leave)\b'
                ],
                'responses': [
                    "Goodbye! Have a great day!",
                    "See you later! Take care!",
                    "Farewell! It was nice chatting with you.",
                    "Bye! Feel free to come back anytime."
                ]
            },
            'name': {
                'patterns': [
                    r'\bwhat\'s your name\b',
                    r'\bwho are you\b',
                    r'\byour name\b'
                ],
                'responses': [
                    "I'm ChatBot, your friendly AI assistant!",
                    "You can call me ChatBot. I'm here to help!",
                    "I'm ChatBot, nice to meet you!"
                ]
            },
            'help': {
                'patterns': [
                    r'\bhelp\b',
                    r'\bwhat can you do\b',
                    r'\bcapabilities\b',
                    r'\bcommands\b'
                ],
                'responses': [
                    "I can chat with you, answer questions, tell jokes, and help with basic information. What would you like to know?",
                    "I'm here to have a conversation! You can ask me about myself, request jokes, or just chat.",
                    "I can help with casual conversation, basic questions, and entertainment. Try asking me something!"
                ]
            },
            'joke': {
                'patterns': [
                    r'\btell.*joke\b',
                    r'\bjoke\b',
                    r'\bfunny\b',
                    r'\bmake me laugh\b'
                ],
                'responses': [
                    "Why don't scientists trust atoms? Because they make up everything!",
                    "I told my wife she was drawing her eyebrows too high. She looked surprised.",
                    "Why don't eggs tell jokes? They'd crack each other up!",
                    "What do you call a fake noodle? An impasta!",
                    "Why did the scarecrow win an award? He was outstanding in his field!"
                ]
            },
            'time': {
                'patterns': [
                    r'\bwhat time\b',
                    r'\bcurrent time\b',
                    r'\btime is it\b'
                ],
                'responses': [
                    f"The current time is {datetime.now().strftime('%I:%M %p')}",
                    f"It's {datetime.now().strftime('%I:%M %p')} right now.",
                    f"The time is {datetime.now().strftime('%I:%M %p')}."
                ]
            },
            'weather': {
                'patterns': [
                    r'\bweather\b',
                    r'\btemperature\b',
                    r'\bhot|cold|sunny|rainy\b'
                ],
                'responses': [
                    "I don't have access to real weather data, but I hope it's nice where you are!",
                    "I can't check the weather for you, but you could try a weather app or website!",
                    "Weather sounds interesting! I wish I could look outside for you."
                ]
            },
            'compliment': {
                'patterns': [
                    r'\byou\'re (great|awesome|cool|amazing|wonderful)\b',
                    r'\bi like you\b',
                    r'\byou\'re helpful\b'
                ],
                'responses': [
                    "Thank you so much! That's very kind of you to say.",
                    "I appreciate the compliment! You're pretty great yourself!",
                    "That means a lot to me, thank you!",
                    "You're too kind! I'm just doing my best to help."
                ]
            },
            'default': {
                'patterns': [],
                'responses': [
                    "That's interesting! Can you tell me more?",
                    "I'm not sure I understand completely, but I'm listening!",
                    "Hmm, that's something to think about. What else is on your mind?",
                    "I see! What would you like to talk about next?",
                    "That's a good point. Can you elaborate?"
                ]
            }
        }
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess the input text"""
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\?\!\.\,]', '', text)
        
        return text
    
    def tokenize_and_stem(self, text: str) -> List[str]:
        """Tokenize text and apply stemming"""
        try:
            tokens = word_tokenize(text)
            # Remove stopwords and apply stemming
            stemmed_tokens = [
                self.stemmer.stem(token) 
                for token in tokens 
                if token not in self.stop_words and token.isalnum()
            ]
            return stemmed_tokens
        except:
            # Fallback if NLTK fails
            tokens = text.split()
            return [token for token in tokens if token.isalnum()]
    
    def find_best_match(self, user_input: str) -> Tuple[str, float]:
        """Find the best matching pattern for user input"""
        preprocessed_input = self.preprocess_text(user_input)
        best_match = 'default'
        best_score = 0.0
        
        for intent, data in self.patterns.items():
            if intent == 'default':
                continue
                
            for pattern in data['patterns']:
                if re.search(pattern, preprocessed_input, re.IGNORECASE):
                    # Simple scoring based on pattern match
                    score = len(re.findall(pattern, preprocessed_input, re.IGNORECASE))
                    if score > best_score:
                        best_score = score
                        best_match = intent
        
        return best_match, best_score
    
    def generate_response(self, user_input: str) -> str:
        """Generate a response based on user input"""
        # Store the conversation
        self.conversation_history.append({
            'timestamp': datetime.now(),
            'user': user_input,
            'bot': None  # Will be filled after response generation
        })
        
        # Find the best matching intent
        intent, confidence = self.find_best_match(user_input)
        
        # Handle special cases
        if intent == 'time':
            response = f"The current time is {datetime.now().strftime('%I:%M %p')}"
        else:
            # Get a random response from the matched intent
            responses = self.patterns[intent]['responses']
            response = random.choice(responses)
        
        # Update conversation history
        self.conversation_history[-1]['bot'] = response
        
        return response
    
    def chat(self):
        """Main chat loop"""
        print("🤖 ChatBot: Hello! I'm your friendly chatbot. Type 'quit' to exit.")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Check for exit conditions
                if re.search(r'\b(quit|exit|bye|goodbye)\b', user_input.lower()):
                    print(f"\n🤖 ChatBot: {random.choice(self.patterns['goodbye']['responses'])}")
                    break
                
                # Generate and display response
                response = self.generate_response(user_input)
                print(f"\n🤖 ChatBot: {response}")
                
            except KeyboardInterrupt:
                print(f"\n\n🤖 ChatBot: {random.choice(self.patterns['goodbye']['responses'])}")
                break
            except Exception as e:
                print(f"\n🤖 ChatBot: Sorry, I encountered an error: {e}")
    
    def get_conversation_history(self) -> List[Dict]:
        """Return the conversation history"""
        return self.conversation_history
    
    def save_conversation(self, filename: str = "conversation_log.json"):
        """Save conversation history to a file"""
        try:
            conversation_data = []
            for entry in self.conversation_history:
                conversation_data.append({
                    'timestamp': entry['timestamp'].isoformat(),
                    'user': entry['user'],
                    'bot': entry['bot']
                })
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            
            print(f"Conversation saved to {filename}")
        except Exception as e:
            print(f"Error saving conversation: {e}")

def main():
    """Main function to run the chatbot"""
    print("🚀 Initializing ChatBot...")
    print("Loading NLP components...")
    
    chatbot = SimpleChatbot()
    chatbot.chat()
    
    # Ask if user wants to save conversation
    save_option = input("\nWould you like to save this conversation? (y/n): ").lower()
    if save_option in ['y', 'yes']:
        chatbot.save_conversation()

if __name__ == "__main__":
    main()
