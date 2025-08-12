import requests
import json
import re
import random
from urllib.parse import quote_plus
import time
from datetime import datetime
import nltk
from textblob import TextBlob

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class MultiSourceChatbot:
    """Advanced chatbot with multiple API sources for comprehensive knowledge"""
    
    def __init__(self):
        self.conversation_history = []
        self.search_cache = {}
        
        # API endpoints and configurations
        self.apis = {
            'wikipedia': 'https://en.wikipedia.org/api/rest_v1/page/summary/',
            'duckduckgo': 'https://api.duckduckgo.com/',
            'newsapi': 'https://newsapi.org/v2/everything',  # Requires API key
            'openweather': 'https://api.openweathermap.org/data/2.5/weather',  # Requires API key
        }
        
        # Response templates
        self.responses = {
            'greeting': [
                "Hello! I'm your intelligent assistant with access to real-time information from across the internet!",
                "Hi there! I can answer questions about anything using multiple reliable sources.",
                "Hey! Ask me about current events, weather, facts, definitions, and much more!",
                "Hello! I have access to comprehensive knowledge - what would you like to know?"
            ],
            'goodbye': [
                "Goodbye! It was great helping you today.",
                "See you later! Feel free to come back with more questions.",
                "Take care! I'm always here when you need information.",
                "Bye! Thanks for the interesting conversation!"
            ],
            'thanks': [
                "You're very welcome! I'm glad I could help.",
                "Happy to assist! That's what I'm here for.",
                "My pleasure! Feel free to ask anything else.",
                "Glad I could provide useful information!"
            ],
            'error': [
                "I encountered an issue while searching. Let me try a different approach.",
                "Sorry, I'm having trouble finding that information right now. Could you rephrase your question?",
                "There seems to be a temporary issue. Please try asking again.",
                "I'm having difficulty accessing that information at the moment."
            ]
        }
        
        # Knowledge categories for better responses
        self.categories = {
            'weather': ['weather', 'temperature', 'rain', 'sunny', 'cloudy', 'forecast', 'climate'],
            'news': ['news', 'latest', 'current', 'breaking', 'headlines', 'today', 'recent'],
            'science': ['science', 'research', 'study', 'discovery', 'experiment', 'theory'],
            'technology': ['technology', 'tech', 'computer', 'software', 'AI', 'internet', 'digital'],
            'history': ['history', 'historical', 'past', 'ancient', 'timeline', 'era', 'century'],
            'geography': ['country', 'city', 'location', 'place', 'capital', 'continent', 'geography'],
            'health': ['health', 'medical', 'medicine', 'disease', 'symptoms', 'treatment', 'doctor'],
            'education': ['learn', 'study', 'education', 'school', 'university', 'course', 'degree']
        }
    
    def detect_category(self, text):
        """Detect the category of the question"""
        text_lower = text.lower()
        
        for category, keywords in self.categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return 'general'
    
    def search_wikipedia(self, query):
        """Search Wikipedia for information"""
        try:
            encoded_query = quote_plus(query.replace(' ', '_'))
            url = f"{self.apis['wikipedia']}{encoded_query}"
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'extract' in data and data['extract']:
                    return {
                        'source': 'Wikipedia',
                        'title': data.get('title', ''),
                        'content': data.get('extract', ''),
                        'url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                        'type': 'encyclopedia'
                    }
        except Exception as e:
            print(f"Wikipedia search error: {e}")
        
        return None
    
    def search_duckduckgo(self, query):
        """Search DuckDuckGo instant answers"""
        try:
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = requests.get(self.apis['duckduckgo'], params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for instant answer
                if data.get('Abstract'):
                    return {
                        'source': 'DuckDuckGo',
                        'content': data['Abstract'],
                        'url': data.get('AbstractURL', ''),
                        'type': 'instant_answer'
                    }
                
                # Check for definition
                if data.get('Definition'):
                    return {
                        'source': 'DuckDuckGo',
                        'content': data['Definition'],
                        'url': data.get('DefinitionURL', ''),
                        'type': 'definition'
                    }
                
                # Check for answer
                if data.get('Answer'):
                    return {
                        'source': 'DuckDuckGo',
                        'content': data['Answer'],
                        'type': 'direct_answer'
                    }
        
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
        
        return None
    
    def get_current_time_date(self):
        """Get current time and date information"""
        now = datetime.now()
        
        return {
            'source': 'System',
            'content': f"Current date and time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}",
            'type': 'time_date'
        }
    
    def search_multiple_sources(self, query):
        """Search multiple sources and combine results"""
        results = []
        
        # Check cache first
        cache_key = query.lower().strip()
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
        
        print("🔍 Searching multiple sources...")
        
        # Time/date queries
        if any(word in query.lower() for word in ['time', 'date', 'today', 'current time']):
            results.append(self.get_current_time_date())
        
        # Search Wikipedia
        wiki_result = self.search_wikipedia(query)
        if wiki_result:
            results.append(wiki_result)
        
        # Search DuckDuckGo
        ddg_result = self.search_duckduckgo(query)
        if ddg_result:
            results.append(ddg_result)
        
        # Cache results
        if results:
            self.search_cache[cache_key] = results
        
        return results
    
    def format_response(self, results, query):
        """Format the search results into a coherent response"""
        if not results:
            return random.choice(self.responses['error'])
        
        # Single comprehensive result
        if len(results) == 1:
            result = results[0]
            response = f"📚 **{result.get('title', 'Information')}** ({result['source']})\n\n"
            response += result['content']
            
            if result.get('url'):
                response += f"\n\n🔗 Source: {result['url']}"
            
            return response
        
        # Multiple results - combine intelligently
        response = f"🤖 **Here's what I found about '{query}':**\n\n"
        
        for i, result in enumerate(results, 1):
            response += f"**{i}. {result['source']}**"
            if result.get('title'):
                response += f" - {result['title']}"
            response += ":\n"
            response += f"{result['content']}\n"
            
            if result.get('url'):
                response += f"🔗 {result['url']}\n"
            
            response += "\n"
        
        return response.strip()
    
    def detect_intent(self, text):
        """Enhanced intent detection"""
        text_lower = text.lower().strip()
        
        # Greeting patterns
        if re.search(r'\b(hello|hi|hey|good morning|good afternoon|good evening)\b', text_lower):
            return 'greeting'
        
        # Goodbye patterns
        if re.search(r'\b(bye|goodbye|see you|farewell|take care|exit|quit)\b', text_lower):
            return 'goodbye'
        
        # Thanks patterns
        if re.search(r'\b(thank you|thanks|appreciate)\b', text_lower):
            return 'thanks'
        
        # Question patterns
        question_patterns = [
            r'what is|what are|what\'s',
            r'how to|how do|how does|how can',
            r'why is|why do|why does|why are',
            r'when is|when do|when does|when did',
            r'where is|where do|where does|where can',
            r'who is|who are|who was|who were',
            r'which is|which are|which one',
            r'tell me about|explain',
            r'define|definition of'
        ]
        
        for pattern in question_patterns:
            if re.search(pattern, text_lower):
                return 'question'
        
        # If it ends with ? or contains question words, treat as question
        if text.strip().endswith('?') or any(word in text_lower.split() for word in 
                                           ['what', 'how', 'why', 'when', 'where', 'who', 'which']):
            return 'question'
        
        return 'question'  # Default to question for comprehensive answers
    
    def generate_response(self, user_input):
        """Generate intelligent response using multiple sources"""
        if not user_input.strip():
            return "I didn't catch that. Could you please ask me something?"
        
        # Store conversation
        self.conversation_history.append({
            'user_input': user_input,
            'timestamp': time.time(),
            'category': self.detect_category(user_input)
        })
        
        # Detect intent
        intent = self.detect_intent(user_input)
        
        # Handle basic intents
        if intent == 'greeting':
            return random.choice(self.responses['greeting'])
        elif intent == 'goodbye':
            return random.choice(self.responses['goodbye'])
        elif intent == 'thanks':
            return random.choice(self.responses['thanks'])
        
        # For questions, search multiple sources
        if intent == 'question':
            results = self.search_multiple_sources(user_input)
            return self.format_response(results, user_input)
        
        # Default: treat as question
        results = self.search_multiple_sources(user_input)
        return self.format_response(results, user_input)
    
    def get_conversation_summary(self):
        """Get a summary of the conversation"""
        if not self.conversation_history:
            return "No questions asked yet!"
        
        categories = {}
        for conv in self.conversation_history:
            cat = conv['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        summary = f"📊 **Conversation Summary**:\n"
        summary += f"Total questions: {len(self.conversation_history)}\n"
        summary += f"Topics discussed: {', '.join(categories.keys())}\n"
        summary += f"Most asked about: {max(categories.items(), key=lambda x: x[1])[0]}"
        
        return summary
    
    def chat(self):
        """Main chat interface"""
        print("🌟" + "="*70 + "🌟")
        print("🤖 UNIVERSAL KNOWLEDGE CHATBOT")
        print("🌍 Access to Real-time Information from Multiple Sources")
        print("🌟" + "="*70 + "🌟")
        print("\n💡 I can answer questions about ANYTHING using real-time web data!")
        print("💡 Ask about: Current events, science, technology, history, weather, etc.")
        print("💡 I search Wikipedia, DuckDuckGo, and other reliable sources.")
        print("💡 Type 'quit', 'exit', or 'bye' to end our conversation.\n")
        
        while True:
            try:
                user_input = input("🧑 You: ").strip()
                
                if not user_input:
                    continue
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print(f"\n🤖 Bot: {random.choice(self.responses['goodbye'])}")
                    
                    # Show conversation summary
                    if len(self.conversation_history) > 0:
                        print(f"\n{self.get_conversation_summary()}")
                    
                    break
                
                # Special commands
                if user_input.lower() == 'summary':
                    print(f"\n🤖 Bot: {self.get_conversation_summary()}\n")
                    continue
                
                # Generate and display response
                print("\n🤖 Bot:")
                response = self.generate_response(user_input)
                print(response)
                print("\n" + "-"*80 + "\n")
                
            except KeyboardInterrupt:
                print(f"\n\n🤖 Bot: {random.choice(self.responses['goodbye'])}")
                break
            except Exception as e:
                print(f"\n🤖 Bot: I encountered an unexpected error: {str(e)}")
                print("Please try asking your question differently.\n")

# Specialized chatbots for specific domains
class NewsChatbot(MultiSourceChatbot):
    """Specialized chatbot focused on news and current events"""
    
    def __init__(self):
        super().__init__()
        self.default_queries = [
            "latest world news",
            "breaking news today",
            "current events",
            "trending topics"
        ]
    
    def get_default_news(self):
        """Get default news when no specific query is provided"""
        query = random.choice(self.default_queries)
        return self.search_multiple_sources(query)

class EducationChatbot(MultiSourceChatbot):
    """Specialized chatbot for educational content"""
    
    def __init__(self):
        super().__init__()
        self.educational_sources = [
            "According to educational sources",
            "Based on academic information",
            "From educational content"
        ]
    
    def format_educational_response(self, results, query):
        """Format response with educational context"""
        response = self.format_response(results, query)
        
        if response and not any(phrase in response for phrase in self.educational_sources):
            response = f"📚 **Educational Information**:\n\n{response}"
        
        return response

if __name__ == "__main__":
    print("Initializing Universal Knowledge Chatbot...")
    
    # Choose chatbot type
    print("\nChoose your chatbot experience:")
    print("1. Universal Knowledge Chatbot (All topics)")
    print("2. News & Current Events Chatbot")
    print("3. Educational Content Chatbot")
    
    choice = input("Enter your choice (1, 2, or 3): ").strip()
    
    if choice == "2":
        bot = NewsChatbot()
        print("\n📰 News & Current Events Chatbot initialized!")
    elif choice == "3":
        bot = EducationChatbot()
        print("\n📚 Educational Content Chatbot initialized!")
    else:
        bot = MultiSourceChatbot()
        print("\n🌍 Universal Knowledge Chatbot initialized!")
    
    # Start chatting
    bot.chat()
