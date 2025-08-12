import re
import random
import string
import nltk
import requests
import json
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from textblob import TextBlob
from urllib.parse import quote_plus
import time
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class WebSearcher:
    """Handles web searches and information retrieval"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.search_cache = {}
    
    def search_google(self, query, num_results=5):
        """Search Google and return results"""
        try:
            # Check cache first
            if query in self.search_cache:
                return self.search_cache[query]
            
            # Encode query for URL
            encoded_query = quote_plus(query)
            url = f"https://www.google.com/search?q={encoded_query}&num={num_results}"
            
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            
            # Extract search results
            for result in soup.find_all('div', class_='g'):
                title_elem = result.find('h3')
                link_elem = result.find('a')
                snippet_elem = result.find('span', class_=['aCOpRe', 'st'])
                
                if title_elem and link_elem:
                    title = title_elem.get_text()
                    link = link_elem.get('href', '')
                    snippet = snippet_elem.get_text() if snippet_elem else ""
                    
                    if link.startswith('/url?q='):
                        link = link.split('/url?q=')[1].split('&')[0]
                    
                    results.append({
                        'title': title,
                        'link': link,
                        'snippet': snippet
                    })
            
            # Cache results
            self.search_cache[query] = results
            return results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def get_wikipedia_summary(self, query):
        """Get Wikipedia summary for a topic"""
        try:
            # Wikipedia API endpoint
            url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
            encoded_query = quote_plus(query.replace(' ', '_'))
            
            response = requests.get(url + encoded_query, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'title': data.get('title', ''),
                    'summary': data.get('extract', ''),
                    'url': data.get('content_urls', {}).get('desktop', {}).get('page', '')
                }
        except Exception as e:
            print(f"Wikipedia error: {e}")
        
        return None
    
    def get_news_headlines(self, topic="latest news", num_articles=5):
        """Get latest news headlines"""
        try:
            # Using a simple news search
            query = f"{topic} news"
            results = self.search_google(query, num_articles)
            
            news_results = []
            for result in results:
                if any(news_site in result['link'].lower() for news_site in 
                      ['bbc', 'cnn', 'reuters', 'news', 'times', 'post']):
                    news_results.append(result)
            
            return news_results[:num_articles]
        except Exception as e:
            print(f"News error: {e}")
            return []

class KnowledgeExtractor:
    """Extracts and processes knowledge from search results"""
    
    def __init__(self):
        self.nlp_processor = self.setup_nlp()
    
    def setup_nlp(self):
        """Setup NLP components"""
        try:
            from nlp_utils import NLPProcessor
            return NLPProcessor()
        except ImportError:
            # Fallback basic NLP
            return None
    
    def extract_key_information(self, search_results):
        """Extract key information from search results"""
        if not search_results:
            return "I couldn't find any relevant information about that topic."
        
        # Combine all snippets
        combined_text = ""
        sources = []
        
        for result in search_results[:3]:  # Use top 3 results
            if result['snippet']:
                combined_text += result['snippet'] + ". "
                sources.append({
                    'title': result['title'],
                    'url': result['link']
                })
        
        if not combined_text.strip():
            return "I found some results but couldn't extract clear information."
        
        # Clean and summarize the text
        summary = self.summarize_text(combined_text)
        
        return {
            'answer': summary,
            'sources': sources
        }
    
    def summarize_text(self, text, max_sentences=3):
        """Create a summary of the text"""
        try:
            # Simple extractive summarization
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) <= max_sentences:
                return text.strip()
            
            # Score sentences by length and position
            scored_sentences = []
            for i, sentence in enumerate(sentences):
                # Prefer longer sentences and earlier positions
                score = len(sentence.split()) - (i * 0.1)
                scored_sentences.append((score, sentence))
            
            # Get top sentences
            scored_sentences.sort(reverse=True)
            top_sentences = [sent for _, sent in scored_sentences[:max_sentences]]
            
            # Reorder by original position
            summary_sentences = []
            for sentence in sentences:
                if sentence in top_sentences:
                    summary_sentences.append(sentence)
                if len(summary_sentences) >= max_sentences:
                    break
            
            return " ".join(summary_sentences)
            
        except Exception as e:
            # Fallback: return first few sentences
            sentences = text.split('.')[:max_sentences]
            return '. '.join(sentences) + '.'

class SmartChatbot:
    """Advanced chatbot with internet search capabilities"""
    
    def __init__(self):
        self.web_searcher = WebSearcher()
        self.knowledge_extractor = KnowledgeExtractor()
        self.conversation_history = []
        
        # Basic responses
        self.responses = {
            'greeting': [
                "Hello! I'm your smart assistant with access to real-time information. What would you like to know?",
                "Hi there! I can help you find answers to any question using the latest information from the web.",
                "Hey! I'm here to help with any questions you have. I can search the internet for the most current information.",
                "Hello! Ask me anything - I have access to comprehensive knowledge from across the internet."
            ],
            'goodbye': [
                "Goodbye! Feel free to ask me anything anytime.",
                "See you later! I'm always here when you need information.",
                "Take care! Come back whenever you have questions.",
                "Bye! It was great helping you today."
            ],
            'thanks': [
                "You're very welcome! Happy to help anytime.",
                "My pleasure! That's what I'm here for.",
                "Glad I could help! Feel free to ask more questions.",
                "You're welcome! I love helping people find information."
            ]
        }
        
        # Question patterns
        self.question_patterns = [
            r'what is|what are|what\'s',
            r'how to|how do|how does|how can',
            r'why is|why do|why does|why are',
            r'when is|when do|when does|when did',
            r'where is|where do|where does|where can',
            r'who is|who are|who was|who were',
            r'which is|which are|which one',
            r'tell me about|explain',
            r'define|definition of',
            r'latest|recent|current|news about'
        ]
    
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
        for pattern in self.question_patterns:
            if re.search(pattern, text_lower):
                return 'question'
        
        # If it ends with ? it's likely a question
        if text.strip().endswith('?'):
            return 'question'
        
        # If it contains question words, treat as question
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        if any(word in text_lower.split() for word in question_words):
            return 'question'
        
        return 'question'  # Default to treating as question for comprehensive answers
    
    def is_news_query(self, text):
        """Check if the query is asking for news"""
        news_keywords = ['news', 'latest', 'recent', 'current', 'today', 'headlines', 'breaking']
        return any(keyword in text.lower() for keyword in news_keywords)
    
    def search_and_respond(self, query):
        """Search for information and generate response"""
        try:
            print("🔍 Searching for information...")
            
            # Check if it's a news query
            if self.is_news_query(query):
                news_results = self.web_searcher.get_news_headlines(query)
                if news_results:
                    response = "Here are the latest news updates:\n\n"
                    for i, news in enumerate(news_results, 1):
                        response += f"{i}. **{news['title']}**\n"
                        if news['snippet']:
                            response += f"   {news['snippet']}\n"
                        response += f"   Source: {news['link']}\n\n"
                    return response
            
            # Try Wikipedia first for factual information
            wiki_info = self.web_searcher.get_wikipedia_summary(query)
            if wiki_info and wiki_info['summary']:
                response = f"📚 **{wiki_info['title']}**\n\n"
                response += f"{wiki_info['summary']}\n\n"
                response += f"🔗 Learn more: {wiki_info['url']}"
                return response
            
            # Search Google for comprehensive results
            search_results = self.web_searcher.search_google(query)
            
            if search_results:
                knowledge = self.knowledge_extractor.extract_key_information(search_results)
                
                if isinstance(knowledge, dict):
                    response = f"🤖 **Answer**: {knowledge['answer']}\n\n"
                    response += "📖 **Sources**:\n"
                    for i, source in enumerate(knowledge['sources'], 1):
                        response += f"{i}. {source['title']}\n   {source['url']}\n"
                    return response
                else:
                    return knowledge
            
            return "I couldn't find specific information about that topic. Could you try rephrasing your question?"
            
        except Exception as e:
            return f"I encountered an error while searching: {str(e)}. Please try again."
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of the text"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                return "positive"
            elif polarity < -0.1:
                return "negative"
            else:
                return "neutral"
        except:
            return "neutral"
    
    def generate_response(self, user_input):
        """Generate intelligent response"""
        if not user_input.strip():
            return "I didn't catch that. Could you please ask me something?"
        
        # Store conversation
        self.conversation_history.append({
            'user_input': user_input,
            'timestamp': time.time()
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
        
        # For questions, search for information
        if intent == 'question':
            return self.search_and_respond(user_input)
        
        # Default: treat as a question
        return self.search_and_respond(user_input)
    
    def get_conversation_stats(self):
        """Get conversation statistics"""
        if not self.conversation_history:
            return "No conversations yet!"
        
        return {
            'total_questions': len(self.conversation_history),
            'recent_topics': [conv['user_input'][:50] + '...' for conv in self.conversation_history[-5:]]
        }
    
    def chat(self):
        """Main chat interface"""
        print("🌟" + "="*60 + "🌟")
        print("🤖 SMART CHATBOT - Your AI Assistant with Internet Access")
        print("🌟" + "="*60 + "🌟")
        print("\n💡 I can answer ANY question using real-time information from the internet!")
        print("💡 Ask me about current events, facts, how-to guides, definitions, and more.")
        print("💡 Type 'quit', 'exit', or 'bye' to end the conversation.\n")
        
        while True:
            try:
                user_input = input("🧑 You: ").strip()
                
                if not user_input:
                    continue
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print(f"\n🤖 Bot: {random.choice(self.responses['goodbye'])}")
                    break
                
                # Generate and display response
                print("\n🤖 Bot:", end=" ")
                response = self.generate_response(user_input)
                print(response)
                print("\n" + "-"*80 + "\n")
                
            except KeyboardInterrupt:
                print(f"\n\n🤖 Bot: {random.choice(self.responses['goodbye'])}")
                break
            except Exception as e:
                print(f"\n🤖 Bot: I encountered an error: {str(e)}")
                print("Please try again with a different question.\n")

# Enhanced chatbot with additional features
class SuperSmartChatbot(SmartChatbot):
    """Even more advanced chatbot with additional capabilities"""
    
    def __init__(self):
        super().__init__()
        self.user_preferences = {}
        self.topic_memory = {}
    
    def remember_user_preference(self, preference_type, value):
        """Remember user preferences"""
        self.user_preferences[preference_type] = value
    
    def get_personalized_response(self, query, base_response):
        """Add personalization to responses"""
        # Add user's name if known
        if 'name' in self.user_preferences:
            name = self.user_preferences['name']
            return f"Hi {name}! {base_response}"
        
        return base_response
    
    def generate_follow_up_questions(self, topic):
        """Generate relevant follow-up questions"""
        follow_ups = [
            f"Would you like to know more about {topic}?",
            f"Are you interested in recent developments regarding {topic}?",
            f"Do you have any specific questions about {topic}?",
            f"Would you like me to find more detailed information about {topic}?"
        ]
        return random.choice(follow_ups)

if __name__ == "__main__":
    print("Initializing Smart Chatbot...")
    
    # Choose chatbot type
    print("\nChoose your chatbot experience:")
    print("1. Smart Chatbot (Internet-enabled)")
    print("2. Super Smart Chatbot (With memory & personalization)")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "2":
        bot = SuperSmartChatbot()
        print("\n🚀 Super Smart Chatbot initialized!")
        
        # Get user's name for personalization
        name = input("What's your name? (optional): ").strip()
        if name:
            bot.remember_user_preference('name', name)
    else:
        bot = SmartChatbot()
        print("\n🚀 Smart Chatbot initialized!")
    
    # Start chatting
    bot.chat()
