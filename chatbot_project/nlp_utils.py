import nltk
import re
import string
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob
import numpy as np

# Download required NLTK data
nltk_downloads = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'vader_lexicon']
for download in nltk_downloads:
    try:
        nltk.data.find(f'tokenizers/{download}')
    except LookupError:
        try:
            nltk.data.find(f'corpora/{download}')
        except LookupError:
            nltk.download(download, quiet=True)

class NLPProcessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def tokenize(self, text):
        """Tokenize text into words"""
        return word_tokenize(text.lower())
    
    def tokenize_sentences(self, text):
        """Tokenize text into sentences"""
        return sent_tokenize(text)
    
    def remove_punctuation(self, text):
        """Remove punctuation from text"""
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def remove_stopwords(self, tokens):
        """Remove stopwords from token list"""
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def stem_tokens(self, tokens):
        """Apply stemming to tokens"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize_tokens(self, tokens):
        """Apply lemmatization to tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_pipeline(self, text, remove_punct=True, remove_stops=True, 
                          apply_stemming=False, apply_lemmatization=True):
        """Complete preprocessing pipeline"""
        # Remove punctuation
        if remove_punct:
            text = self.remove_punctuation(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        if remove_stops:
            tokens = self.remove_stopwords(tokens)
        
        # Apply stemming or lemmatization
        if apply_stemming:
            tokens = self.stem_tokens(tokens)
        elif apply_lemmatization:
            tokens = self.lemmatize_tokens(tokens)
        
        return tokens
    
    def get_word_frequency(self, text, top_n=10):
        """Get word frequency from text"""
        tokens = self.preprocess_pipeline(text)
        freq_dist = Counter(tokens)
        return freq_dist.most_common(top_n)
    
    def extract_keywords(self, text, top_n=5):
        """Extract keywords using simple frequency-based approach"""
        tokens = self.preprocess_pipeline(text, remove_stops=True)
        
        # Filter out very short words
        tokens = [token for token in tokens if len(token) > 2]
        
        freq_dist = Counter(tokens)
        return [word for word, freq in freq_dist.most_common(top_n)]
    
    def calculate_text_similarity(self, text1, text2):
        """Calculate similarity between two texts using Jaccard similarity"""
        tokens1 = set(self.preprocess_pipeline(text1))
        tokens2 = set(self.preprocess_pipeline(text2))
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        if len(union) == 0:
            return 0.0
        
        return len(intersection) / len(union)
    
    def analyze_sentiment_detailed(self, text):
        """Detailed sentiment analysis using TextBlob"""
        blob = TextBlob(text)
        
        sentiment_info = {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity,
            'sentiment_label': self._get_sentiment_label(blob.sentiment.polarity),
            'confidence': abs(blob.sentiment.polarity)
        }
        
        return sentiment_info
    
    def _get_sentiment_label(self, polarity):
        """Convert polarity score to sentiment label"""
        if polarity > 0.1:
            return 'positive'
        elif polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def extract_named_entities(self, text):
        """Simple named entity extraction using TextBlob"""
        blob = TextBlob(text)
        
        # Extract noun phrases as potential entities
        entities = list(blob.noun_phrases)
        
        return entities
    
    def detect_language(self, text):
        """Detect language using TextBlob"""
        try:
            blob = TextBlob(text)
            return blob.detect_language()
        except:
            return 'unknown'
    
    def get_text_statistics(self, text):
        """Get comprehensive text statistics"""
        sentences = self.tokenize_sentences(text)
        words = self.tokenize(text)
        unique_words = set(words)
        
        stats = {
            'character_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'unique_word_count': len(unique_words),
            'avg_words_per_sentence': len(words) / len(sentences) if sentences else 0,
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'lexical_diversity': len(unique_words) / len(words) if words else 0
        }
        
        return stats

class IntentClassifier:
    """Simple intent classification using keyword matching"""
    
    def __init__(self):
        self.intent_keywords = {
            'greeting': ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening'],
            'goodbye': ['bye', 'goodbye', 'see you', 'farewell', 'take care', 'exit'],
            'question': ['what', 'how', 'why', 'when', 'where', 'who', 'which'],
            'help': ['help', 'assist', 'support', 'guide', 'explain'],
            'complaint': ['problem', 'issue', 'wrong', 'error', 'bug', 'broken'],
            'compliment': ['good', 'great', 'excellent', 'amazing', 'wonderful', 'perfect'],
            'information': ['tell me', 'information', 'details', 'about', 'explain'],
            'request': ['please', 'can you', 'could you', 'would you', 'do']
        }
        
        self.nlp_processor = NLPProcessor()
    
    def classify_intent(self, text):
        """Classify intent based on keyword matching"""
        tokens = self.nlp_processor.preprocess_pipeline(text, remove_stops=False)
        text_lower = text.lower()
        
        intent_scores = {}
        
        for intent, keywords in self.intent_keywords.items():
            score = 0
            for keyword in keywords:
                # Check for exact phrase match
                if keyword in text_lower:
                    score += 2
                # Check for individual word matches
                keyword_tokens = keyword.split()
                if all(token in tokens for token in keyword_tokens):
                    score += 1
            
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])
        
        return ('unknown', 0)
    
    def get_intent_confidence(self, text):
        """Get confidence score for intent classification"""
        intent, score = self.classify_intent(text)
        
        # Simple confidence calculation based on score and text length
        max_possible_score = len(self.intent_keywords.get(intent, []))
        confidence = min(score / max_possible_score, 1.0) if max_possible_score > 0 else 0
        
        return {
            'intent': intent,
            'confidence': confidence,
            'score': score
        }

if __name__ == "__main__":
    # Example usage
    nlp = NLPProcessor()
    classifier = IntentClassifier()
    
    # Test text
    sample_text = "Hello! I'm having trouble with my computer. Can you help me fix this problem?"
    
    print("=== NLP Analysis Demo ===")
    print(f"Text: {sample_text}")
    print()
    
    # Text preprocessing
    tokens = nlp.preprocess_pipeline(sample_text)
    print(f"Processed tokens: {tokens}")
    print()
    
    # Sentiment analysis
    sentiment = nlp.analyze_sentiment_detailed(sample_text)
    print(f"Sentiment analysis: {sentiment}")
    print()
    
    # Intent classification
    intent_info = classifier.get_intent_confidence(sample_text)
    print(f"Intent classification: {intent_info}")
    print()
    
    # Text statistics
    stats = nlp.get_text_statistics(sample_text)
    print(f"Text statistics: {stats}")
    print()
    
    # Keywords extraction
    keywords = nlp.extract_keywords(sample_text)
    print(f"Keywords: {keywords}")
