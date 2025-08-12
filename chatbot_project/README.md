# 🤖 Build Your Own Chatbot (Python + NLP Basics)

A comprehensive project that demonstrates how to build a chatbot using Python and fundamental Natural Language Processing (NLP) techniques.

## 📋 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [NLP Concepts Covered](#nlp-concepts-covered)
- [Examples](#examples)
- [Extending the Chatbot](#extending-the-chatbot)
- [Troubleshooting](#troubleshooting)

## ✨ Features

### Chatbot Capabilities
- **Intent Recognition**: Detects user intentions using pattern matching
- **Sentiment Analysis**: Analyzes emotional tone of user messages
- **Knowledge Base**: Uses TF-IDF and cosine similarity for information retrieval
- **Context Awareness**: Maintains conversation flow with appropriate responses
- **Learning Mode**: Advanced version that learns from conversations

### NLP Techniques Demonstrated
- **Text Preprocessing**: Tokenization, stemming, lemmatization
- **Feature Extraction**: TF-IDF vectorization
- **Similarity Matching**: Cosine similarity for response selection
- **Named Entity Recognition**: Basic entity extraction
- **Text Statistics**: Comprehensive text analysis
- **Language Detection**: Automatic language identification

## 📁 Project Structure

```
chatbot_project/
│
├── chatbot.py          # Main chatbot implementation
├── nlp_utils.py        # NLP utilities and processing functions
├── setup.py           # Setup script for dependencies
├── requirements.txt   # Python package dependencies
└── README.md         # This documentation file
```

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Quick Setup
1. **Clone or download the project files**
2. **Run the setup script**:
   ```bash
   python setup.py
   ```

### Manual Installation
If you prefer to install manually:

1. **Install Python packages**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download NLTK data**:
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

## 💻 Usage

### Running the Chatbot
```bash
python chatbot.py
```

You'll be prompted to choose between:
- **Simple Chatbot** (1): Basic functionality
- **Learning Chatbot** (2): Advanced version with conversation tracking

### Testing NLP Utilities
```bash
python nlp_utils.py
```

This will run a demonstration of various NLP processing capabilities.

### Example Conversation
```
🤖 Chatbot: Hello! I'm a simple chatbot. Type 'quit' to exit.
==================================================
You: Hello there!
🤖 Chatbot: Hello! How can I help you today?
--------------------------------------------------
You: What is natural language processing?
🤖 Chatbot: Natural Language Processing (NLP) helps computers understand human language.
--------------------------------------------------
You: I'm having a problem
🤖 Chatbot: I'm sorry to hear that. I'm still learning. Could you explain differently?
--------------------------------------------------
You: Thanks for your help!
🤖 Chatbot: You're welcome!
--------------------------------------------------
```

## 🧠 NLP Concepts Covered

### 1. Text Preprocessing
- **Tokenization**: Breaking text into words/sentences
- **Normalization**: Converting to lowercase, removing punctuation
- **Stop Word Removal**: Filtering common words (the, and, is, etc.)
- **Stemming/Lemmatization**: Reducing words to base forms

### 2. Feature Extraction
- **TF-IDF**: Term Frequency-Inverse Document Frequency
- **Bag of Words**: Text representation as word counts
- **N-grams**: Sequences of N words for context

### 3. Similarity Matching
- **Cosine Similarity**: Measuring text similarity using vectors
- **Jaccard Similarity**: Set-based similarity measurement

### 4. Intent Classification
- **Pattern Matching**: Rule-based intent detection
- **Keyword Matching**: Identifying intents from keywords
- **Confidence Scoring**: Measuring classification certainty

### 5. Sentiment Analysis
- **Polarity Detection**: Positive/negative/neutral classification
- **Subjectivity Analysis**: Objective vs. subjective content
- **Emotion Recognition**: Basic emotional tone detection

## 🔧 Examples

### Basic Text Processing
```python
from nlp_utils import NLPProcessor

nlp = NLPProcessor()
text = "Hello! How are you doing today?"

# Tokenization
tokens = nlp.tokenize(text)
print(tokens)  # ['hello', '!', 'how', 'are', 'you', 'doing', 'today', '?']

# Preprocessing pipeline
processed = nlp.preprocess_pipeline(text)
print(processed)  # ['hello', 'today']

# Sentiment analysis
sentiment = nlp.analyze_sentiment_detailed(text)
print(sentiment)  # {'polarity': 0.0, 'subjectivity': 0.0, 'sentiment_label': 'neutral', 'confidence': 0.0}
```

### Intent Classification
```python
from nlp_utils import IntentClassifier

classifier = IntentClassifier()
result = classifier.get_intent_confidence("Can you help me with this problem?")
print(result)  # {'intent': 'help', 'confidence': 0.25, 'score': 1}
```

### Text Similarity
```python
from nlp_utils import NLPProcessor

nlp = NLPProcessor()
text1 = "I love programming in Python"
text2 = "Python programming is great"

similarity = nlp.calculate_text_similarity(text1, text2)
print(f"Similarity: {similarity:.2f}")  # Similarity: 0.67
```

## 🚀 Extending the Chatbot

### Adding New Intents
1. **Update the patterns dictionary** in `chatbot.py`:
   ```python
   self.patterns = {
       'greeting': r'hello|hi|hey',
       'new_intent': r'your_pattern_here',
       # ... existing patterns
   }
   ```

2. **Add response handling** in the `generate_response` method:
   ```python
   elif intent == 'new_intent':
       return "Your custom response here"
   ```

### Adding Knowledge Base Entries
Expand the `knowledge_base` list in the `SimpleChatbot` class:
```python
self.knowledge_base = [
    "Existing knowledge...",
    "Your new knowledge entry here",
    # ... more entries
]
```

### Implementing Machine Learning
For more advanced functionality, consider:
- **Neural Networks**: Use frameworks like TensorFlow or PyTorch
- **Transformer Models**: Integrate pre-trained models like BERT
- **Deep Learning**: Implement RNNs, LSTMs, or GRUs for better context understanding

### Database Integration
Connect to a database for persistent knowledge:
```python
import sqlite3

class DatabaseChatbot(SimpleChatbot):
    def __init__(self):
        super().__init__()
        self.db = sqlite3.connect('chatbot.db')
        self.load_knowledge_from_db()
    
    def load_knowledge_from_db(self):
        cursor = self.db.cursor()
        cursor.execute("SELECT content FROM knowledge_base")
        self.knowledge_base = [row[0] for row in cursor.fetchall()]
```

## 🐛 Troubleshooting

### Common Issues

**1. NLTK Data Not Found**
```bash
python -c "import nltk; nltk.download('all')"
```

**2. Package Installation Errors**
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

**3. Import Errors**
Make sure all files are in the same directory and Python path is correct.

**4. Performance Issues**
For large knowledge bases, consider:
- Using more efficient similarity algorithms
- Implementing caching for frequent queries
- Using database indexing

### Debug Mode
Add debug prints to understand the chatbot's decision process:
```python
def generate_response(self, user_input):
    print(f"DEBUG: Processing input: {user_input}")
    intent = self.detect_intent(user_input)
    print(f"DEBUG: Detected intent: {intent}")
    # ... rest of the method
```

## 📚 Learning Resources

### Books
- "Natural Language Processing with Python" by Steven Bird
- "Speech and Language Processing" by Jurafsky & Martin

### Online Courses
- Coursera: Natural Language Processing Specialization
- edX: Introduction to Natural Language Processing (Microsoft)

### Documentation
- [NLTK Documentation](https://www.nltk.org/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [TextBlob Documentation](https://textblob.readthedocs.io/)

## 🤝 Contributing

Feel free to extend this project! Some ideas:
- Add more sophisticated NLP techniques
- Implement web interface using Flask/Django
- Add voice recognition capabilities
- Create a GUI using tkinter or PyQt
- Integrate with messaging platforms (Discord, Slack, etc.)

## 📄 License

This project is for educational purposes. Feel free to modify and use it for learning NLP and chatbot development.

---

Happy coding! 🎉 Build amazing chatbots and explore the fascinating world of Natural Language Processing!
