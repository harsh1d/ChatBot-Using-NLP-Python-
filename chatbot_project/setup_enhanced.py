#!/usr/bin/env python3
"""
Enhanced Setup script for the Smart Chatbot Project
This script installs all dependencies including web search capabilities
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def install_requirements():
    """Install all required packages for smart chatbot"""
    print("🚀 Installing Python packages for Smart Chatbot...")
    
    required_packages = [
        "nltk==3.8.1",
        "scikit-learn==1.3.0", 
        "numpy==1.24.3",
        "pandas==2.0.3",
        "textblob==0.17.1",
        "requests==2.31.0",
        "beautifulsoup4==4.12.2",
        "lxml==4.9.3"
    ]
    
    failed_packages = []
    
    for package in required_packages:
        if not install_package(package):
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n❌ Failed to install: {', '.join(failed_packages)}")
        print("Please install them manually using: pip install <package_name>")
        return False
    
    print("\n✅ All packages installed successfully!")
    return True

def download_nltk_data():
    """Download required NLTK data"""
    print("\n📚 Downloading NLTK data...")
    
    import nltk
    
    nltk_data = [
        'punkt',
        'stopwords', 
        'wordnet',
        'omw-1.4',
        'vader_lexicon'
    ]
    
    for data in nltk_data:
        try:
            nltk.download(data, quiet=True)
            print(f"✅ Downloaded {data}")
        except Exception as e:
            print(f"❌ Failed to download {data}: {e}")

def test_web_connectivity():
    """Test internet connectivity for web search features"""
    print("\n🌐 Testing web connectivity...")
    
    try:
        import requests
        
        # Test basic connectivity
        response = requests.get('https://httpbin.org/get', timeout=10)
        if response.status_code == 200:
            print("✅ Internet connectivity: OK")
        
        # Test Wikipedia API
        wiki_response = requests.get('https://en.wikipedia.org/api/rest_v1/page/summary/Python_programming_language', timeout=10)
        if wiki_response.status_code == 200:
            print("✅ Wikipedia API: OK")
        
        # Test DuckDuckGo API
        ddg_response = requests.get('https://api.duckduckgo.com/?q=test&format=json', timeout=10)
        if ddg_response.status_code == 200:
            print("✅ DuckDuckGo API: OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Web connectivity test failed: {e}")
        print("⚠️  The chatbot will work with limited functionality without internet access")
        return False

def test_installation():
    """Test if all components work correctly"""
    print("\n🧪 Testing chatbot components...")
    
    try:
        # Test basic imports
        import nltk
        import sklearn
        import numpy
        import pandas
        import textblob
        import requests
        import bs4
        
        print("✅ All imports successful")
        
        # Test NLP components
        from nlp_utils import NLPProcessor, IntentClassifier
        
        nlp = NLPProcessor()
        classifier = IntentClassifier()
        
        # Test text processing
        test_text = "What is artificial intelligence?"
        tokens = nlp.tokenize(test_text)
        intent = classifier.classify_intent(test_text)
        
        print("✅ NLP components working correctly")
        print(f"   Sample tokens: {tokens[:3]}...")
        print(f"   Sample intent: {intent[0]}")
        
        # Test web search components
        print("\n🔍 Testing web search capabilities...")
        
        try:
            from smart_chatbot import WebSearcher, KnowledgeExtractor
            
            searcher = WebSearcher()
            extractor = KnowledgeExtractor()
            
            # Test Wikipedia search
            wiki_result = searcher.get_wikipedia_summary("Python programming")
            if wiki_result:
                print("✅ Wikipedia search: Working")
            else:
                print("⚠️  Wikipedia search: Limited functionality")
            
            print("✅ Web search components initialized successfully")
            
        except Exception as e:
            print(f"⚠️  Web search test warning: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False

def create_demo_script():
    """Create a demo script to showcase chatbot capabilities"""
    demo_script = '''#!/usr/bin/env python3
"""
Smart Chatbot Demo Script
Run this to test different chatbot capabilities
"""

def demo_basic_chatbot():
    print("=== BASIC CHATBOT DEMO ===")
    from chatbot import SimpleChatbot
    
    bot = SimpleChatbot()
    
    test_queries = [
        "Hello!",
        "What is machine learning?",
        "Thank you!",
        "Goodbye"
    ]
    
    for query in test_queries:
        print(f"User: {query}")
        response = bot.generate_response(query)
        print(f"Bot: {response}")
        print("-" * 40)

def demo_smart_chatbot():
    print("\\n=== SMART CHATBOT DEMO ===")
    from smart_chatbot import SmartChatbot
    
    bot = SmartChatbot()
    
    test_queries = [
        "What is artificial intelligence?",
        "Tell me about Python programming",
        "What's the weather like?",
        "Latest news about technology"
    ]
    
    for query in test_queries:
        print(f"User: {query}")
        try:
            response = bot.generate_response(query)
            print(f"Bot: {response[:200]}...")
        except Exception as e:
            print(f"Bot: Error - {e}")
        print("-" * 40)

def demo_api_chatbot():
    print("\\n=== API CHATBOT DEMO ===")
    from api_chatbot import MultiSourceChatbot
    
    bot = MultiSourceChatbot()
    
    test_queries = [
        "What is quantum computing?",
        "Current time and date",
        "Define blockchain"
    ]
    
    for query in test_queries:
        print(f"User: {query}")
        try:
            response = bot.generate_response(query)
            print(f"Bot: {response[:200]}...")
        except Exception as e:
            print(f"Bot: Error - {e}")
        print("-" * 40)

if __name__ == "__main__":
    print("🤖 SMART CHATBOT DEMONSTRATION")
    print("=" * 50)
    
    try:
        demo_basic_chatbot()
        demo_smart_chatbot()
        demo_api_chatbot()
        
        print("\\n✅ All demos completed successfully!")
        print("\\nTo run the chatbots interactively:")
        print("  python chatbot.py          # Basic chatbot")
        print("  python smart_chatbot.py    # Smart chatbot with web search")
        print("  python api_chatbot.py      # Multi-source API chatbot")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
'''
    
    try:
        with open('demo_chatbots.py', 'w', encoding='utf-8') as f:
            f.write(demo_script)
        print("✅ Created demo script: demo_chatbots.py")
        return True
    except Exception as e:
        print(f"❌ Failed to create demo script: {e}")
        return False

def main():
    """Main setup function"""
    print("=" * 60)
    print("🤖 SMART CHATBOT PROJECT SETUP")
    print("🌍 With Real-time Internet Knowledge Access")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Setup failed during package installation")
        sys.exit(1)
    
    # Download NLTK data
    download_nltk_data()
    
    # Test web connectivity
    web_ok = test_web_connectivity()
    
    # Test installation
    if test_installation():
        print("\n" + "=" * 60)
        print("🎉 SMART CHATBOT SETUP COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        if web_ok:
            print("\\n🌟 Your chatbot has full internet access capabilities!")
        else:
            print("\\n⚠️  Limited internet access - some features may be restricted")
        
        print("\\n🚀 Available Chatbots:")
        print("  python chatbot.py          # Basic NLP chatbot")
        print("  python smart_chatbot.py    # Internet-enabled smart chatbot")
        print("  python api_chatbot.py      # Multi-source knowledge chatbot")
        print("\\n🧪 Test all features:")
        print("  python demo_chatbots.py    # Run automated demos")
        
        # Create demo script
        create_demo_script()
        
        print("\n💡 Tips:")
        print("  - Ask about current events, facts, definitions")
        print("  - Try questions like 'What is...', 'How to...', 'Tell me about...'")
        print("  - The smart chatbots search multiple sources for accurate answers")
        
    else:
        print("\n❌ Setup completed with errors")
        print("Please check the error messages above")

if __name__ == "__main__":
    main()
