#!/usr/bin/env python3
"""
Setup script for the Chatbot Project
This script installs dependencies and sets up the environment
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
    """Install all required packages"""
    print("🚀 Installing Python packages...")
    
    required_packages = [
        "nltk==3.8.1",
        "scikit-learn==1.3.0", 
        "numpy==1.24.3",
        "pandas==2.0.3",
        "textblob==0.17.1"
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

def test_installation():
    """Test if all components work correctly"""
    print("\n🧪 Testing installation...")
    
    try:
        # Test imports
        import nltk
        import sklearn
        import numpy
        import pandas
        import textblob
        
        print("✅ All imports successful")
        
        # Test basic functionality
        from nlp_utils import NLPProcessor, IntentClassifier
        
        nlp = NLPProcessor()
        classifier = IntentClassifier()
        
        # Test text processing
        test_text = "Hello, this is a test!"
        tokens = nlp.tokenize(test_text)
        intent = classifier.classify_intent(test_text)
        
        print("✅ NLP components working correctly")
        print(f"   Sample tokens: {tokens[:3]}...")
        print(f"   Sample intent: {intent[0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("=" * 50)
    print("🤖 CHATBOT PROJECT SETUP")
    print("=" * 50)
    
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
    
    # Test installation
    if test_installation():
        print("\n" + "=" * 50)
        print("🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("\nYou can now run the chatbot:")
        print("  python chatbot.py")
        print("\nOr test the NLP utilities:")
        print("  python nlp_utils.py")
    else:
        print("\n❌ Setup completed with errors")
        print("Please check the error messages above")

if __name__ == "__main__":
    main()
