# test_setup.py
import sys
print(f"Python version: {sys.version}")

# Test core dependencies
try:
    import pandas as pd
    print("✅ pandas imported successfully")
except ImportError as e:
    print(f"❌ pandas import failed: {e}")

try:
    import numpy as np
    print("✅ numpy imported successfully")
except ImportError as e:
    print(f"❌ numpy import failed: {e}")

try:
    import torch
    print(f"✅ torch imported successfully - Version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"❌ torch import failed: {e}")

try:
    from transformers import T5Tokenizer
    print("✅ transformers imported successfully")
except ImportError as e:
    print(f"❌ transformers import failed: {e}")

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy and English model loaded successfully")
except ImportError as e:
    print(f"❌ spaCy import failed: {e}")
except OSError as e:
    print(f"❌ spaCy English model not found: {e}")
    print("Run: python -m spacy download en_core_web_sm")

try:
    import nltk
    print("✅ NLTK imported successfully")
except ImportError as e:
    print(f"❌ NLTK import failed: {e}")

try:
    from google import genai
    print("✅ Google GenAI imported successfully")
except ImportError as e:
    print(f"❌ Google GenAI import failed: {e}")