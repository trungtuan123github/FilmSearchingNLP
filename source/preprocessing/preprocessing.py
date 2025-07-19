import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import contractions
from typing import List, Optional, Set

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

### LSA/SVD preprocessing class
class LSASVDPipeline:
    """
    Preprocessing pipeline for LSA/SVD
    """

    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    html_pattern = re.compile(r'<[^>]+>')
    non_alpha_pattern = re.compile(r'[^a-zA-Z\s]')

    def __init__(
        self,
        extra_stopwords: Optional[Set[str]] = None
    ):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        base_sw = set(stopwords.words('english'))
        self.stop_words = base_sw.union(extra_stopwords or set())

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        # lowercase
        text = text.lower()
        # remove HTML
        text = self.html_pattern.sub('', text)
        # remove URLs
        text = self.url_pattern.sub('', text)
        # remove non-alphabetic
        text = self.non_alpha_pattern.sub(' ', text)
        # normalize spaces
        return re.sub(r'\s+', ' ', text).strip()

    def tokenize_filter(self, text: str) -> List[str]:
        tokens = word_tokenize(text)
        return [tok for tok in tokens if tok not in self.stop_words and len(tok) > 2]

    def stem(self, tokens: List[str]) -> List[str]:
        return [self.stemmer.stem(tok) for tok in tokens]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        return [self.lemmatizer.lemmatize(tok) for tok in tokens]

    def preprocess(self, text: str, use_stemming: bool = True) -> str:
        cleaned = self.clean_text(text)
        tokens = self.tokenize_filter(cleaned)
        processed = self.stem(tokens) if use_stemming else self.lemmatize(tokens)
        return ' '.join(processed)

    def batch(self, texts: List[str], use_stemming: bool = True) -> List[str]:
        return [self.preprocess(txt, use_stemming) for txt in texts]

### Word Embedding preprocessing classs
class WordEmbeddingPipeline:
    """
    Preprocessing pipeline for WordEmbedding (Word2Vec/GloVe/FastText)
    --> CORE: Lightweight preprocessing. Maintain context.
    """

    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    html_pattern = re.compile(r'<[^>]+>')
    punct_except_basic = re.compile(r'[^\w\s\.\!?]')

    def __init__(self, minimal_stopwords: Optional[Set[str]] = None):
        self.lemmatizer = WordNetLemmatizer()
        defaults = {"a", "an", "the", "and", "or", "but", "is", "are", "was", "were"}
        self.stop_words = defaults.union(minimal_stopwords or set())

    def expand_contractions(self, text: str) -> str:
        return contractions.fix(text)

    def clean_gentle(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = self.expand_contractions(text)
        text = text.lower()
        text = self.html_pattern.sub('', text)
        text = self.url_pattern.sub(' ', text)
        text = self.punct_except_basic.sub(' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def tokenize_sentences(self, text: str) -> List[List[str]]:
        sentences = sent_tokenize(text)
        result = []
        for sent in sentences:
            toks = word_tokenize(sent)
            filtered = [tok for tok in toks if tok not in self.stop_words and len(tok) > 1]
            if filtered:
                result.append(filtered)
        return result

    def lemmatize(self, tokens: List[str]) -> List[str]:
        return [self.lemmatizer.lemmatize(tok) for tok in tokens]

    def preprocess(self, text: str) -> List[List[str]]:
        cleaned = self.clean_gentle(text)
        sents = self.tokenize_sentences(cleaned)
        return [self.lemmatize(sent) for sent in sents]

    def preprocess_single_text(self, text: str) -> List[str]:
        sents = self.preprocess(text)
        return [tok for sent in sents for tok in sent]

    def flatten(self, text: str) -> str:
        return ' '.join(self.preprocess_single_text(text))

    def batch(self, texts: List[str]) -> List[List[List[str]]]:
        return [self.preprocess(txt) for txt in texts]
    