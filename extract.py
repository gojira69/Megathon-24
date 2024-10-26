import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords as sw
from nltk.tag import pos_tag
import contractions
import string
from typing import List, Set, Optional
import re

nltk.download('stopwords', quiet = True)
nltk.download('punkt', quiet = True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class MentalHealthExtractor:
    def __init__(self):
        self.mental_health_indicators = {
            'emotions': {'feeling', 'feel', 'felt', 'feels'},
            'states': {
                'anxious', 'anxiety', 'depressed', 'depression', 'worried', 'worry',
                'stressed', 'stress', 'overwhelmed', 'hopeless', 'hopeful', 'afraid',
                'scared', 'fearful', 'lonely', 'alone', 'tired', 'exhausted',
                'confused', 'panic', 'angry', 'sad', 'happy', 'motivated', 'unmotivated',
                'low', 'excited', 'better'
            },
            'intensifiers': {
                'very', 'extremely', 'constantly', 'really', 'quite', 'so', 'much',
                'little', 'slightly', 'somewhat', 'highly', 'increasingly',
                'particularly', 'especially', 'exceptionally'
            },
            'contexts': {
                'job': ['prospects', 'situation', 'search', 'performance', 'security'],
                'health': ['condition', 'issues', 'problems', 'symptoms'],
                'future': ['career', 'plans', 'opportunities', 'prospects'],
                'work': ['environment', 'situation', 'conditions', 'pressure'],
                'family': ['situation', 'issues', 'relationships', 'matters'],
                'financial': ['situation', 'problems', 'issues', 'status']
            },
            'keep_phrases': {
                'feeling hopeful', 'feeling better', "can't sleep well",
                'feeling much better', 'happy and excited', 'constantly worried',
                'extremely stressed', 'worried about health', 'not eating properly',
                'confused about job prospects'
            },
            'behavioral_patterns': {
                'eating': ['not eating properly', 'eating poorly', "can't eat"],
                'sleeping': ["can't sleep", 'not sleeping', 'sleeping poorly']
            }
        }
        self.stopwords = self._get_stopwords()
        self.extract_words = self._load_extract_words()
        self.pattern_rules = self._create_pattern_rules()

    def _get_stopwords(self) -> Set[str]:
        """Get filtered stopwords excluding important context words."""
        stopwords = set(sw.words("english"))
        exclude_words = set()
        
        # Add all context-related words to exclude
        for context, details in self.mental_health_indicators['contexts'].items():
            exclude_words.add(context)
            exclude_words.update(details)
        
        # Add other important words to exclude
        exclude_words.update({
            "n't", "not", "never", "no", "feeling", "feel", "feels", "felt",
            "very", "constantly", "much", "and", "well", "better", "lately",
            "these", "days", "about", "like", "been", "extremely", "particularly",
            "especially", "somewhat", "slightly", "little", "properly", "eating", 
            "sleeping"
        })
        
        additional_symbols = {'--', '...', '–', '—'}
        punctuations = set(string.punctuation).union(additional_symbols)
        return stopwords.union(punctuations) - exclude_words

    def _create_pattern_rules(self):
        """Create regex patterns for different extraction rules."""
        # Create context patterns
        context_patterns = []
        for base_context, details in self.mental_health_indicators['contexts'].items():
            context_pattern = f"{base_context}(?:\\s+(?:{'|'.join(details)}))?\\b"
            context_patterns.append(context_pattern)
        
        context_regex = '|'.join(context_patterns)
        
        return [
            # Rule 1: state + about + context (highest priority)
            (r'({})\s+about\s+({})'.format(
                '|'.join(self.mental_health_indicators['states']),
                context_regex
            ), lambda m: f"{m.group(1)} about {m.group(2)}"),
            
            # Rule 2: not eating/sleeping properly
            (r'not\s+(?:eating|sleeping)\s+properly\b', 
             lambda m: m.group(0)),
            
            # Rule 3: feeling + state (with optional intensifier)
            (r'feeling\s+(?:{}\s+)?({})\b'.format(
                '|'.join(self.mental_health_indicators['intensifiers']),
                '|'.join(self.mental_health_indicators['states'])
            ), lambda m: m.group(0)),
            
            # Rule 4: state + and + state
            (r'({})\s+and\s+({})\b'.format(
                '|'.join(self.mental_health_indicators['states']),
                '|'.join(self.mental_health_indicators['states'])
            ), lambda m: f"{m.group(1)} and {m.group(2)}"),
            
            # Rule 5: intensifier + state (lowest priority)
            (r'({})\s+({})\b'.format(
                '|'.join(self.mental_health_indicators['intensifiers']),
                '|'.join(self.mental_health_indicators['states'])
            ), lambda m: f"feeling {m.group(1)} {m.group(2)}")
        ]

    def _load_extract_words(self) -> Set[str]:
        try:
            with open("extract.txt", "r") as file:
                text = file.read().lower()
                return set(word_tokenize(text))
        except FileNotFoundError:
            print("Warning: extract.txt not found. Using default mental health terms.")
            return set().union(*self.mental_health_indicators.values())

    def _preprocess_sentence(self, sentence: str) -> str:
        """Preprocess the input sentence."""
        # Fix contractions but preserve can't
        sentence = sentence.replace("can't", "cannot")
        sentence = contractions.fix(sentence)
        sentence = sentence.replace("cannot", "can't")
        
        # Standardize context phrases
        processed = sentence.lower()
        
        # Standardize various context phrases
        for base_context, details in self.mental_health_indicators['contexts'].items():
            for detail in details:
                # Create variations of the context phrase
                variations = [
                    f"{base_context} {detail}",
                    f"{base_context}-{detail}",
                    f"{base_context}{detail}"
                ]
                for variation in variations:
                    processed = processed.replace(variation, f"{base_context} {detail}")
        
        return processed

    def _check_context_pattern(self, words: List[str], state_idx: int) -> Optional[List[str]]:
        """Check for context patterns around a state word."""
        if state_idx >= len(words):
            return None
            
        # Look for "about" followed by context
        if state_idx + 2 < len(words) and words[state_idx + 1] == 'about':
            # Get the base context word
            potential_context = words[state_idx + 2]
            
            # Check if it's a known context
            if potential_context in self.mental_health_indicators['contexts']:
                # Look for additional context details
                if state_idx + 3 < len(words):
                    for details in self.mental_health_indicators['contexts'][potential_context]:
                        if words[state_idx + 3] in details.split():
                            return [words[state_idx], 'about', potential_context, words[state_idx + 3]]
                
                # Return base context if no details match
                return [words[state_idx], 'about', potential_context]
                
        return None

    def extract_concern(self, sentence: str) -> List[str]:
        """Extract mental health concerns from a sentence."""
        processed_text = self._preprocess_sentence(sentence)
        words = word_tokenize(processed_text)
        
        # First check for exact kept phrases
        for phrase in self.mental_health_indicators['keep_phrases']:
            if phrase in processed_text:
                return phrase.split()
        
        # Apply pattern rules
        for pattern, formatter in self.pattern_rules:
            match = re.search(pattern, processed_text)
            if match:
                return formatter(match).split()
        
        # Check for states with context
        for i, word in enumerate(words):
            if word in self.mental_health_indicators['states']:
                # First priority: Check for context patterns
                context_match = self._check_context_pattern(words, i)
                if context_match:
                    return context_match
                
                # Second priority: Check for feeling + state
                if 'feeling' in words[:i]:
                    return ['feeling', word]
                
                # Third priority: Check for intensifier + state
                if i > 0 and words[i-1] in self.mental_health_indicators['intensifiers']:
                    return ['feeling', words[i-1], word]
                
                # Last resort: return just the state
                return [word]
        
        # Check behavioral patterns as last resort
        for category, patterns in self.mental_health_indicators['behavioral_patterns'].items():
            for pattern in patterns:
                if pattern in processed_text:
                    return pattern.split()
        
        return []

    def process_file(self, input_file: str = "sentences.txt", max_sentences: int = 1000) -> float:
        try:
            with open(input_file, "r") as file:
                sentences = file.read().split("\n")
            with open("extract.txt", "r") as file:
                expected_extracts = file.read().split("\n")

            correct_count = 0
            total_count = min(max_sentences, len(sentences))
            
            with open("output.txt", "w") as output_file:
                for index, sentence in enumerate(sentences[:total_count]):
                    if index >= max_sentences:
                        break
                    
                    extraction = self.extract_concern(sentence)
                    extracted_text = " ".join(extraction)
                    expected_text = expected_extracts[index]
                    
                    if extracted_text == expected_text:
                        correct_count += 1
                    else:
                        output_file.write(f"Sentence {index+1} is incorrect\n")
                        output_file.write(f"Extracted sentence: {extracted_text}\n")
                        output_file.write(f"Expected sentence: {expected_text}\n\n")
                    
                    if (index + 1) % 50 == 0:
                        print(f"{(index + 1) / total_count * 100:.2f}% done", flush=True)
            
            accuracy = (correct_count / total_count) * 100
            print(f"Accuracy: {accuracy:.2f}%", flush=True)
            return accuracy
            
        except FileNotFoundError as e:
            print(f"Error: Required file not found - {e.filename}")
            return 0.0

def main():
    extractor = MentalHealthExtractor()
    extractor.process_file()

if __name__ == "__main__":
    main()