import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as sw, wordnet
import contractions
import string
from typing import List, Set, Optional, Dict, Tuple
import re

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class MentalHealthExtractor:
    def __init__(self):
        try:
            # Initialize mental health indicators
            self.mental_health_indicators = {
                'emotions': {'feeling', 'feel', 'felt', 'feels'},
                'states': {
                    'anxious', 'anxiety', 'depressed', 'depression', 'worried', 'worry',
                    'stressed', 'stress', 'overwhelmed', 'hopeless', 'hopeful', 'afraid',
                    'scared', 'fearful', 'lonely', 'alone', 'tired', 'exhausted',
                    'confused', 'panic', 'angry', 'sad', 'happy', 'motivated', 'unmotivated',
                    'low', 'excited', 'better', 'tensed'  # Added 'tensed' to states
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
            # Initialize other attributes
            self.synonym_cache = {}
            self.synonym_mapping = {}
            self.stopwords = self._get_stopwords()
            self._initialize_synonym_cache()
            self.pattern_rules = self._create_pattern_rules()
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def _create_pattern_rules(self) -> Dict[str, List[str]]:
        """
        Create pattern rules for text extraction.
        Returns a dictionary of pattern rules.
        """
        try:
            # The (?:been\s+)? makes "been" optional in the patterns
            pattern_rules = {
                'feeling_state': [
                    r'(?:am|is|are|\'m|\'re|been)\s+feeling\s+(?:{})\s+(?:{})'.format(
                        '|'.join(self.mental_health_indicators['intensifiers']),
                        '|'.join(self.mental_health_indicators['states'])
                    ),
                    r'feeling\s+(?:{})\s+(?:{})'.format(
                        '|'.join(self.mental_health_indicators['intensifiers']),
                        '|'.join(self.mental_health_indicators['states'])
                    ),
                    r'(?:am|is|are|\'m|\'re|been)\s+feeling\s+(?:{})'.format(
                        '|'.join(self.mental_health_indicators['states'])
                    ),
                    r'feeling\s+(?:{})'.format(
                        '|'.join(self.mental_health_indicators['states'])
                    )
                ],
                'emotion_state': [
                    r'(?:{})\s+(?:{})'.format(
                        '|'.join(self.mental_health_indicators['emotions']),
                        '|'.join(self.mental_health_indicators['states'])
                    )
                ],
                'intensified_state': [
                    r'(?:{})\s+(?:{})'.format(
                        '|'.join(self.mental_health_indicators['intensifiers']),
                        '|'.join(self.mental_health_indicators['states'])
                    )
                ],
                'context_state': [
                    r'(?:{})\s+(?:about|regarding|concerning)\s+(?:{})'.format(
                        '|'.join(self.mental_health_indicators['states']),
                        '|'.join([context for context in self.mental_health_indicators['contexts']])
                    )
                ]
            }
            # Compile all patterns for efficiency
            return {key: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
                   for key, patterns in pattern_rules.items()}
        except Exception as e:
            print(f"Error creating pattern rules: {str(e)}")
            return {}

    def _get_stopwords(self) -> Set[str]:
        """
        Get stopwords while excluding important mental health related terms.
        """
        try:
            # Get base stopwords from nltk
            stopwords = set(sw.words("english"))
            
            # Words to exclude from stopwords because they're important for mental health context
            exclude_words = set()
            
            # Add context words to exclusions
            for context, details in self.mental_health_indicators['contexts'].items():
                exclude_words.add(context)
                exclude_words.update(details)
            
            # Add other important words to exclusions
            additional_exclude = {
                "n't", "not", "never", "no", 
                "feeling", "feel", "feels", "felt",
                "very", "constantly", "much", "and", 
                "well", "better", "lately",
                "these", "days", "about", "like", 
                "been", "extremely", "particularly",
                "especially", "somewhat", "slightly", 
                "little", "properly", "eating", 
                "sleeping"
            }
            exclude_words.update(additional_exclude)
            
            # Add punctuation and special symbols to stopwords
            additional_symbols = {'--', '...', '–', '—'}
            punctuations = set(string.punctuation).union(additional_symbols)
            
            return stopwords.union(punctuations) - exclude_words
        except Exception as e:
            print(f"Error getting stopwords: {str(e)}")
            return set()

    def _get_wordnet_pos(self, word: str) -> List[str]:
        """Get all possible parts of speech for a word."""
        try:
            pos_tags = nltk.pos_tag([word])
            tag = pos_tags[0][1][0].upper()
            tag_dict = {
                "J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV
            }
            return [tag_dict.get(tag, wordnet.NOUN)]
        except Exception as e:
            print(f"Error getting word POS: {str(e)}")
            return [wordnet.NOUN]

    def _get_synonyms(self, word: str) -> Set[str]:
        """Get synonyms for a word using WordNet with part of speech consideration."""
        try:
            if word in self.synonym_cache:
                return self.synonym_cache[word]

            synonyms = set()
            pos_tags = self._get_wordnet_pos(word)
            
            for pos in pos_tags:
                for syn in wordnet.synsets(word, pos=pos):
                    # Add lemma names
                    for lemma in syn.lemmas():
                        synonym = lemma.name().lower().replace('_', ' ')
                        synonyms.add(synonym)
                        self.synonym_mapping[synonym] = word
                    
                    # Add hypernyms
                    for hypernym in syn.hypernyms():
                        for lemma in hypernym.lemmas():
                            synonym = lemma.name().lower().replace('_', ' ')
                            synonyms.add(synonym)
                            self.synonym_mapping[synonym] = word
                    
                    # Add similar_tos for adjectives
                    if pos == wordnet.ADJ:
                        for similar in syn.similar_tos():
                            for lemma in similar.lemmas():
                                synonym = lemma.name().lower().replace('_', ' ')
                                synonyms.add(synonym)
                                self.synonym_mapping[synonym] = word

            self.synonym_cache[word] = synonyms
            return synonyms
        except Exception as e:
            print(f"Error getting synonyms for {word}: {str(e)}")
            return set()

    def extract_concern(self, sentence: str) -> List[str]:
        """Extract mental health concerns from a sentence with pattern matching."""
        try:
            processed_text = sentence.lower()
            
            # Check for exact kept phrases first
            for phrase in self.mental_health_indicators['keep_phrases']:
                if phrase in processed_text:
                    return phrase.split()
            
            # Check behavioral patterns
            for category, patterns in self.mental_health_indicators['behavioral_patterns'].items():
                for pattern in patterns:
                    if pattern in processed_text:
                        return pattern.split()
            
            # Check for feeling patterns first
            for pattern in self.pattern_rules['feeling_state']:
                matches = pattern.search(processed_text)
                if matches:
                    match_text = matches.group(0)
                    # Clean up any leading/trailing whitespace and split into words
                    words = [word for word in match_text.split() if word]
                    if 'been' in words:
                        words.remove('been')
                    if any(word in ['am', 'is', 'are', "'m", "'re"] for word in words):
                        words = [word for word in words if word not in ['am', 'is', 'are', "'m", "'re"]]
                    return words
            
            # If no feeling pattern matches, check other patterns
            words = word_tokenize(processed_text)
            
            # Fall back to word-by-word analysis
            for i, word in enumerate(words):
                if word not in self.stopwords:
                    # Check if word is a state or synonym of a state
                    is_state, matching_state = self._find_matching_state(word)
                    if is_state:
                        # Look for intensifiers
                        for j in range(max(0, i-1), i):
                            if words[j] in self.mental_health_indicators['intensifiers']:
                                return [words[j], matching_state]
                        
                        # Look for context
                        context_window = words[max(0, i-2):min(len(words), i+3)]
                        for context_word in context_window:
                            if context_word not in self.stopwords:
                                has_context, context_type, matched_context = self._find_matching_context(context_word)
                                if has_context:
                                    return [matching_state, 'about', matched_context]
                        
                        return [matching_state]
            
            return []
        except Exception as e:
            print(f"Error extracting concerns: {str(e)}")
            return []

    def _find_matching_state(self, word: str) -> Tuple[bool, str]:
        """Check if a word or its synonyms match any mental health state."""
        try:
            if word in self.mental_health_indicators['states']:
                return True, word
            
            if word in self.synonym_mapping:
                original_word = self.synonym_mapping[word]
                if original_word in self.mental_health_indicators['states']:
                    return True, original_word
            
            word_synonyms = self._get_synonyms(word)
            for state in self.mental_health_indicators['states']:
                state_synonyms = self._get_synonyms(state)
                if word in state_synonyms or any(syn in state_synonyms for syn in word_synonyms):
                    return True, state
            
            return False, ''
        except Exception as e:
            print(f"Error finding matching state: {str(e)}")
            return False, ''

    def _find_matching_context(self, word: str) -> Tuple[bool, str, str]:
        """Check if a word or its synonyms match any context or context detail."""
        try:
            for context, details in self.mental_health_indicators['contexts'].items():
                if word == context or word in details:
                    return True, context, word
                
                context_synonyms = self._get_synonyms(context)
                if word in context_synonyms:
                    return True, context, context
                    
                for detail in details:
                    detail_synonyms = self._get_synonyms(detail)
                    if word in detail_synonyms:
                        return True, context, detail
                        
            return False, '', ''
        except Exception as e:
            print(f"Error finding matching context: {str(e)}")
            return False, '', ''

    def _initialize_synonym_cache(self):
        """Pre-calculate synonyms for all known states and contexts."""
        try:
            for state in self.mental_health_indicators['states']:
                self._get_synonyms(state)
            
            for context, details in self.mental_health_indicators['contexts'].items():
                self._get_synonyms(context)
                for detail in details:
                    self._get_synonyms(detail)
        except Exception as e:
            print(f"Error initializing synonym cache: {str(e)}")

    def process_file(self, input_file: str = "sentences.txt") -> float:
        """Process input file and calculate accuracy."""
        try:
            with open(input_file, "r") as file:
                sentences = file.read().split("\n")
            with open("extract.txt", "r") as file:
                expected_extracts = file.read().split("\n")

            correct_count = 0
            total_count = len(sentences)
            
            with open("output.txt", "w") as output_file:
                for index, sentence in enumerate(sentences[:total_count]):
                    extraction = self.extract_concern(sentence)
                    extracted_text = " ".join(extraction)
                    expected_text = expected_extracts[index]
                    
                    if extracted_text == expected_text:
                        correct_count += 1
                    else:
                        output_file.write(f"Sentence {index+1} is incorrect\n")
                        output_file.write(f"Sentence: {sentence}\n")
                        output_file.write(f"Extracted sentence: {extracted_text}\n")
                        output_file.write(f"Expected sentence: {expected_text}\n\n")
                    
                    if (index + 1) % 5000 == 0:
                        print(f"{(index + 1) / total_count * 100:.2f}% done", flush=True)
            
            accuracy = (correct_count / total_count) * 100
            print(f"Accuracy: {accuracy:.2f}%", flush=True)
            return accuracy
            
        except FileNotFoundError as e:
            print(f"Error: Required file not found - {e.filename}")
            return 0.0
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return 0.0

def main():
    try:
        extractor = MentalHealthExtractor()
        extractor.process_file()
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()