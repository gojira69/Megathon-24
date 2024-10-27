import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as sw, wordnet
import contractions
import string
from typing import List, Set, Optional, Dict, Tuple
import re
import sys
import os

user_name = sys.argv[1]

print(user_name)

file_path = os.path.join("Data", user_name, "therapist_notes.txt")
output_file_path = os.path.join("Data", user_name, "extracted_concern.txt")

try:
    with open(file_path, 'r') as file:
        user_input = file.read()  # Read the entire file content into a string
except FileNotFoundError:
    print(f"The file {file_path} does not exist.")
    sys.exit(1)  # Exit if the file is not found
except Exception as e:
    print(f"An error occurred while reading the file: {e}")
    sys.exit(1)  # Exit if any other error occurs

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class MentalHealthExtractor:
    def __init__(self):
        try:
            # Initialize base mental health indicators
            self.mental_health_indicators = {
                'emotions': {'feeling', 'feel', 'felt', 'feels'},
                'states': {
                    'anxious', 'anxiety', 'depressed', 'depression', 'worried', 'worry',
                    'stressed', 'stress', 'overwhelmed', 'hopeless', 'hopeful', 'afraid',
                    'scared', 'fearful', 'lonely', 'alone', 'tired', 'exhausted',
                    'confused', 'panic', 'angry', 'sad', 'happy', 'motivated', 'unmotivated',
                    'low', 'excited', 'better', 'tensed', 'perturbed', 'disturbed', 'upset',
                    'distressed', 'troubled', 'concerned', 'uneasy', 'agitated'
                },
                'intensifiers': {
                    'very', 'extremely', 'constantly', 'really', 'quite', 'so', 'much',
                    'little', 'slightly', 'somewhat', 'highly', 'increasingly',
                    'particularly', 'especially', 'exceptionally'
                },
                'contexts': {
                    'job': ['prospects', 'situation', 'search', 'performance', 'security'],
                    'health': ['condition', 'issues', 'problems', 'symptoms', 'wellness'],
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
            
            self.synonym_cache = {}
            self.synonym_mapping = {}
            self.expanded_states = set()
            self.expanded_contexts = {}
            self.stopwords = self._get_stopwords()
            self._initialize_synonym_cache()
            self._expand_mental_states()
            self._expand_contexts()
            self.pattern_rules = self._create_pattern_rules()
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def _create_pattern_rules(self) -> Dict[str, List[str]]:
        """
        Create pattern rules for text extraction with improved feeling patterns.
        Returns a dictionary of pattern rules.
        """
        try:
            pattern_rules = {
                'feeling_patterns': [
                    # Basic feeling patterns
                    r'(?:feeling|feel|feels?\s+like)\s+(?:{})?(?:{})'.format(
                        r'(?:' + r'|'.join(self.mental_health_indicators['intensifiers']) + r')\s+',
                        r'|'.join(self.mental_health_indicators['states'])
                    ),
                    # "I am/I'm feeling" patterns
                    r'(?:I\s+am|I\'m|I\'ve\s+been|am|is|are|\'m|\'re|been)\s+(?:feeling|feel)\s+(?:{})?(?:{})'.format(
                        r'(?:' + r'|'.join(self.mental_health_indicators['intensifiers']) + r')\s+',
                        r'|'.join(self.mental_health_indicators['states'])
                    )
                ],
                'feeling_state_context': [
                    # Patterns for "feeling [intensifier] [state] about [context]"
                    r'(?:am|is|are|\'m|\'re|been|feels\s+like\s+(?:I|we|he|she|they)(?:\'m|\'re|\'s)?)\s+(?:feeling\s+)?(?:{})(?:{})\s+(?:about\s+(?:{}))?'.format(
                        r'(?:' + r'|'.join(self.mental_health_indicators['intensifiers']) + r')\s+',
                        r'|'.join(self.mental_health_indicators['states']),
                        r'|'.join([context for context in self.mental_health_indicators['contexts']])
                    )
                ],
                'state_context': [
                    # Direct patterns for "[state] about [context]"
                    r'(?:{})\s+about\s+(?:{})'.format(
                        r'|'.join(self.mental_health_indicators['states']),
                        r'|'.join([context for context in self.mental_health_indicators['contexts']])
                    )
                ],
                'intensified_state_context': [
                    # Patterns for "[intensifier] [state] about [context]"
                    r'(?:{})\s+(?:{})\s+about\s+(?:{})'.format(
                        r'|'.join(self.mental_health_indicators['intensifiers']),
                        r'|'.join(self.mental_health_indicators['states']),
                        r'|'.join([context for context in self.mental_health_indicators['contexts']])
                    )
                ]
            }
            # Compile all patterns for efficiency
            return {key: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
                   for key, patterns in pattern_rules.items()}
        except Exception as e:
            print(f"Error creating pattern rules: {str(e)}")
            return {}

    def _expand_mental_states(self):
        """Expand mental states using WordNet synonyms."""
        try:
            for state in self.mental_health_indicators['states']:
                # Get direct synonyms
                synonyms = self._get_synonyms(state)
                # Get related adjectives
                for syn in wordnet.synsets(state):
                    if syn.pos() in ['a', 's', 'v']:  # adjectives and verbs
                        for lemma in syn.lemmas():
                            synonyms.add(lemma.name().lower())
                            # Add antonyms if relevant
                            for antonym in lemma.antonyms():
                                if antonym.name().lower() in self.mental_health_indicators['states']:
                                    synonyms.add(antonym.name().lower())
                
                self.expanded_states.update(synonyms)
            
            # Update states with expanded set
            self.mental_health_indicators['states'].update(self.expanded_states)
        except Exception as e:
            print(f"Error expanding mental states: {str(e)}")

    def _expand_contexts(self):
        """Expand contexts using WordNet synonyms."""
        try:
            for context, terms in self.mental_health_indicators['contexts'].items():
                expanded_terms = set(terms)
                for term in terms:
                    synonyms = self._get_synonyms(term)
                    expanded_terms.update(synonyms)
                self.expanded_contexts[context] = list(expanded_terms)
            
            # Update contexts with expanded terms
            self.mental_health_indicators['contexts'] = self.expanded_contexts
        except Exception as e:
            print(f"Error expanding contexts: {str(e)}")

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

    def extract_concern(self, sentence: str) -> str:
        """
        Extract mental health concerns from a sentence and return as a string.
        
        Args:
            sentence (str): Input sentence to analyze
            
        Returns:
            str: Extracted mental health concern or empty string if none found
        """
        try:
            extracted_words = self._extract_concern_list(sentence)
            return " ".join(extracted_words) if extracted_words else ""
        except Exception as e:
            print(f"Error extracting concerns: {str(e)}")
            return ""

    def _extract_concern_list(self, sentence: str) -> List[str]:
        """Extract mental health concerns from a sentence with improved pattern matching."""
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
            
            # Check feeling patterns first
            for pattern in self.pattern_rules['feeling_patterns']:
                match = pattern.search(processed_text)
                if match:
                    match_text = match.group(0)
                    words = match_text.split()
                    
                    # Process the matched words
                    result = []
                    feeling_found = False
                    for word in words:
                        if word in {'feeling', 'feel', 'feels'}:
                            result.append('feeling')
                            feeling_found = True
                        elif (word in self.mental_health_indicators['intensifiers'] or
                              word in self.mental_health_indicators['states']):
                            result.append(word)
                    
                    if result and not feeling_found and 'feeling' in processed_text:
                        result.insert(0, 'feeling')
                    
                    if result:
                        return result
            
            # Check other pattern types
            for pattern_type in ['feeling_state_context', 'state_context', 'intensified_state_context']:
                for pattern in self.pattern_rules[pattern_type]:
                    match = pattern.search(processed_text)
                    if match:
                        match_text = match.group(0)
                        words = [word.strip() for word in match_text.split() if word.strip()]
                        
                        words = [w for w in words if w not in {'am', 'is', 'are', "'m", "'re", 'been', 
                                                             'feels', 'feel', 'i', 'like'}]
                        
                        result = []
                        for word in words:
                            if (word in self.mental_health_indicators['states'] or
                                word in self.mental_health_indicators['intensifiers'] or
                                word == 'about' or
                                any(word == context for context in self.mental_health_indicators['contexts'])):
                                result.append(word)
                        
                        if result:
                            if 'feeling' in processed_text and not result[0] == 'feeling':
                                result.insert(0, 'feeling')
                            return result
            
            return []
        except Exception as e:
            print(f"Error extracting concerns: {str(e)}")
            return []

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
        
        with open(output_file_path, 'w') as output_file:  # Open the file for writing
            while True:
                user_input = input("> ").strip()
                if user_input.lower() == 'quit':
                    break
                
                result = extractor.extract_concern(user_input)
                if result:
                    output_message = f"User Input: {user_input}\nExtracted concern: {result}\n"
                    print(output_message)  # Print to console
                    output_file.write(output_message)  # Write to file
                else:
                    output_message = f"User Input: {user_input}\nNo mental health concerns detected.\n"
                    print(output_message)  # Print to console
                    output_file.write(output_message)  # Write to file

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()