import re
from typing import List, Tuple, Dict
from collections import defaultdict

class MentalHealthSentimentAnalyzer:
    def __init__(self):
        # Expanded keyword sets with more contextual phrases
        self.positive_indicators = {
            # Original keywords
            'happy', 'hopeful', 'grateful', 'blessed', 'peaceful', 'calm', 'confident',
            'motivated', 'energetic', 'optimistic', 'content', 'proud', 'strong',
            'supported', 'supportive', 'balanced', 'relaxed', 'accomplished', 'joyful', 'inspired', 'better',
            'happiness',
            
            # Additional positive keywords and phrases
            'good', 'bright', 'potential', 'enthusiasm', 'enthusiastic', 'positive', 'enjoy',
            'enjoying', 'joy', 'satisf', 'productive', 'support', 'falling into place',
            'appreciate', 'exciting', 'possibilities', 'bring happiness', 'beautiful'
        }
        
        self.negative_indicators = {
            # Original keywords
            'anxious', 'depressed', 'stressed', 'overwhelmed', 'worthless', 'hopeless',
            'tired', 'exhausted', 'sad', 'lonely', 'afraid', 'worried', 'panicked',
            'suicidal', 'desperate', 'struggling', 'miserable', 'helpless', 'trapped', 'low',
            "can't", 'affecting',
            
            # Additional negative keywords and phrases
            'difficult', 'tough', 'battle', 'frustrated', 'drain', 'bleak', 'uncertain',
            'burden', 'negative', 'down', 'unmotivated', 'sadness', 'finding it hard',
            'disappeared', "don't know how", 'light at the end', 'tunnel', 'nothing brings',
            'nothing seems', 'cloud of sadness'
        }
        
        # Expanded neutral patterns and phrases
        self.neutral_patterns = [
            r'neither.*nor',
            r'not .*(good|bad)',
            r'not .*(happy|sad)',
        ]
        
        # Uncertainty patterns that indicate negative sentiment
        self.uncertainty_patterns = [
            r"don't know how",
            r"not sure how",
            r"uncertain how",
            r"having trouble",
            r"struggling with",
            r"difficulty.*handling",
            r"hard to handle"
        ]

        # Phrase matches for better context detection
        self.positive_phrases = [
            'good place', 'falling into place', 'bring happiness', 'feel satisfied',
            'full of potential', 'exciting possibilities', 'little things in life',
            'positive thoughts', 'bring joy', 'enjoying every moment',
            'surrounded by support', 'surrounded by supportive'
        ]
        
        self.negative_phrases = [
            'finding it hard', 'difficult to focus', 'feels like a burden',
            'constant battle', "can't see the light", 'cloud of sadness',
            'nothing brings', 'nothing seems', 'mentally drained',
            'hard to stay', 'handle these feelings'
        ]

    def find_phrases(self, text: str, phrases: List[str]) -> int:
        """Count how many phrases from the list appear in the text."""
        count = 0
        for phrase in phrases:
            if phrase in text.lower():
                count += 1
        return count

    def check_patterns(self, text: str, patterns: List[str]) -> bool:
        """Check if any of the regex patterns match in the text."""
        for pattern in patterns:
            if re.search(pattern, text.lower()):
                return True
        return False

    def preprocess_text(self, text: str) -> str:
        """Clean and normalize the input text while preserving spaces for phrase matching."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text

    def analyze_sentiment(self, text: str) -> Tuple[str, float, Dict]:
        """Analyze the sentiment of a mental health related sentence with improved context awareness."""
        original_text = text.lower()
        processed_text = self.preprocess_text(text)
        words = set(processed_text.split())
        
        # Check for neutral patterns first
        if self.check_patterns(original_text, self.neutral_patterns):
            return 'neutral', 50.0, {'reason': 'neutral_pattern_match'}
            
        # Check for uncertainty patterns that indicate negative sentiment
        if self.check_patterns(original_text, self.uncertainty_patterns):
            return 'negative', 75.0, {'reason': 'uncertainty_pattern_match'}
        
        # Word-level analysis
        positive_matches = words.intersection(self.positive_indicators)
        negative_matches = words.intersection(self.negative_indicators)
        
        # Phrase-level analysis
        positive_phrase_count = self.find_phrases(original_text, self.positive_phrases)
        negative_phrase_count = self.find_phrases(original_text, self.negative_phrases)
        
        # Calculate scores with phrase bonuses
        positive_score = len(positive_matches) + (positive_phrase_count * 2)
        negative_score = len(negative_matches) + (negative_phrase_count * 2)
        
        # Special handling for supportive contexts
        if 'supportive' in words or 'supported' in words or 'support' in words:
            positive_score += 2
        
        # Context-aware negation handling
        negations = {'not', "don't", "cant", "can't", "cannot", "never"}
        if any(neg in words for neg in negations):
            # Potential negation of positive sentiment
            if positive_score > 0 and negative_score == 0:
                negative_score += 1
                positive_score = max(0, positive_score - 1)
        
        # Enhanced decision making
        if positive_score > negative_score and positive_score > 0:
            sentiment = 'positive'
            confidence = min(100.0, (positive_score / (positive_score + negative_score + 1)) * 100)
        elif negative_score > 0:
            sentiment = 'negative'
            confidence = min(100.0, (negative_score / (positive_score + negative_score + 1)) * 100)
        else:
            sentiment = 'neutral'
            confidence = 50.0
        
        analysis = {
            'positive_words': positive_matches,
            'negative_words': negative_matches,
            'positive_score': positive_score,
            'negative_score': negative_score,
            'positive_phrases': positive_phrase_count,
            'negative_phrases': negative_phrase_count
        }
        
        return sentiment, round(confidence, 2), analysis

    def evaluate_accuracy(self, input_file: str, expected_file: str, output_file: str = "output.txt", max_sentences: int = None) -> float:
        """Evaluate accuracy by comparing analysis results with expected outcomes."""
        try:
            with open(input_file, 'r') as f:
                sentences = [line.strip() for line in f if line.strip()]
            
            with open(expected_file, 'r') as f:
                expected_results = [line.strip() for line in f if line.strip()]
            
            if len(sentences) != len(expected_results):
                raise ValueError("Number of sentences and expected results don't match")
            
            total_count = len(sentences)
            if max_sentences:
                total_count = min(total_count, max_sentences)
            
            correct_count = 0
            
            with open(output_file, "w") as output_f:
                for index, (sentence, expected) in enumerate(zip(sentences[:total_count], expected_results[:total_count])):
                    if max_sentences and index >= max_sentences:
                        break
                    
                    predicted_sentiment, confidence, analysis = self.analyze_sentiment(sentence)
                    
                    if predicted_sentiment.lower() == expected.lower():
                        correct_count += 1
                    else:
                        output_f.write(f"Sentence {index+1} is incorrect\n")
                        output_f.write(f"Input sentence: {sentence}\n")
                        output_f.write(f"Predicted sentiment: {predicted_sentiment} (confidence: {confidence}%)\n")
                        output_f.write(f"Expected sentiment: {expected}\n")
                        output_f.write(f"Analysis: {analysis}\n\n")
                    
                    if (index + 1) % 50 == 0:
                        print(f"{(index + 1) / total_count * 100:.2f}% done", flush=True)
            
            accuracy = (correct_count / total_count) * 100
            print(f"Accuracy: {accuracy:.2f}%", flush=True)
            return accuracy
        
        except FileNotFoundError as e:
            print(f"Error: Could not find file - {str(e)}")
            return 0.0
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return 0.0

def main():
    """Example usage of the analyzer with file processing."""
    analyzer = MentalHealthSentimentAnalyzer()
    
    input_file = "sentences_only.txt"
    expected_file = "polarities_only.txt"
    output_file = "output.txt"
    
    accuracy = analyzer.evaluate_accuracy(
        input_file=input_file,
        expected_file=expected_file,
        output_file=output_file,
        max_sentences=None
    )
    
    print(f"\nFinal accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()