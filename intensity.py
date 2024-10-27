import re
from typing import List, Dict

quantifiers = {
    "very": 4, "extremely": 5, "constantly": 2, "really": 3, "quite": 2,
    "so": 4, "much": 1, "little": -2, "slightly": -1, "somewhat": 0,
    "highly": 5, "increasingly": 3, "particularly": 2, "especially": 3,
    "exceptionally": 5, "scarcely": -4, "barely": -3, "slightly": -2,
    "marginally": -2, "somewhat": -1, "mildly": -1, "partially": 0,
    "relatively": 0, "fairly": 1, "moderately": 1, "noticeably": 2,
    "considerably": 2, "significantly": 3, "greatly": 4, "remarkably": 4,
    "hugely": 5, "tremendously": 5, "immensely": 5, "profoundly": 5,
    "massively": 5, "intensely": 5
}

BASE_INTENSITY = 5

def extract_quantifiers(sentence: str) -> List[str]:
    """Extract quantifiers from the sentence based on predefined quantifiers list."""
    # Normalize the sentence and split words
    sentence = sentence.lower()
    words = re.findall(r'\b\w+\b', sentence)
    
    # Identify quantifiers present in the sentence
    found_quantifiers = [word for word in words if word in quantifiers]
    return found_quantifiers

def calculate_intensity(category: str, sentence: str) -> int:
    """Calculate the intensity score based on quantifiers in the sentence."""
    intensity = BASE_INTENSITY
    
    # Extract quantifiers from the sentence
    found_quantifiers = extract_quantifiers(sentence)
    
    for quantifier in found_quantifiers:
        intensity += quantifiers[quantifier]
    
    return intensity

def process_sentences(sentences: List[str], categories: List[str]) -> List[Dict[str, int]]:
    """
    Process multiple sentences to calculate the intensity for each associated category.
    Expects `categories` list to be in the same order as `sentences`.
    """
    intensity_results = []
    
    for sentence, category in zip(sentences, categories):
        intensity = calculate_intensity(category, sentence)
        intensity_results.append({'category': category, 'sentence': sentence, 'intensity': intensity})
    
    return intensity_results

sentences = [
    "I'm feeling very anxious about my health.",
    "I am extremely stressed about work.",
    "I've been having trouble sleeping lately.",
    "It’s a struggle, I’m little worried."
]

# Assume we have found these categories for each sentence
categories = ["Health Anxiety", "Stress", "Insomnia", "Stress"]

results = process_sentences(sentences, categories)

for result in results:
    print(f"Sentence: '{result['sentence']}'")
    print(f"Category: {result['category']}, Intensity: {result['intensity']}")
    print("-" * 50)
