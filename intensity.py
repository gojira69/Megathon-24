import re

# Quantifiers dictionary with intensity scores
quantifiers = {
    "very": 4, "extremely": 5, "constantly": 2, "really": 3, "quite": 2,
    "so": 4, "much": 1, "little": -2, "slightly": -1, "somewhat": 0,
    "highly": 5, "increasingly": 3, "particularly": 2, "especially": 3,
    "exceptionally": 5, "scarcely": -4, "barely": -3, "marginally": -2,
    "mildly": -1, "relatively": 0, "fairly": 1, "moderately": 1,
    "noticeably": 2, "considerably": 2, "significantly": 3, "greatly": 4,
    "remarkably": 4, "hugely": 5, "tremendously": 5, "immensely": 5,
    "profoundly": 5, "massively": 5, "intensely": 5
}

BASE_INTENSITY = 5

# Function to extract quantifiers
def extract_quantifiers(sentence: str):
    words = re.findall(r'\b\w+\b', sentence.lower())
    return [word for word in words if word in quantifiers]

# Function to calculate intensity
def calculate_intensity(sentence: str) -> int:
    intensity = BASE_INTENSITY
    found_quantifiers = extract_quantifiers(sentence)
    for quantifier in found_quantifiers:
        intensity += quantifiers[quantifier]
    return intensity

# Wrapper function to calculate intensity from a single input sentence
def get_intensity(sentence: str) -> int:
    return calculate_intensity(sentence)

# Example usage
sentence = "I'm feeling very anxious about my health."
intensity = get_intensity(sentence)
print(f"Intensity Score: {intensity}")
