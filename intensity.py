import re
import sys
import os

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

def main():
    user_name = sys.argv[1]
    input_file = os.path.join("Data", user_name, "therapist_notes.txt")

    output_file = os.path.join("Data", user_name, "intensity.txt")

    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                sentence = line.strip()
                if sentence:  # Check if the line is not empty
                    intensity = get_intensity(sentence)
                    output_message = f"Sentence: '{sentence}' | Intensity Score: {intensity}\n"
                    print(output_message.strip())  # Print to console
                    outfile.write(output_message)  # Write to file
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
