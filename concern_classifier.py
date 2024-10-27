import sys
from transformers import pipeline
import os

# Initialize the zero-shot classifier and define the categories
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
categories = [
    'Insomnia', 'Anxiety', 'Depression', 'Career Confusion',
    'Positive Outlook', 'Stress', 'Health Anxiety', 'Eating Disorder'
]

# Wrapper function to get the category with the highest probability
def classify_concern(concern):
    # Perform zero-shot classification
    result = classifier(concern, candidate_labels=categories, multi_label=False)
    
    # Extract the highest probability category
    max_category = result['labels'][0]
    max_score = result['scores'][0]
    
    return max_category, max_score

def main():
    input_file = sys.argv[1]

    input_file = os.path.join("Data", input_file, "extracted_concern.txt")
    output_file = os.path.join("Data", input_file, "predicted_category.txt")

    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                user_concern = line.strip()
                if user_concern:  # Check if the line is not empty
                    category, score = classify_concern(user_concern)
                    output_message = f"Concern: '{user_concern}' | Predicted Category: {category} (Score: {score:.2f})\n"
                    print(output_message.strip())  # Print to console
                    outfile.write(output_message)  # Write to file
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
