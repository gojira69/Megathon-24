from transformers import pipeline

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

# Example usage
user_concern = "I'm feeling extremely anxious about my job security and future prospects."
category, score = classify_concern(user_concern)
print(f"Predicted Category: {category} (Score: {score:.2f})")
