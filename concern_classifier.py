from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

categories = [
    'Insomnia', 'Anxiety', 'Depression', 'Career Confusion',
    'Positive Outlook', 'Stress', 'Health Anxiety', 'Eating Disorder'
]

def predict_category(concern):
    result = classifier(concern, candidate_labels=categories, multi_label=False)
    category_scores = {label: score for label, score in zip(result['labels'], result['scores'])}
    return category_scores

concern = "very much jobless"
scores = predict_category(concern)
print("Category Scores:", scores)
