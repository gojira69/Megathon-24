from transformers import pipeline

# Load sentiment analysis pipeline with the cardiffnlp model
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment"
)

# Define the mapping for human-readable labels
label_mapping = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

# Test sentences
texts = [
    "I am constantly worried these days.",
    "I’m trying, but I’m still constantly worried.",
    "I am worried about health these days.",
    "Every day I’m happy and excited.",
    "I feel happy and excited lately.",
    "Sometimes, I think I'm feeling very low.",
    "My mind feels like it’s can't sleep well.",
    "It’s a struggle, I’m constantly worried.",
    "Lately, I’ve been feeling very anxious.",
    "The food was okay.",
]

# Run sentiment analysis
results = sentiment_pipeline(texts)

# Print results with mapped labels
for text, result in zip(texts, results):
    sentiment = label_mapping[result['label']]
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment} (Original: {result['label']}), Confidence: {result['score']:.4f}\n")
