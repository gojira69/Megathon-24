from transformers import pipeline

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment"
)

label_mapping = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

def get_sentiment_label(text):
    result = sentiment_pipeline(text)
    label = label_mapping[result[0]['label']]
    return label

example_text = "I am constantly worried these days."
predicted_label = get_sentiment_label(example_text)
print(f"Predicted Sentiment Label: {predicted_label}")
