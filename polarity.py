from transformers import pipeline
import sys
import os

user_name = sys.argv[1]

print(f"User Name: {user_name}")

file_path = os.path.join("Data", user_name, "therapist_notes.txt")

try:
    with open(file_path, 'r') as file:
        user_input = file.read()  # Read the entire file content into a string
except FileNotFoundError:
    print(f"The file {file_path} does not exist.")
    sys.exit(1)  # Exit if the file is not found
except Exception as e:
    print(f"An error occurred while reading the file: {e}")
    sys.exit(1)  # Exit if any other error occurs

# Initialize the sentiment analysis pipeline
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment"
)

# Define the mapping for sentiment labels
label_mapping = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

def get_sentiment_label(text):
    """Get the sentiment label for the input text."""
    result = sentiment_pipeline(text)
    label = label_mapping[result[0]['label']]
    return label

predicted_label = get_sentiment_label(user_input)
print(f"Predicted Sentiment Label: {predicted_label}")

output_file_path = os.path.join("Data", user_name, "polarity_label.txt")
try:
    with open(output_file_path, 'w') as output_file:
        output_file.write(predicted_label)
    print(f"Predicted sentiment label saved to {output_file_path}")
except Exception as e:
    print(f"An error occurred while writing to the file: {e}")