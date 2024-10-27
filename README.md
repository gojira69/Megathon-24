# Megathon-24: MyTak

# Mental Health Support Platform

A Flask-based web application for mental health professionals to manage patient data, process therapy sessions, and analyze mental health indicators. This platform combines audio processing, natural language processing, and mental health analysis tools.

## Current Components

### 1. Web Application (`app.py`)
- Flask-based web interface for patient data management
- Features:
  - Landing page and dashboard
  - File upload system for therapist notes and audio files
  - Patient-specific data organization
  - Automatic processing of uploaded files

### 2. Temporal Analysis Module (`temporal_shifts.py`)

This module analyzes temporal patterns in emotional expressions using zero-shot classification. Features include:

- Zero-shot classification using Facebook's BART model
- Analysis of emotional patterns across different time scales:
  - Daily analysis
  - Weekly trends
  - Monthly patterns
- Categorization into 8 emotional states:
  - Insomnia
  - Anxiety
  - Depression
  - Career Confusion
  - Positive Outlook
  - Stress
  - Health Anxiety
  - Eating Disorder

Outputs:
- Generates visualization plots saved as PNG files:
  - `daily_emotional_score_first_week.png`
  - `weekly_emotional_score.png`
  - `monthly_emotional_score.png`
- Detailed analysis report in `output_analysis.txt`

### 3. Concern Classification (`concern_classifier.py`)
- Zero-shot classification for mental health concerns
- Categories include:
  - Insomnia
  - Anxiety
  - Depression
  - Career Confusion
  - Positive Outlook
  - Stress
  - Health Anxiety
  - Eating Disorder

### 4. Mental Health Text Analysis (`extract.py`)
- Advanced text analysis for mental health indicators
- Features:
  - Emotion and state detection
  - Context analysis
  - Pattern recognition
  - Synonym expansion using WordNet
  - Behavioral pattern analysis

### 5. Intensity Analysis (`intensity.py`)
- Measures the intensity of expressed mental health concerns
- Features:
  - Quantifier-based scoring system
  - Base intensity calculation
  - Contextual intensity adjustment




### 6. Sentiment Analysis Module (`polarity.py`)
This module implements sentiment analysis using the Hugging Face transformers library with the CardiffNLP Twitter RoBERTa model. Features include:

- Sentiment classification into three categories: Negative, Neutral, and Positive
- Pre-configured pipeline using the `cardiffnlp/twitter-roberta-base-sentiment` model
- Simple interface through the `get_sentiment_label()` function
- Example usage demonstration included in the script


## Dependencies
```
flask
librosa
transformers
pydub
nltk
pandas
```

## Setup Instructions

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Download NLTK data:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

3. Create necessary directories:
```bash
mkdir Data
```

[Note: Additional components and setup instructions will be added in Part 2]

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Access the web interface at `http://localhost:5000`

3. Upload patient data:
   - Navigate to the dashboard
   - Upload audio files and therapist notes
   - System will automatically process and analyze the data

## Current Features

### Audio Processing
```python
# Example usage
from audio_text import transcribe_audio_file
transcribe_audio_file("patient_recording.mp3", "transcription.txt")
```

### Concern Classification
```python
# Example usage
from concern_classifier import classify_concern
category, score = classify_concern("I'm feeling extremely anxious about my job security")
```

### Mental Health Text Analysis
```python
# Example usage
from extract import MentalHealthExtractor
extractor = MentalHealthExtractor()
concern = extractor.extract_concern("I've been feeling very depressed lately")
```

### Intensity Analysis
```python
# Example usage
from intensity import get_intensity
intensity_score = get_intensity("I'm feeling very anxious about my health")
```

## Note
This is Part 1 of the documentation. Additional components and their documentation will be added in Part 2.

## Security Considerations
- Ensure proper access controls are implemented
- Keep patient data confidential and encrypted
- Regular backups of the Data directory
- Monitor system access logs

## Additional Components

### Spotify Data Processing Script (`script.py`)

This script processes Spotify listening history data and analyzes user listening patterns. Key functionalities include:

- Data extraction from Spotify data export ZIP files
- Streaming history processing and filtering
- Audio feature retrieval using the Spotify API
- Temporal pattern analysis for various music features
- Generation of diurnal, daily, and monthly listening patterns

Key features:
- Handles invalid timestamp correction
- Filters out short songs (< 30 seconds)
- Retrieves detailed audio features for each track
- Creates visualizations for listening patterns
- Saves processed data in CSV format

Required environment variables:
```
SPOTIFY_CLIENT_ID
SPOTIFY_CLIENT_SECRET
```


### Audio Processing (`audio_text.py`)

- Transcribes therapy session recordings
- Features:
  - MP3 to WAV conversion using `pydub`
  - Speech recognition using OpenAI's Whisper model
  - Automatic transcription saving

## Dependencies

Additional dependencies required for these components:

```
transformers
spotipy
pandas
matplotlib
tqdm
zipfile
ast
logging
filecmp
```

## Usage

### Sentiment Analysis

```python
from polarity import get_sentiment_label

text = "I am feeling great today!"
sentiment = get_sentiment_label(text)
print(f"Sentiment: {sentiment}")
```

### Spotify Data Processing

```python
python script.py <username>
```

Replace `<username>` with the target user's identifier. Ensure the Spotify data export is placed in `Data/<username>/my_spotify_data.zip`.

### Temporal Analysis

1. Prepare your input data in a CSV file named `input_sentences.csv` with a column named "Sentence"
2. Run the analysis:
```python
python temporal_shifts.py
```

## Output Files

The scripts generate several output files:

- `filtered_streaming_history.csv`: Processed Spotify listening history
- `diurnal_patterns.csv`: Hour-by-hour listening patterns
- `daywise_week_patterns.csv`: Day-of-week patterns
- `daywise_month_patterns.csv`: Day-of-month patterns
- `output_analysis.txt`: Detailed emotional pattern analysis
- Various PNG files containing visualizations

## Notes

- Ensure all required API credentials are properly configured
- The Spotify script requires a valid Spotify data export in ZIP format
- Large datasets may require significant processing time
- Consider memory requirements when processing large music libraries