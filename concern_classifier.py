from transformers import pipeline
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

# Initialize the zero-shot classifier and define the categories
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
categories = [
    'Insomnia', 'Anxiety', 'Depression', 'Career Confusion',
    'Positive Outlook', 'Stress', 'Health Anxiety', 'Eating Disorder'
]

# Category to Score Mapping
category_scores = {
    'Positive Outlook': 8,
    'Career Confusion': 7,
    'Health Anxiety': 6,
    'Anxiety': 5,
    'Stress': 4,
    'Depression': 3,
    'Insomnia': 2,
    'Eating Disorder': 1
}
score_categories = {v: k for k, v in category_scores.items()}

# Function to predict the category and return scores
def predict_category(concern):
    result = classifier(concern, candidate_labels=categories, multi_label=False)
    category_scores = {label: score for label, score in zip(result['labels'], result['scores'])}
    return category_scores

# Example list of 100 sentences (replace this with actual user sentences)
sentences = [
    "I'm feeling very anxious about my health.",
    "I am extremely stressed about work.",
    "I've been having trouble sleeping lately.",
    "Feeling hopeful about my future.",
    "I feel excited about the new opportunities at work.",
    "Feeling low energy and unable to concentrate.",
    "I’m scared about the uncertainty in my career.",
    "I am having trouble focusing on my tasks.",
    "I feel motivated to make changes in my life.",
    "I am stressed about my relationship issues.",
    "I’ve been feeling so tired all the time.",
    "Having difficulty sleeping and feeling exhausted.",
    "I feel more confident about my work now.",
    "I worry about my mental health frequently.",
]

# Daily analysis
daily_categories = []
daily_scores = []

for sentence in sentences:
    scores = predict_category(sentence)
    max_prob_category = max(scores, key=scores.get)
    
    # Handle ties by rounding probabilities to 2 decimal places
    top_categories = [cat for cat, score in scores.items() if round(score, 2) == round(scores[max_prob_category], 2)]
    
    day_score = sum(category_scores[cat] for cat in top_categories) / len(top_categories)
    daily_categories.append(top_categories)
    daily_scores.append(day_score)

# Create DataFrame with Day, Category, and Score columns
df = pd.DataFrame({
    'Day': range(1, len(daily_scores) + 1),
    'Category': daily_categories,
    'Score': daily_scores
})

# Weekly and Monthly columns for plotting
df['Week'] = (df['Day'] - 1) // 7 + 1
df['Month'] = (df['Day'] - 1) // 28 + 1

# Plot Daily Emotional Score for First Week
first_week_df = df[df['Week'] == 1]
plt.figure(figsize=(10, 4))
plt.plot(first_week_df['Day'], first_week_df['Score'], marker='o', linestyle='-', color='purple')
plt.xticks(first_week_df['Day'])
plt.xlabel('Day')
plt.ylabel('Emotional Score')
plt.title('Daily Emotional Score (First Week)')
plt.grid(True)
plt.show()

# Plot Weekly Emotional Score (most frequent score for each week)
weekly_emotional_score = []
for week in df['Week'].unique():
    week_df = df[df['Week'] == week]
    mode_scores = week_df['Score'].mode().values
    weekly_emotional_score.append(mode_scores)

# Handle multiple modes by plotting each mode separately
plt.figure(figsize=(12, 4))
for i, mode in enumerate(weekly_emotional_score, start=1):
    if len(mode) == 1:
        plt.plot(i, mode[0], 'bo')  # Single mode point
    else:
        for m in mode:
            plt.plot(i, m, 'bo')  # Multiple points for ties

plt.xticks(range(1, len(weekly_emotional_score) + 1))
plt.xlabel('Week')
plt.ylabel('Most Frequent Emotional Score')
plt.title('Weekly Emotional Score')
plt.grid(True)
plt.show()

# Plot Monthly Emotional Score (most frequent score for each month)
monthly_emotional_score = []
for month in df['Month'].unique():
    month_df = df[df['Month'] == month]
    mode_scores = month_df['Score'].mode().values
    monthly_emotional_score.append(mode_scores)

# Handle multiple modes by plotting each mode separately
plt.figure(figsize=(12, 4))
for i, mode in enumerate(monthly_emotional_score, start=1):
    if len(mode) == 1:
        plt.plot(i, mode[0], 'go')  # Single mode point
    else:
        for m in mode:
            plt.plot(i, m, 'go')  # Multiple points for ties

plt.xticks(range(1, len(monthly_emotional_score) + 1))
plt.xlabel('Month')
plt.ylabel('Most Frequent Emotional Score')
plt.title('Monthly Emotional Score')
plt.grid(True)
plt.show()

# **Daily Analysis for First Week**
print("**Daily Analysis (First Week):**")
first_week_df = df[df['Week'] == 1]
for day in first_week_df['Day']:
    score = first_week_df[first_week_df['Day'] == day]['Score'].values[0]
    category = score_categories[score]
    print(f"Day {day}: Predominant emotional category is '{category}' with a score of {score}.")

# **Transitions within the First Week**
print("\n**Transitions within the First Week:**")
for i in range(1, len(first_week_df)):
    prev_score = first_week_df.iloc[i - 1]['Score']
    curr_score = first_week_df.iloc[i]['Score']
    prev_category = score_categories[prev_score]
    curr_category = score_categories[curr_score]
    
    if prev_category != curr_category:
        print(f"Transition from '{prev_category}' on Day {first_week_df.iloc[i - 1]['Day']} to '{curr_category}' on Day {first_week_df.iloc[i]['Day']}.")

# **Weekly Analysis**
print("\n**Weekly Analysis (All Weeks):**")
for week in df['Week'].unique():
    week_df = df[df['Week'] == week]
    week_mode_score = week_df['Score'].mode().values
    week_mode_categories = [score_categories[score] for score in week_mode_score]
    
    if week == 1:
        print(f"Week {week}: Most frequent category is {', '.join(week_mode_categories)}.")
    else:
        prev_week_df = df[df['Week'] == week - 1]
        prev_week_mode_score = prev_week_df['Score'].mode().values
        prev_week_mode_categories = [score_categories[score] for score in prev_week_mode_score]
        
        print(f"Week {week}: Most frequent category is {', '.join(week_mode_categories)}.")
        
        if set(prev_week_mode_categories) != set(week_mode_categories):
            print(f"Transition from Week {week - 1} to Week {week}: Predominant category shifted from {', '.join(prev_week_mode_categories)} to {', '.join(week_mode_categories)}.")
