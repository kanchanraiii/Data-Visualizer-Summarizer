import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from wordcloud import WordCloud
from collections import Counter
import string

# Download NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

# Standardize designation function
def standardize_designation(designation):
    if pd.isna(designation):
        return 'Unknown'
    designation = designation.strip().lower()
    if 'student' in designation:
        return 'Student'
    elif 'ar' in designation or 'vr' in designation or 'xr' in designation:
        return 'AR/VR Developer'
    elif 'sde' in designation or 'developer' in designation or 'software engineer' in designation:
        return 'Software Developer'
    elif 'ui' in designation or 'ux' in designation:
        return 'UI/UX Designer'
    elif 'founder' in designation:
        return 'Founder'
    else:
        return 'Other'

# Define the visualize function
def visualize(df):
    # Standardize the designations
    df['Standardized Designation'] = df['What is your designation?'].str.lower().apply(standardize_designation)
    
    # Plot 1: People who joined event after registration and Survey Response Rating
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 6))
    axes1[0].hist(df["has_joined_event"], bins=5, edgecolor="black")
    axes1[0].set_xlabel("People who joined event after registration")
    axes1[0].set_ylabel("Frequency")
    axes1[0].set_title("Event Participation")

    survey_rating = df["survey_response_rating"].value_counts()
    axes1[1].pie(survey_rating, labels=survey_rating.index, autopct=lambda p: '{:.0f}'.format(p * sum(survey_rating) / 100), startangle=140)
    axes1[1].set_title("Survey Response Rating")

    # Adjust layout
    fig1.tight_layout()

    # Display the first set of plots
    st.pyplot(fig1)

    # Plot 2: Standardized Designations and Knowledge of AR/VR
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6))
    sns.countplot(data=df, x='Standardized Designation', ax=axes2[0])
    axes2[0].set_title('Standardized Designations')
    axes2[0].set_xlabel('Designation')
    axes2[0].set_ylabel('Count')
    axes2[0].tick_params(axis='x', rotation=45)
    axes2[0].grid(axis='y')

    knowledge_ar_vr = df["How much do you know about AR/VR?"]
    value_counts = knowledge_ar_vr.value_counts().sort_index()
    axes2[1].bar(value_counts.index, value_counts.values, edgecolor="black")
    axes2[1].set_xlabel("Domain Knowledge Level")
    axes2[1].set_ylabel("Frequency")
    axes2[1].set_title("Knowledge of AR/VR")

    # Adjust layout
    fig2.tight_layout()

    # Display the second set of plots
    st.pyplot(fig2)

    # Plot 3: Participation by Designation and Count of students who joined the event and provided survey responses
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 6))
    
    # Participation by Designation
    participation_by_designation = df[df["has_joined_event"] == "Yes"]["Standardized Designation"].value_counts()
    sns.barplot(x=participation_by_designation.index, y=participation_by_designation.values, palette="viridis", ax=axes3[0])
    axes3[0].set_title('Event Participation by Designation')
    axes3[0].set_xlabel('Designation')
    axes3[0].set_ylabel('Count')
    axes3[0].tick_params(axis='x', rotation=85)

    # Count of students who joined the event and provided survey responses
    joined_event = df[df['has_joined_event'] == 'Yes']
    rating_count = joined_event['survey_response_rating'].notna().sum()
    feedback_count = joined_event['survey_response_feedback'].notna().sum()

    sns.barplot(x=['Rating', 'Feedback'], y=[rating_count, feedback_count], ax=axes3[1])
    axes3[1].set_title('Count of Students Who Joined the Event and Provided Survey Responses')
    axes3[1].set_xlabel('Response Type')
    axes3[1].set_ylabel('Count')

    # Adjust layout
    fig3.tight_layout()

    # Display the third set of plots
    st.pyplot(fig3)

    # Word cloud of feedback if available
    if "survey_response_feedback" in df.columns:
        feedback_text = ' '.join(df["survey_response_feedback"].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(feedback_text)

        fig4, ax4 = plt.subplots(figsize=(8, 4))
        ax4.imshow(wordcloud, interpolation='bilinear')
        ax4.axis('off')
        ax4.set_title('Feedback Word Cloud')

        # Display the word cloud
        st.pyplot(fig4)

# Define the summarize_text function
def summarize_text(df):
    # Standardize the designations to lowercase for consistency
    df['Standardized Designation'] = df['What is your designation?'].str.lower().apply(standardize_designation)
    
    # Feedback summary
    if "survey_response_feedback" in df.columns:
        feedback_text = ' '.join(df["survey_response_feedback"].dropna())
        feedback_summary = summarize_feedback(feedback_text)
    else:
        feedback_summary = "No feedback provided."

    # Summarize knowledge of AR/VR
    ar_vr_levels = df["How much do you know about AR/VR?"].dropna().str.lower().value_counts()
    
    # Summarize designations with percentages
    designation_summary = df['Standardized Designation'].value_counts(normalize=True) * 100

    # Percentage of people who joined event after registration
    join_event_percentage = df['has_joined_event'].value_counts(normalize=True) * 100
    
    # Additional insights
    avg_survey_rating = df['survey_response_rating'].mean()
    common_feedback_words = summarize_feedback(feedback_text) if feedback_text else "No common feedback words available."

    # Generalized improvements section based on AR/VR knowledge levels
    if 'beginner' in ar_vr_levels.index:
        if ar_vr_levels['beginner'] < 50:
            improvements = "Suggestion: Increase efforts to attract more beginners to start their journey in AR/VR."
    if 'intermediate' in ar_vr_levels.index:
        if ar_vr_levels['intermediate'] < 50:
            improvements = "Suggestion: Offer more intermediate level AR/VR sessions to cater to growing intermediate participants."
    if 'advanced' in ar_vr_levels.index:
        if ar_vr_levels['advanced'] < 50:
            improvements = "Suggestion: Provide advanced AR/VR sessions to meet the demands of advanced participants."

    summary_paragraph = f"Most of the attendees are {designation_summary.idxmax()}s with {ar_vr_levels.idxmax()}-level knowledge of AR/VR. " \
                        f"Approximately {join_event_percentage.get('Yes', 0):.2f}% of the registered individuals joined the event. " \
                        f"The survey responses indicate a predominant rating of {df['survey_response_rating'].mode()[0]}. " \
                        f"The average survey rating is {avg_survey_rating:.2f}.\n\n{improvements}"

    return summary_paragraph, feedback_summary, ar_vr_levels, designation_summary, join_event_percentage


# Define the summarize_feedback function
def summarize_feedback(feedback_text):
    # Here you can implement your logic to summarize feedback text
    # For example, extracting key insights or generating a summary sentence
    # For demonstration, let's create a summary sentence of key aspects
    word_freq = Counter(nltk.word_tokenize(feedback_text.lower()))
    top_words = word_freq.most_common(10)  # Example: Top 10 most frequent words

    summary_sentence = f"The feedback highlights concerns about {', '.join([word for word, _ in top_words])}."

    return summary_sentence

# Streamlit app
st.set_page_config(layout="wide")  # Set the layout to wide

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Visualizer", "Summarizer"])

# Main content based on button clicked
if page == "Home":
    st.title("Welcome to the Summarizer and Visualizer App")
    st.write("This app helps you to visualize and summarize event data easily and efficiently.")
    st.write("")
    st.image("img.jpg", width = 700)
    st.write("")
    st.write("Navigate to the sidebar to get started with the Visualizer and Summariser services.")

elif page == "Visualizer":
    st.title("Data Visualizer")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("## Data Preview")
        st.dataframe(df.head())

        # Add a button to trigger visualization
        if st.button('Visualize Data'):
            st.write("## Visualizations")
            visualize(df)

elif page == "Summarizer":
    st.title("Data Summarizer")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("## Data Preview")
        st.dataframe(df.head())

        # Add a button to trigger summarization
        if st.button('Summarize Data'):
            

            # Call summarize_text function
            summary_paragraph, feedback_summary, ar_vr_levels, designation_summary, join_event_percentage = summarize_text(df)
            
            # Display SUmmary Paragraph
            st.write("## Summary Paragraph")
            if isinstance(summary_paragraph, str):
                st.write(summary_paragraph)
            
            # Display Feedback summary
            st.write("### Feedback Summary")
            if isinstance(feedback_summary, str):
                st.write(feedback_summary)

            # Display a paragraph summarizing key statistics
            st.write(f"The summary of key statistics including knowledge levels and participation percentages is as follows:")

            # Display tables for AR/VR knowledge levels, Designation summary, and Event participation
            st.write("## Tables")

            # Display AR/VR knowledge levels
            st.write("### AR/VR Knowledge Levels")
            st.write(ar_vr_levels)

            # Display Designation summary with percentages
            st.write("### Designation Summary")
            st.write(designation_summary)

            # Display Percentage of people who joined event after registration
            st.write("### Percentage of People Who Joined Event After Registration")
            st.write(join_event_percentage)