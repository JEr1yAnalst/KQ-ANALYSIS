import streamlit as st
import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Load the saved model
model_filename = 'bestmodel1.pkl'
pipeline = joblib.load(model_filename)

# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
sia = SentimentIntensityAnalyzer()

def preprocess_text(text):
    """Preprocess the review text."""
    words = word_tokenize(text.lower())
    words = [ps.stem(w) for w in words if w not in stop_words and w.isalpha()]
    return ' '.join(words)

def get_review_sentiment(review):
    """Get sentiment from review using sentiment analysis."""
    sentiment_score = sia.polarity_scores(review)['compound']
    if sentiment_score > 0.1:
        return 'positive'
    elif sentiment_score < -0.1:
        return 'negative'
    else:
        return 'neutral'

def get_sentiment_label(rating, review_sentiment):
    """Convert Rating and review sentiment into a final sentiment label."""
    if review_sentiment == 'positive':
        if rating >= 4:
            return 'positive'
        elif rating <= 2:
            return 'conflicting'
        else:
            return 'neutral'
    elif review_sentiment == 'negative':
        if rating <= 2:
            return 'negative'
        elif rating >= 4:
            return 'conflicting'
        else:
            return 'neutral'
    else:
        return 'neutral'

def predict_sentiment(review, rating):
    """Predict sentiment based on review and rating."""
    processed_review = preprocess_text(review)
    input_data = pd.DataFrame({
        'Processed_Review': [processed_review],
        'Rating': [rating]
    })
    sentiment = pipeline.predict(input_data)[0]
    return sentiment

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title='Sentiment Analysis App', layout='wide')
    
    st.sidebar.title('Sentiment Analysis App')
    
    with st.sidebar.expander("About", expanded=True):
        st.write("""
        This application uses a machine learning model to predict the sentiment of reviews based on the review text and rating. 
        The model flags conflicts between review sentiment and rating. Use this app to analyze the sentiment of your reviews and see how well they align with your ratings.
        """)
    
    st.title('Sentiment Analysis App')
    
    st.markdown("""
    <style>
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    .stButton > button {
        border-radius: 10px;
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.write("## Enter Review and Rating")
    st.write("Provide a review and rating to get the sentiment prediction. Both inputs will be used to determine the overall sentiment.")
    
    # Create user input fields
    review = st.text_area('Review:', help='Enter the review text here.', height=150)
    rating = st.slider('Rating:', min_value=0, max_value=5, value=3, help='Select the rating from 0 to 5.')
    
    # Add a section for example reviews
    with st.sidebar.expander("Example Reviews", expanded=True):
        st.write("**Example 1:**")
        st.write("Review: 'This product is fantastic!'")
        st.write("Rating: 5")
        st.write("Sentiment: Positive")
        
        st.write("**Example 2:**")
        st.write("Review: 'I had a terrible experience with this service.'")
        st.write("Rating: 1")
        st.write("Sentiment: Negative")
    
    # Display current input values
    st.write("### Current Input Values")
    st.write(f"**Review:** {review}")
    st.write(f"**Rating:** {rating}")
    
    # Button to trigger sentiment prediction
    if st.button('Predict Sentiment'):
        if review:
            review_sentiment = get_review_sentiment(review)
            final_sentiment = get_sentiment_label(rating, review_sentiment)
            st.write(f'### The predicted sentiment is: **{final_sentiment}**')
        else:
            st.write('Please enter a review.')
    
    # Prediction history (for demonstration purposes)
    if 'history' not in st.session_state:
        st.session_state.history = []

    if st.button('Add to History'):
        if review:
            review_sentiment = get_review_sentiment(review)
            final_sentiment = get_sentiment_label(rating, review_sentiment)
            st.session_state.history.append({
                'Review': review,
                'Rating': rating,
                'Sentiment': final_sentiment
            })
            st.success("Added to history!")

    if st.session_state.history:
        st.write("### Prediction History")
        history_df = pd.DataFrame(st.session_state.history)
        st.write(history_df)
        
        # Add download option for history
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="Download Prediction History",
            data=csv,
            file_name='prediction_history.csv',
            mime='text/csv'
        )
    
    # Model Information and Instructions
    st.write("### Model Information")
    st.write("This sentiment analysis model considers both the review text and the rating to predict sentiment. It handles conflicting cases where the review sentiment and rating do not align.")

if __name__ == '__main__':
    main()
