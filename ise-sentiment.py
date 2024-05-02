import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.special import softmax

nltk.download('vader_lexicon')

st.title('Sentiment Analysis')

user_input = st.text_area("Enter a sentence")

if user_input:
    # VADER Sentiment Analysis
    sia = SentimentIntensityAnalyzer()
    vader_result = sia.polarity_scores(user_input)
    st.write("VADER Sentiment Analysis:")
    st.write(vader_result)

    # RoBERTa Sentiment Analysis
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

    def polarity_scores_roberta(text):
        encoded_text = tokenizer(text, return_tensors='pt')
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        return {'roberta_neg': scores[0], 'roberta_neu': scores[1], 'roberta_pos': scores[2]}

    roberta_result = polarity_scores_roberta(user_input)
    st.write("RoBERTa Sentiment Analysis:")
    st.write(roberta_result)

    # Transformer Pipeline Sentiment Analysis
    sent_pipeline = pipeline("sentiment-analysis")
    pipeline_result = sent_pipeline(user_input)
    st.write("Transformer Pipeline Sentiment Analysis:")
    st.write(pipeline_result)

    # Create a DataFrame for visualization
    data = {'Model': ['VADER', 'RoBERTa', 'Pipeline'],
            'Negative': [vader_result['neg'], roberta_result['roberta_neg'], pipeline_result[0]['score']],
            'Neutral': [vader_result['neu'], roberta_result['roberta_neu'], 0],
            'Positive': [vader_result['pos'], roberta_result['roberta_pos'], 1 - pipeline_result[0]['score']]}
    df = pd.DataFrame(data)

    # Visualization
    st.write("Sentiment Analysis Results:")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    sns.barplot(data=df, x='Model', y='Negative', ax=axs[0])
    sns.barplot(data=df, x='Model', y='Neutral', ax=axs[1])
    sns.barplot(data=df, x='Model', y='Positive', ax=axs[2])
    axs[0].set_title('Negative')
    axs[1].set_title('Neutral')
    axs[2].set_title('Positive')
    plt.tight_layout()
    st.pyplot(fig)