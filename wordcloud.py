import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("Data1.csv")
if "survey_response_feedback" in df.columns:
        from wordcloud import WordCloud

        feedback_text = ' '.join(df["survey_response_feedback"].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(feedback_text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Feedback Word Cloud')
        plt.show()
