import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
from collections import Counter
import spacy

# Download required NLTK data
nltk.download('stopwords')

# Load the spaCy model for NER
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000  # Increase the max_length limit

def generate_wordcloud_and_csv(excel_file, column_name, output_csv):
    """
    Generate a word cloud from an Excel column while removing stop words,
    and create a CSV file with the top 100 words, their frequency, and NER tags.
    
    Parameters:
    excel_file (str): Path to the Excel file
    column_name (str): Name of the column containing text data
    output_csv (str): Path to the output CSV file
    """
    # Read the Excel file
    df = pd.read_excel(excel_file)
    
    # Combine all text from the column into a single string
    text = ' '.join(df[column_name].astype(str).values)
    
    # Get English stop words
    stop_words = set(stopwords.words('english'))
    
    # Tokenize and filter out stop words
    words = [word for word in text.split() if word.lower() not in stop_words]
    
    # Count word frequencies
    word_freq = Counter(words)
    
    # Get the top 100 words
    top_words = word_freq.most_common(100)
    
    # Perform NER on the text
    doc = nlp(text)
    ner_tags = {ent.text: ent.label_ for ent in doc.ents}
    
    # Prepare data for CSV
    csv_data = []
    for word, freq in top_words:
        ner_tag = ner_tags.get(word, "O")  # "O" for words without a specific NER tag
        csv_data.append({"Word": word, "Frequency": freq, "NER": ner_tag})
    
    # Create a DataFrame and save to CSV
    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv(output_csv, index=False)
    
    # Create and configure the WordCloud object
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stop_words,
        min_font_size=10,
        max_font_size=150,
        random_state=42
    )
    
    # Generate the word cloud
    wordcloud.generate(text)
    
    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()
    
    # Save the word cloud to a file
    wordcloud.to_file('wordcloud.png')

# Example usage
if __name__ == "__main__":
    # Replace these with your actual file path and column name
    excel_file = r"F:\EUCLIDO\Tasks\selenium-twitter-scraper-master\tweets\New folder\all\embedding\all_embedding.xlsx"
    column_name = "translation"
    output_csv = r"F:\EUCLIDO\Tasks\_____self\semeval-humayun\grasp-choq2\stop_words.csv"
    
    generate_wordcloud_and_csv(excel_file, column_name, output_csv)