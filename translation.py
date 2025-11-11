import pandas as pd
import time
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize clients
client = OpenAI(
    base_url="https://api.openrouter.ai/v1",  
    api_key=os.getenv("OPENROUTER_API_KEY")   
)

def load_tweet_data(file_path):
    """
    Load tweet data from Excel or CSV file
    """
    if file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Use .xlsx or .csv")

def translate_tweet(tweet, model="gpt-4o"):
    """
    Translate tweet to English using LLM
    """
    translation_prompt = f"""
    Task: Translate the following tweet into English. If the tweet is already in English, output the original tweet. Do NOT translate proper nouns (e.g., names of people, organizations, specific places).
    
    Examples:
    Tweet (Bangla): প্রধানমন্ত্রী শেখ হাসিনা আজ একটি নতুন প্রকল্প উদ্বোধন করবেন।
    Translated Tweet (English): Prime Minister Sheikh Hasina will inaugurate a new project today.
    
    Tweet (English): Just attended the Google I/O conference.
    Translated Tweet (English): Just attended the Google I/O conference.
    
    Tweet: {tweet}
    Translated Tweet (English):
    """
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that translates tweets to English while preserving proper nouns. If the tweet is already in English, you output the original tweet."},
            {"role": "user", "content": translation_prompt}
        ],
        temperature=0
    )
    translated_tweet = response.choices[0].message.content
    return translated_tweet

# def translate_tweet_with_examples(tweet, model="gpt-4o"):
#     """
#     Translate tweet into English, preserving proper nouns.
#     If the tweet is already in English, output the original tweet.
#     """
#     translation_prompt = f"""
#     Task: Translate the following tweet into English. If the tweet is already in English, output the original tweet. Do NOT translate proper nouns (e.g., names of people, organizations, specific places).
#     
#     Examples:
#     Tweet (Non-English): [Example tweet in a non-English language with proper nouns]
#     Translated Tweet (English): [Correct English translation, preserving proper nouns]
#     
#     Tweet (English): [Example English tweet]
#     Translated Tweet (English): [Same English tweet]
#     
#     Tweet (Non-English): [Another example tweet in a different non-English language]
#     Translated Tweet (English): [Correct English translation, preserving proper nouns]
#     
#     Tweet: {tweet}
#     Translated Tweet (English):
#     """
#     
#     response = client.chat.completions.create(
#         model=model,
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant that translates tweets to English while preserving proper nouns. If the tweet is already in English, you output the original tweet."},
#             {"role": "user", "content": translation_prompt}
#         ],
#         temperature=0
#     )
#     translated_tweet = response.choices[0].message.content
#     return translated_tweet

def translate_dataset(input_file, output_file):
    """
    Translate tweets from the Content column to English
    """
    df = load_tweet_data(input_file)
    print(f"Dataset size: {len(df)} tweets")
    
    # Translate tweets
    for index, row in df.iterrows():
        if index % 10 == 0:
            print(f"Processing tweet {index}/{len(df)}")
            
        if 'Content' in df.columns and not pd.isna(row['Content']):
            try:
                translation = translate_tweet(row['Content'])
                df.at[index, 'translation'] = translation
                
            except Exception as e:
                print(f"Error processing row {index}: {e}")
                continue
                
        time.sleep(1)  # Rate limiting
    
    # Save processed data
    df.to_excel(output_file, index=False)
    print(f"Translation complete. Dataset saved to '{output_file}'")
    return df

if __name__ == "__main__":
    # Process the BPDisC dataset
    input_file = "BPDisC_with_stance.xlsx"
    output_file = "BPDisC_translated.xlsx"
    
    print("Translating tweets from BPDisC Dataset")
    df = translate_dataset(input_file, output_file)