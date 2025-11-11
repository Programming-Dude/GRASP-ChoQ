import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

def preprocess_twitter_dataset(tweets_df):
    """
    Preprocesses Twitter dataset according to the following criteria:
    1. Excludes image tweets
    2. Excludes tweets from users whose bios don't indicate political stance
    3. Removes tweets with fewer than 5 words
    
    Args:
        tweets_df: DataFrame containing Twitter data with columns:
                  - 'tweet_text': the content of the tweet
                  - 'has_image': boolean indicating if tweet has image
                  - 'user_bio': the user's Twitter bio text
    
    Returns:
        Preprocessed DataFrame with filtered tweets
    """
    print(f"Original dataset size: {len(tweets_df)}")
    
    # 1. Filter out tweets with images
    if 'has_image' in tweets_df.columns:
        tweets_df = tweets_df[~tweets_df['has_image']]
        print(f"After removing image tweets: {len(tweets_df)}")
    else:
        print("No 'has_image' column found, skipping image filtering")
    
    # 2. Filter users based on political stance in bio
    # Define keywords indicating political stance
    political_keywords = [
        
        # Bangladesh political keywords
        'awami league', 'bangladesh awami league', 'al', 'bnp', 
        'bangladesh nationalist party', 'jatiya party', 'jp',
        'jamaat-e-islami', 'jamaat', 'jatiyo party',
        'sheikh hasina', 'hasina', 'khaleda zia', 'zia',
        'pro-awami', 'pro-bnp', 'BAL', 'bangladesh awami',
        'nationalist', '#awamileague', '#bnp', '#bangladeshpolitics'
    ]
    
    # Function to check if bio contains political keywords
    def has_political_stance(bio):
        if pd.isna(bio):
            return False
        bio = str(bio).lower()
        return any(keyword in bio for keyword in political_keywords)
    
    # Apply filter for users with political stance if user_bio column exists
    if 'user_bio' in tweets_df.columns:
        tweets_df = tweets_df[tweets_df['user_bio'].apply(has_political_stance)]
        print(f"After filtering users without political stance in bio: {len(tweets_df)}")
    elif 'stance' in tweets_df.columns:
        # If the dataset already has a stance column, use that instead
        tweets_df = tweets_df[~tweets_df['stance'].isna()]
        print(f"Using existing stance column. After filtering: {len(tweets_df)}")
    else:
        print("No 'user_bio' or 'stance' column found, skipping political stance filtering")
    
    # 3. Remove tweets with fewer than 5 words
    # Identify the text column (could be 'tweet_text', 'text', 'tweet', or 'Content')
    text_column = None
    for possible_column in ['tweet_text', 'text', 'tweet', 'Content']:
        if possible_column in tweets_df.columns:
            text_column = possible_column
            break
    
    if text_column:
        def count_words(text):
            if pd.isna(text):
                return 0
            # Tokenize text into words
            words = word_tokenize(str(text))
            return len(words)
        
        tweets_df = tweets_df[tweets_df[text_column].apply(count_words) >= 5]
        print(f"After removing tweets with fewer than 5 words: {len(tweets_df)}")
    else:
        print("No text column found for word count filtering")
    
    return tweets_df

# Example usage
if __name__ == "__main__":
    try:
        # Load the BPDisC_with_stance.xlsx dataset
        file_path = 'BPDisC_with_stance.xlsx'
        tweets_df = pd.read_excel(file_path)
        
        # Display column names to help with debugging
        print(f"Columns in the dataset: {tweets_df.columns.tolist()}")
        
        # Apply preprocessing
        processed_df = preprocess_twitter_dataset(tweets_df)
        
        # Save the preprocessed dataset
        output_file = 'preprocessed_BPDisC_dataset.xlsx'
        processed_df.to_excel(output_file, index=False)
        print(f"Preprocessing complete. Dataset saved to '{output_file}'")
        
    except FileNotFoundError:
        print(f"File not found. Please ensure 'BPDisC_with_stance.xlsx' exists in the current directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")