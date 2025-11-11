from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Initialize clients
client = OpenAI(
    base_url="https://api.openrouter.ai/v1",  
    api_key=os.getenv("OPENROUTER_API_KEY")   
)

# Initialize the language model
llm = ChatOpenAI(client=client, model="gpt-4o")  # Ensure the client is passed correctly

# Define the prompt template for few-shot classification
prompt = ChatPromptTemplate.from_template("""
Task: Analyze the following tweets and determine if the author’s stance is in FAVOR of or AGAINST the specified target entity.
Target Entity: Awami League

Examples:
Tweet: "The country is moving forward under the leadership of Sheikh Hasina. #AwamiLeague"
Stance: Favor

Tweet: "Corruption is rampant, and the government is not listening to the people. #Bangladesh"
Stance: Against

Tweet: "{tweet}"
Stance:
""")

def classify_stance_few_shot(tweet):
    """
    Classify the stance of a tweet towards the Awami League using few-shot examples.
    
    Parameters:
    tweet (str): The tweet text to analyze
    
    Returns:
    str: The stance expressed in the tweet (FAVOR or AGAINST)
    """
    # Invoke the language model with the prompt
    result = prompt | llm.invoke({"tweet": tweet})
    
    # Extract and return the stance from the result
    stance = result.strip()  # Assuming the model returns a simple stance
    return stance

# Example usage
if __name__ == "__main__":
    new_tweet = "The economic policies are benefiting everyone. #AwamiLeague"
    stance = classify_stance_few_shot(new_tweet)
    print(f"The stance towards 'Awami League' is: {stance}")