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

# Define the prompt template for zero-shot classification
prompt = ChatPromptTemplate.from_template("""
Task: Read the tweet and determine whether it expresses a stance in FAVOR of or AGAINST the specified target entity.
Target Entity: {entity}
Tweet: {tweet}
""")

def classify_stance(tweet, entity="Awami League"):
    """
    Classify the stance of a tweet towards the specified target entity.
    
    Parameters:
    tweet (str): The tweet text to analyze
    entity (str): The target entity to determine stance towards
    
    Returns:
    str: The stance expressed in the tweet (FAVOR or AGAINST)
    """
    # Invoke the language model with the prompt
    result = prompt | llm.invoke({"entity": entity, "tweet": tweet})
    
    # Extract and return the stance from the result
    stance = result.strip()  # Assuming the model returns a simple stance
    return stance

# Example usage
if __name__ == "__main__":
    tweet = "I think Awami League is doing a great job!"
    stance = classify_stance(tweet)
    print(f"The stance towards 'Awami League' is: {stance}")