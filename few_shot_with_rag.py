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

# Define the prompt template for few-shot classification with context
prompt = ChatPromptTemplate.from_template("""
Task: Analyze the following tweets and determine if the author’s stance is in FAVOR of or AGAINST the specified target entity, using the tweet and context provided.
Target Entity: Awami League

Examples:
Tweet: "The country is moving forward under the leadership of Sheikh Hasina. #AwamiLeague"
Context: "Sheikh Hasina is the leader of the Awami League and has been praised for infrastructure development."
Stance: Favor

Tweet: "Corruption is rampant, and the government is not listening to the people. #Bangladesh"
Context: "The Awami League has been criticized in the media for alleged corruption and authoritarian practices."
Stance: Against

Tweet: "{tweet}"
Context: "{context}"
Stance:
""")

def classify_stance_with_context(tweet, context):
    """
    Classify the stance of a tweet towards the Awami League using few-shot examples and context.
    
    Parameters:
    tweet (str): The tweet text to analyze
    context (str): The context related to the tweet
    
    Returns:
    str: The stance expressed in the tweet (FAVOR or AGAINST)
    """
    # Invoke the language model with the prompt
    result = prompt | llm.invoke({"tweet": tweet, "context": context})
    
    # Extract and return the stance from the result
    stance = result.strip()  # Assuming the model returns a simple stance
    return stance

# Example usage
if __name__ == "__main__":
    new_tweet = "The economic policies are benefiting everyone. #AwamiLeague"
    retrieved_context = "Awami League's economic policies have been praised for boosting growth."
    stance = classify_stance_with_context(new_tweet, retrieved_context)
    print(f"The stance towards 'Awami League' is: {stance}")