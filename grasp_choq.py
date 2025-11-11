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

# Define the prompt template for GRASP-ChoQ
prompt = ChatPromptTemplate.from_template("""
Tweet: {tweet}
Read the tweet above. The tweet has a political stance. It may express a view either in favor of the Awami League of Bangladesh or against it.
Detect the stance of the tweet with respect to the Awami League. Use reasoning based on political references or implied affiliations.

ASK QUESTIONS TO DETECT STANCE:
Q: Who is being criticized here?
A: Muhammad Yunus. Because the tweet uses "illegal" to describe him. Since Yunus is opposed to Sheikh Hasina (leader of Awami League), this implies support for Awami League.
Q: Which government is depicted in power in the tweet?
A: Muhammad Yunus’s government. As he is seen to follow Hasina, and is portrayed negatively, the stance favors the Awami League.

To aid your decision, general background knowledge about political figures and affiliations is provided.
TWEET_INFO: {tweet_info}
GENERAL_INFO: {relational_text}
EXTRA_INFO: {unstructured_data}
""")

def detect_stance_grasp_choq(tweet, tweet_info, relational_text, unstructured_data):
    """
    Detect the political stance of a tweet towards the Awami League using GRASP-ChoQ prompts.
    
    Parameters:
    tweet (str): The tweet text to analyze
    tweet_info (str): Additional information about the tweet
    relational_text (str): General political background information
    unstructured_data (str): Extra contextual information
    
    Returns:
    str: The stance expressed in the tweet (FAVOR or AGAINST)
    """
    # Invoke the language model with the prompt
    result = prompt | llm.invoke({
        "tweet": tweet,
        "tweet_info": tweet_info,
        "relational_text": relational_text,
        "unstructured_data": unstructured_data
    })
    
    # Extract and return the stance from the result
    stance = result.strip()  # Assuming the model returns a simple stance
    return stance

# Example usage
if __name__ == "__main__":
    tweet = "The head of the UN Human Rights Commission’s visit to Bangladesh raises concerns over the illegal Yunus government!"
    tweet_info = "Additional tweet information here."
    relational_text = "General political background information here."
    unstructured_data = "Extra contextual information here."
    stance = detect_stance_grasp_choq(tweet, tweet_info, relational_text, unstructured_data)
    print(f"The stance towards 'Awami League' is: {stance}")