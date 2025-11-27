from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from dotenv import load_dotenv
import os

# Load environment variables (for API keys)
load_dotenv()

# Set your OpenAI API key (make sure to set it in your environment variables)
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

def run_langchain_demo():
    # Initialize the language model
    llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
    
    # Define the prompt template
    template = """You are a helpful AI assistant. Answer the following question:
    
    Question: {question}
    
    Answer:"""
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful assistant that provides detailed answers."),
        HumanMessagePromptTemplate.from_template(template)
    ])
    
    # Create the chain
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Example questions
    questions = [
        "What is LangChain?",
        "How does LangChain help with LLM applications?",
        "What are the main components of LangChain?"
    ]
    
    print("LangChain Demo - Interactive Q&A\n" + "="*50 + "\n")
    
    for question in questions:
        # Get the response
        response = chain.invoke({"question": question})
        
        # Print the question and response
        print(f"Q: {question}")
        print(f"A: {response['text'].strip()}\n")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    run_langchain_demo()
