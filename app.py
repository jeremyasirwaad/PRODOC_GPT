import os
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI

# Set your OpenAI API key
key = ""
os.environ['OPENAI_API_KEY'] = key

# Load the document loader
loader = TextLoader('./docs/data.txt')

# Create the index
index = VectorstoreIndexCreator().from_loaders([loader])

# Initialize chat history as an empty list
chat_history = []

# Start a conversation loop
while True:
    # Get user input
    user_input = input("You: ")

    # Append user's message to chat history
    chat_history.append({'role': 'user', 'content': user_input})

    # Perform the query using the index and ChatOpenAI model
    result = index.query(str(chat_history), llm=ChatOpenAI())

    # Append AI's response to chat history
    chat_history.append({'role': 'AI', 'content': result})

    # Print AI's response
    print("AI:", result)
