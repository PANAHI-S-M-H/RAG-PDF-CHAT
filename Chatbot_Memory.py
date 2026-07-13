import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser  

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
parser = StrOutputParser()  # 👈 اضافه شد


chain = llm | parser

chat_history = []

print("AI Chatbot is Ready! (Type 'exit' to stop)")
print("-" * 50)

while True:
    user_input = input("You: ")
    
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break
        
    chat_history.append(HumanMessage(content=user_input))
    
    try:
        
        response_text = chain.invoke(chat_history)
        
        print(f"AI: {response_text}")
        print("-" * 50)
        
        
        chat_history.append(AIMessage(content=response_text))
        
    except Exception as e:
        print(f"Error: {e}")