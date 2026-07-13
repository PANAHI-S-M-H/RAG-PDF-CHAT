import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
parser = StrOutputParser()

template = """
You Are An Expert in Science Fiction World-Building And Lore.
Please Describe The Core Atmosphare And Aesthetic Of The {Universe_Name} Universe in Exactly Two Short Paragraphs.
"""

prompt_template = PromptTemplate.from_template(template)

chain = prompt_template | llm | parser

print("Generating World-Building Lore...")
print("-" * 60)

try:
    response = chain.invoke({"Universe_Name": "Ghost In The Shell"})

    print(response)
    print("-" * 60)

except Exception as e:
    print(f"Error Occurred : {e}")

