
# Chat with Max


This application allows you to chat to a website , pdf or word document it uses tavily for WEB searching, redis cache to keep conversation hsitory and chat gpt for normal chat integration 
When the session starts a session id is created and that is stored in redis 
This application is written in Python and flask and was created to learn the 

## Requirements
Python 3.12
Requests library
Langchain community package

## Api Keys (.env file)
 you would need to have the following 
 - openai  api key
 - Tavily key 
 - upstash url and key 


Tavily Search API: https://app.tavily.com/home
uppstash.com

if you can use Langsmith: https://www.langchain.com/langsmith for debugging 
## Setup 
Create an environment 
python -m venv venv {change the name if you want more personolized environment }
Activate Envionment 
.\venv\scripts\activate


### Explanation of the [`.env`]

The [`.env`] "Langchain Chat with MAX\.env") file is used to store environment variables that are required for the application to run. These variables can include API keys, database connection strings, and other configuration settings that you do not want to hard-code into your application.

#### Example [`.env`] "Langchain Chat with MAX\.env") File


OPENAI_API_KEY=""
TAVILY_API_KEY=""
URL = ""  // upstash redis url 
TOKEN =""// upstash redis token   


you would need a tavily api key
## install Dependancies 
pip install langchain langchain-openai
pip install python-dotenv

pip install langchain_community
pip install beautifulsoup4
pip install faiss-cpu
pip install upstash_redis
pip install flask langchain-openai langchain-community
get openai key 
pip install python-docx
pip install PyPDF2
from langchain.schema import Document  
## run 
py app.py


# 

