from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
from api.config import *

# __all__ = ["FasterWhisper"]