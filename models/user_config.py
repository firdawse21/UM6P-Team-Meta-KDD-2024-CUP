#from models.dummy_model import DummyModel

#UserModel = DummyModel

# Uncomment the lines below to use the Vanilla LLAMA baseline
# from models.vanilla_llama_baseline import ChatModel 
# UserModel = ChatModel


# Uncomment the lines below to use the RAG LLAMA baseline
from models.kg_rag_llama_baseline import RAG_KG_Model
UserModel = RAG_KG_Model