import os
from collections import defaultdict
from typing import Any, Dict, List
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np 
import ray
import torch
import vllm
from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
from sentence_transformers import CrossEncoder
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from langchain_community.embeddings.sentence_transformer import (
  SentenceTransformerEmbeddings,
)

CRAG_MOCK_API_URL = os.getenv("CRAG_MOCK_API_URL", "http://localhost:8000")

NUM_CONTEXT_SENTENCES = 20
MAX_CONTEXT_SENTENCE_LENGTH = 1000
MAX_CONTEXT_REFERENCES_LENGTH = 4000

AICROWD_SUBMISSION_BATCH_SIZE = 1
VLLM_TENSOR_PARALLEL_SIZE = 1
VLLM_GPU_MEMORY_UTILIZATION = 0.85
SENTENTENCE_TRANSFORMER_BATCH_SIZE = 128

class ChunkExtractor:
    def __init__(self):
        self.initialize_models()

    def initialize_models(self):
      self.sentence_model= SentenceTransformerEmbeddings(model_name='models/sentence-transformers/all-MiniLM-L12-v2')
      self.semantic_chunker = SemanticChunker(self.sentence_model, breakpoint_threshold_type="percentile")
    
    @ray.remote
    def _extract_chunks(self, interaction_id, html_source):
        soup = BeautifulSoup(html_source, "lxml")
        text = soup.get_text(" ", strip=True)

        if not text:
            return interaction_id, [""]

        _, offsets = text_to_sentences_and_offsets(text)
        chunks = [text[start:end][:MAX_CONTEXT_SENTENCE_LENGTH] for start, end in offsets]
        return interaction_id, chunks

    

    def extract_chunks(self, batch_interaction_ids, batch_search_results):
        
        
        for idx, search_results in enumerate(batch_search_results):
          chunk_dictionary = defaultdict(list)
          for html_text in search_results:
            soup = BeautifulSoup(html_text["page_result"], "lxml")
            text = soup.get_text(" ", strip=True)
            if not text:
              text=""
            recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)
            chunks = recursive_splitter.split_text(text)
            
            
            chunk_dictionary[idx].extend(chunks)

        chunks, chunk_interaction_ids = self._flatten_chunks(chunk_dictionary)
        return chunks, chunk_interaction_ids

    def _flatten_chunks(self, chunk_dictionary):
        chunks = []
        chunk_interaction_ids = []

        for interaction_id, _chunks in chunk_dictionary.items():
            unique_chunks = list(set(_chunks))
            chunks.extend(unique_chunks)
            chunk_interaction_ids.extend([interaction_id] * len(unique_chunks))

        chunks = np.array(chunks)
        chunk_interaction_ids = np.array(chunk_interaction_ids)
        return chunks, chunk_interaction_ids

class RAGModel:

    def __init__(self):
        self.initialize_models()
        self.chunk_extractor = ChunkExtractor()

    def initialize_models(self):
        self.model_name = "models/meta-llama/Meta-Llama-3-8B-Instruct"

        if not os.path.exists(self.model_name):
            raise Exception(
                f"""
            The evaluators expect the model weights to be checked into the repository,
            but we could not find the model weights at {self.model_name}
            
            Please follow the instructions in the docs below to download and check in the model weights.
            
            https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/blob/master/docs/dataset.md
            """
            )

        bnb_config = BitsAndBytesConfig(
          load_in_4bit=True,
          bnb_4bit_compute_dtype=torch.float16,
          bnb_4bit_quant_type="nf4",
          bnb_4bit_use_double_quant=False,
        )
        self.cross_encoder = CrossEncoder('models/cross-encoder/ms-marco-MiniLM-L-2-v2')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,cache_dir='',)
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir='',
            device_map='auto',
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
        )
        terminators = [
          self.tokenizer.eos_token_id,
          self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        self.generation_pipe =pipeline(
          "text-generation",
          model=self.llm,
          tokenizer=self.tokenizer,
          torch_dtype=torch.float16,
          max_new_tokens=75,
          temperature=0.3,
          device_map="auto",
          eos_token_id=terminators,
          )

        self.sentence_model= SentenceTransformerEmbeddings(model_name='models/sentence-transformers/all-MiniLM-L12-v2')
        self.prompt_template = """
        <|start_header_id|>user<|end_header_id|>
        You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided without using prior knowledge.
        Analyse carefully the context and provide a direct ,concise and short as possible answer without explanation.
        If the question is based on a false premise or assumption, respond with "invalid question."
        If the answer is not in the context, just say "I don't know." Don't make up an answer.
        Question: {question}
        Context: {context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

      
    def calculate_embeddings(self, sentences):
        embeddings = self.sentence_model.encode(
            sentences=sentences,
            normalize_embeddings=True,
            batch_size=SENTENTENCE_TRANSFORMER_BATCH_SIZE,
        )
        return embeddings
    

    def get_batch_size(self) -> int:
        self.batch_size = AICROWD_SUBMISSION_BATCH_SIZE  
        return self.batch_size

    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]

        chunks, chunk_interaction_ids = self.chunk_extractor.extract_chunks(
            batch_interaction_ids, batch_search_results
        )

      
        batch_answers=[]
        batch_retrieval_results = []
        for _idx, interaction_id in enumerate(batch_interaction_ids):
            query = queries[_idx]
            query_time = query_times[_idx]
            #query_embedding = query_embeddings[_idx]

            relevant_chunks_mask = chunk_interaction_ids ==_idx
            relevant_chunks = chunks[relevant_chunks_mask]
            #bm25_retriever = BM25Retriever.from_texts(relevant_chunks)
            #bm25_retriever.k = 30
            faiss_vectorstore = FAISS.from_texts(relevant_chunks, self.sentence_model)
            #faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 30})
            
            result = self.generation_pipe(query)
            result = result[0]["generated_text"]
            hyde_answer = result[len(query):]
            #print('hyde_answer :')
            #print(hyde_answer)
            # initialize the ensemble retriever
            #ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
            #docs = faiss_retriever.invoke(hyde_answer)
            docs = faiss_vectorstore.max_marginal_relevance_search(hyde_answer,k=20, fetch_k=30)
            pairs = []
            for doc in docs:
                pairs.append([query, doc.page_content])
            scores = self.cross_encoder.predict(pairs)
            scored_docs = zip(scores, docs)
            sorted_docs =  sorted(scored_docs, key=lambda x: x[0], reverse=True)
            reranked_docs = [doc for _, doc in sorted_docs][0:20]
            
            references=""
            for doc in reranked_docs:
              references += "<DOC>\n" + doc.page_content+ "\n</DOC>\n"
              references= " ".join(references.split()[:1000])
              
            
            final_prompt = self.prompt_template.format(
                  question=query, context=references 
              )
            # Generate an answer using the formatted prompt.
            result = self.generation_pipe(final_prompt)
            result = result[0]["generated_text"]
            #print('generated_answer :')
            try:
              # Extract the answer from the generated text.
              answer1 = result[len(final_prompt):]
              if answer1 == "" or answer1 == " ":
                answer1= "I don't know"
              #print(answer1)
            except IndexError:
              # If the model fails to generate an answer, return a default response.
              answer1 = "I don't know"
              #print(answer1)    
            
            
            batch_answers.append(answer1)
            batch_retrieval_results.append(reranked_docs)
            
        return batch_answers


    def format_prompts(self, queries, query_times, batch_retrieval_results):
        system_prompt = """You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on references without using prior knowledge.
        Analyse carefully the references and answer the question succinctly, using the fewest words possible.
        If the question is based on a false premise or assumption, respond with "invalid question."
        If the references do not contain the necessary information to answer the question, respond with 'I don't know'. There is no need to explain the reasoning behind your answers."""
        formatted_prompts = []
        
        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            retrieval_results = batch_retrieval_results[_idx]

            user_message = ""
            references = ""
            prompts=[]
            if len(retrieval_results) > 0:
                references += "# References \n"
                # Format the top sentences as references in the model's prompt template.
                for _snippet_idx, snippet in enumerate(retrieval_results):
                    references += f"- {snippet.page_content }\n"
            
                    references = references[:MAX_CONTEXT_REFERENCES_LENGTH]
                    # Limit the length of references to fit the model's input size.

                    user_message += f"{references}\n------\n\n"
                    user_message 
                    user_message += f"Using only the references listed above, answer the following question: \n"
                    user_message += f"Current Time: {query_time}\n"
                    user_message += f"Question: {query}\n"
                    
                    prompts.append(
                        self.tokenizer.apply_chat_template(
                            [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_message},
                            ],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                  )
                formatted_prompts.append(prompts)
              
        return formatted_prompts
