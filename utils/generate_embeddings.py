import numpy as np
import ollama

from transformers import CLIPModel, CLIPProcessor, AutoTokenizer, AutoModel
from FlagEmbedding import BGEM3FlagModel


def generate_short_text_embeddings(query):
    """
    Generate text embeddings using Sentence Transformers model
    
    Args:
    query: str, text to generate embeddings for pgvector
    """
    text_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
    text_model = AutoModel.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    text_inputs = text_tokenizer(text=query, return_tensors="pt")
    text_model_output = text_model(**text_inputs)
    text_embedding = text_model_output.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy().tolist()

    return text_embedding

def generate_text_embeddings(query):
    """
    Generate text embeddings using Sentence Transformers model
    
    Args:
    query: str, text to generate embeddings for pgvector
    """
    text_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        
    text_model = AutoModel.from_pretrained("BAAI/bge-m3")
    # https://bge-model.com/tutorial/1_Embedding/1.2.4.html
    model = BGEM3FlagModel('BAAI/bge-m3',  
                       use_fp16=True)
    text_model_output = model.encode(query, 
                            batch_size=12, 
                            max_length=2000, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                            )['dense_vecs']
    text_embedding = text_model_output.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy().tolist()

    return text_embedding

def generate_image_embeddings(image):
    """
    Generate image embeddings using CLIP model
    Args:
    image: PIL.Image, image to generate embeddings for pgvector and VChord
    """
    image_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    inputs = image_processor(text=["dummy text"], images=image, return_tensors="pt", padding=True)
    outputs = image_model(**inputs)
    image_embeddings = outputs.image_embeds
    image_embedding = image_embeddings.detach().cpu().numpy().tolist()
    return image_embedding[0]


def generate_ollama_embeddings(text, model_name="llama3.2-vision"):
    ollama_embedding = ollama.embed(model=model_name, input=text).embeddings
    pgvector_embedding = np.array(ollama_embedding).tolist()
    
    return pgvector_embedding[0]