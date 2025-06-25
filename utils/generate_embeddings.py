import numpy as np
import ollama

from transformers import CLIPModel, CLIPProcessor, AutoTokenizer, AutoModel


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
def initialize_model(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", trust_remote_code=False):
    """
    Initialize the model and tokenizer for generating embeddings.
    
    Args:
    model_name: str, name of the model to use for generating embeddings
    trust_remote_code: bool, whether to trust remote code for the model
    """
    text_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        
    text_model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    return text_model, text_tokenizer


def generate_short_text_embeddings(query, text_tokenizer=None, text_model=None):
    """
    Generate text embeddings using Sentence Transformers model
    
    Args:
    query: str, text to generate embeddings for pgvector
    """
    if text_tokenizer and text_model:
        text_inputs = text_tokenizer(query, padding=True, truncation=True, return_tensors="pt")
        text_model_output = text_model(**text_inputs)
        text_embeddings = text_model_output.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy().tolist()
    else:
        print("Text model and tokenizer are not initialized.")
        return None

    return text_embeddings

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