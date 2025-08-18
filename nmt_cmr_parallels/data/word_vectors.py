import requests
import os
import nltk
import torch
import zipfile
import logging
from gensim.scripts.glove2word2vec import glove2word2vec
DEFAULT_DATA_PATH = os.path.expanduser(os.path.join('~', '.seq_nlp_data'))

def insert_text_before_extension(filename, text):
    base, ext = os.path.splitext(filename)
    return f"{base}{text}{ext}"

def download_glove(data_dir):

    save_path = os.path.join(data_dir, "glove.6B.zip")

    # URL for the GloVe 6B 300d vectors
    url = "http://nlp.stanford.edu/data/glove.6B.zip"
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure we got a valid response

    if not os.path.exists(save_path):
        logging.info("Downloading GloVe embeddings...")
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    logging.info(f"GloVe vectors downloaded to {save_path}")

    logging.info("Unzipping downloaded files...")
    os.makedirs(os.path.join(data_dir,"glove"), exist_ok=True)
    with zipfile.ZipFile(save_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(data_dir,"glove"))  # Extract all files to a folder named 'glove'

    logging.info("Converting vector files to word2vec format...")
    for txt_file in [os.path.join(data_dir, "glove", x) for x in os.listdir(os.path.join(data_dir,"glove"))]:
        glove2word2vec(txt_file, insert_text_before_extension(txt_file,".word2vec"))
        os.remove(txt_file)
    os.remove(save_path)

def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = values[1:]
            embeddings[word] = vector
    embeddings['<BoS>'] = [0.0 for _ in vector]
    embeddings['<EoS>'] = [1.0 for _ in vector]
    vocab = list(embeddings.keys())
    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    return embeddings, vocab, word_to_index

def initialize_lang_data(data_dir=DEFAULT_DATA_PATH, embedding_dim=50, glove_embedding=False):

    # Download if not downloaded
    nltk.download('wordnet')
    nltk.download('words')
    nltk.download('averaged_perceptron_tagger')

    glove_dir = os.path.join(data_dir,"glove")
    if not os.path.isdir(glove_dir):
        download_glove(data_dir)
    
    # If we're using a pretrained embedding, find the one that corresponds to the
    # dimension we're using
    if glove_embedding:
        for txt_f in [os.path.join(glove_dir,x) for x in os.listdir(glove_dir)]:
            if str(embedding_dim) in txt_f:
                _, vocab, _ = load_glove_embeddings(txt_f)
                return vocab
    else:
        return None

def load_pretrained_inverse_embedding(inverse_mlp, seq_tokens=False):

    # Get the directory where the current file (main_module.py) is located
    current_directory = os.path.dirname(os.path.abspath(__file__))

    if seq_tokens:
        checkpoint_path = os.path.join(current_directory, 'inverse_embedding_tokens.pt')
    else:
        checkpoint_path = os.path.join(current_directory, 'inverse_embedding.pt')
    state_dict = torch.load(checkpoint_path)

    inverse_mlp.load_state_dict(state_dict)
    logging.info("Pretrained inverse embedding loaded...")
        
    return inverse_mlp
