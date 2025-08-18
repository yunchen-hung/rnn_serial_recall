import torch
import logging

from nmt_cmr_parallels.models.encdec_recall_model import EncoderDecoderRecallmodel

def save_recall_model(checkpoint_path, model):

    ckpt_dict = {'state_dict':           model.state_dict(),
                 'vocabulary':           model.vocab,
                 'vocab_size':           model.vocab_size,
                 'frozen_embedding':     model.frozen_embedding,
                 'use_attention':        model.use_attention,
                 'embedding_dim':        model.embedding_dim,
                 'hidden_dim':           model.hidden_dim,
                 'bidirectional':        model.bidirectional,
                 'rnn_mode':             model.rnn_mode,
                 'attention_type':       model.attention_type,
                 'model_type':           model.model_type}
    
    if hasattr(model, 'dropout'):
         ckpt_dict.update({'dropout': model.dropout})
    
    try:
        if not checkpoint_path.endswith('.pt'):
            checkpoint_path += '.pt'
        torch.save(ckpt_dict, checkpoint_path)
        logging.info(f"Saving model to {checkpoint_path}...")
    except:
        print("Error saving model checkpoint.")

def load_recall_model(checkpoint_path, return_vocab = False, device='cpu'):

    # Load the checkpoint
    ckpt_dict = torch.load(checkpoint_path)
    
    # Extract parameters
    vocab = ckpt_dict['vocabulary']
    vocab_size = ckpt_dict['vocab_size']
    frozen_embedding = ckpt_dict['frozen_embedding']
    use_attention = ckpt_dict['use_attention']
    embedding_dim = ckpt_dict['embedding_dim']
    hidden_dim = ckpt_dict['hidden_dim']
    bidirectional = ckpt_dict['bidirectional']
    rnn_mode = ckpt_dict['rnn_mode']
    attention_type = ckpt_dict.get('attention_type', 'none')
    dropout = ckpt_dict.get('dropout', None)
    model_type = ckpt_dict.get('model_type', None)

    # Initialize model
    if model_type == 'encoderdecoder':
         model = EncoderDecoderRecallmodel(vocab_size, 
                                embedding_dim, 
                                hidden_dim, 
                                use_attention=use_attention,
                                frozen_embedding=frozen_embedding,
                                rnn_mode=rnn_mode,
                                attention_type=attention_type,
                                bidirectional=bidirectional,
                                dropout=dropout,
                                vocab=vocab,
                                device=device)
    else:
        raise TypeError("Unrecognized model type in checkpoint.")
    
    # Load the state dictionary into the model
    model.load_state_dict(ckpt_dict['state_dict'])
    if frozen_embedding:
        for param in model.embedding.parameters():
                param.requires_grad = False

    logging.info(f"Model loaded from {checkpoint_path}.")

    if return_vocab:
        return model, vocab
    return model

def save_rl_agent(checkpoint_path, agent):

    critic = agent.critic
    actor = agent.actor
     
    ckpt_dict = {'state_dict':           actor.state_dict(),
                 'vocabulary':           actor.vocab,
                 'vocab_size':           actor.vocab_size,
                 'frozen_embedding':     actor.frozen_embedding,
                 'use_attention':        actor.use_attention,
                 'embedding_dim':        actor.embedding_dim,
                 'hidden_dim':           actor.hidden_dim,
                 'bidirectional':        actor.bidirectional,
                 'rnn_mode':             actor.rnn_mode,
                 'attention_type':       actor.attention_type,
                 'model_type':           actor.model_type,
                 'critic_dict':          critic.state_dict()}
    
    try:
        if not checkpoint_path.endswith('.pt'):
            checkpoint_path += '.pt'
        torch.save(ckpt_dict, checkpoint_path)
        logging.info(f"Saving model to {checkpoint_path}...")
    except:
        print("Error saving model checkpoint.")

def load_rl_agent(checkpoint_path, agent):

    # Load the checkpoint
    ckpt_dict = torch.load(checkpoint_path)
    
    # Extract parameters
    vocab = ckpt_dict['vocabulary']
    vocab_size = ckpt_dict['vocab_size']
    frozen_embedding = ckpt_dict['frozen_embedding']
    use_attention = ckpt_dict['use_attention']
    embedding_dim = ckpt_dict['embedding_dim']
    hidden_dim = ckpt_dict['hidden_dim']
    bidirectional = ckpt_dict['bidirectional']
    rnn_mode = ckpt_dict['rnn_mode']
    attention_type = ckpt_dict.get('attention_type', 'none')
    model_type = ckpt_dict.get('model_type', None)

    # Initialize model
    if model_type == 'encoderdecoder':
         model = EncoderDecoderRecallmodel(vocab_size, 
                                embedding_dim, 
                                hidden_dim, 
                                use_attention=use_attention,
                                frozen_embedding=frozen_embedding,
                                rnn_mode=rnn_mode,
                                attention_type=attention_type,
                                bidirectional=bidirectional,
                                vocab=vocab,
                                device=agent.device)
    else:
        raise TypeError("Unrecognized model type in checkpoint.")
    
    # Load the state dictionary into the model
    model.load_state_dict(ckpt_dict['state_dict'])
    if frozen_embedding:
        for param in model.embedding.parameters():
                param.requires_grad = False

    agent.actor = model
    agent.critic.load_state_dict(ckpt_dict['critic_dict'])
    
    logging.info(f"Model loaded from {checkpoint_path}.")

    return agent