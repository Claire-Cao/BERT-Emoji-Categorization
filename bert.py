import torch
import torch.nn as nn
import torch.nn.functional as F
from base_bert import BertPreTrainedModel
from utils import *


class BertSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads  # h
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)  # d/h
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)

    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    bs, seq_len = x.shape[:2]
    proj = linear_layer(x)  # [bs, seq_len, hidden_state] * layer [hidden_size, all_head_size]
    # Produce multiple heads by spliting the hidden state to self.num_attention_heads
    proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
    # After transpose, proj size is [bs, num_attention_heads, seq_len, attention_head_size].
    proj = proj.transpose(1, 2)
    return proj

  def attention(self, key, query, value, attention_mask):
    # multi head attention
    scale_dk = self.attention_head_size ** (-0.5)
    query = query * scale_dk  # [bs, num_attention_heads, seq_len, attention_head_size]
    attention_score = torch.matmul(query, key.transpose(-1, -2))

    if attention_mask is not None:
      attention_score += attention_mask

    normalized_attention_score = attention_score.softmax(-1)
    normalized_attention_score = self.dropout(normalized_attention_score)  # [bs, num_attention_heads, seq_len, seq_len]

    weighted_values = torch.matmul(normalized_attention_score, value)  # [bs, num_attention_heads, seq_len, attention_head_size]
    weighted_values = weighted_values.transpose(1, 2)
    bs, seq_len = weighted_values.shape[:2]
    # Q1 where to add last linear layer of W0
    return weighted_values.reshape(bs, seq_len, -1)

  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state] # hidden state dimension mostly is embedding dimension d
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """

    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)  # XK
    value_layer = self.transform(hidden_states, self.value)  # XV
    query_layer = self.transform(hidden_states, self.query)  # XQ
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value


class BertLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention.
    self.self_attention = BertSelfAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)  # this equals to the last step linear W0 in multi-head attention
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Feed forward.
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # Add-norm for feed forward.
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add_norm(self, input, output, dense_layer, dropout, ln_layer):
    """
    This function is applied after the multi-head attention layer or the feed forward layer.
    """

    transformed_output = dense_layer(output)  # a linear layer
    additive = input + dropout(transformed_output)
    return ln_layer(additive)

  def forward(self, hidden_states, attention_mask):

    attention_output = self.self_attention.forward(hidden_states, attention_mask)
    normalized_attention_output = self.add_norm(hidden_states, attention_output, self.attention_dense, self.attention_dropout, self.attention_layer_norm)
    feed_forward_output = self.interm_af(self.interm_dense(normalized_attention_output))
    normalized_feed_forward_output = self.add_norm(normalized_attention_output, feed_forward_output, self.out_dense, self.out_dropout, self.out_layer_norm)
    return normalized_feed_forward_output

class BertModel(BertPreTrainedModel):
  """
  The BERT model returns the final embeddings for each token in a sentence.
  """
  def __init__(self, config):
    super().__init__(config)
    self.config = config

    # Embedding layers.
    self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
    self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Register position_ids (1, len position emb) to buffer because it is a constant.
    position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
    self.register_buffer('position_ids', position_ids)

    # BERT encoder.
    self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    # [CLS] token transformations.
    self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.pooler_af = nn.Tanh()

    self.init_weights()

  def embed(self, input_ids):
    input_shape = input_ids.size()  # input_ids: [batch_size, seq_len]
    seq_length = input_shape[1]

    inputs_embeds = None

    inputs_embeds = self.word_embedding(input_ids)  # [batch_size, seq_len, embedding_dimension(i.e. hidden_size)]

    pos_ids = self.position_ids[:, :seq_length]
    pos_embeds = None

    pos_embeds = self.pos_embedding(pos_ids)

    tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
    tk_type_embeds = self.tk_type_embedding(tk_type_ids)

    # Add three embeddings together; then apply embed_layer_norm and dropout and return.
    embedding_combined = inputs_embeds + pos_embeds + tk_type_embeds
    return self.embed_dropout(self.embed_layer_norm(embedding_combined))


  def encode(self, hidden_states, attention_mask):
    """
    hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
    attention_mask: [batch_size, seq_len]
    """
    extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)

    # Pass the hidden states through the encoder layers.
    for i, layer_module in enumerate(self.bert_layers):
      # Feed the encoding from the last bert_layer to the next.
      hidden_states = layer_module(hidden_states, extended_attention_mask)

    return hidden_states

  def forward(self, input_ids, attention_mask):
    """
    input_ids: [batch_size, seq_len], seq_len is the max length of the batch
    attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
    """
    embedding_output = self.embed(input_ids=input_ids)

    sequence_output = self.encode(embedding_output, attention_mask=attention_mask)

    # Get cls token hidden state.
    first_tk = sequence_output[:, 0]
    first_tk = self.pooler_dense(first_tk)
    first_tk = self.pooler_af(first_tk)

    return {'last_hidden_state': sequence_output, 'pooler_output': first_tk}
