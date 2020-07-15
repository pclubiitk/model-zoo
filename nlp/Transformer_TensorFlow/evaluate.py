import tensorflow as tf

from utils import plot_attention_weights
from model import create_masks

def evaluate(inp_sentence,tokenizer_en,tokenizer_pt,MAX_LENGTH,transformer):
  start_token = [tokenizer_pt.vocab_size]
  end_token = [tokenizer_pt.vocab_size + 1]
  
  inp_sentence = start_token + tokenizer_pt.encode(inp_sentence) + end_token
  encoder_input = tf.expand_dims(inp_sentence, 0)
  
  decoder_input = [tokenizer_en.vocab_size]
  output = tf.expand_dims(decoder_input, 0)
    
  for _ in range(MAX_LENGTH):
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        encoder_input, output)
  
    predictions, attention_weights = transformer(encoder_input, 
                                                 output,
                                                 False,
                                                 enc_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask)
    
    predictions = predictions[: ,-1:, :]  

    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    
    if predicted_id == tokenizer_en.vocab_size+1:
      return tf.squeeze(output, axis=0), attention_weights
    
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0), attention_weights

def translate(sentence,tokenizer_en,tokenizer_pt,MAX_LENGTH,transformer, plot=''):
  result, attention_weights = evaluate(sentence,tokenizer_en,tokenizer_pt,MAX_LENGTH,transformer)
  
  predicted_sentence = tokenizer_en.decode([i for i in result if i < tokenizer_en.vocab_size])  

  print('Input: {}'.format(sentence))
  print('Predicted translation: {}'.format(predicted_sentence))
  
  if plot:
    plot_attention_weights(attention_weights, sentence, result, plot,tokenizer_en,tokenizer_pt)  
