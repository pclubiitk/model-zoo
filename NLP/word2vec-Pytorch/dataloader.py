from collections import Counter

def loadData(name):

	with open(name) as f:
        	text = f.read().split()
	return text

def prepareData(text, min_freq):

	freq = Counter(text)
	vocab = [word for word in text if freq[word] > min_freq]
	freq = Counter(vocab)

	#sorted_freq = sorted(freq, key=freq.get, reverse=True)
	vocab_size = sum([freq[k] for k in freq])

	word2idx = {}
	idx2word = {}

	for i,w in enumerate(freq):
	   word2idx[w] = i
	   idx2word[i] = w

	int_text = [word2idx[word] for word in vocab]

	return int_text, word2idx, idx2word, freq, vocab


def loadBatches(words_in_idx, batch_size, window_size):

  num_batches = int(len(words_in_idx)/batch_size)

  for i in range(num_batches):

    batch_idx = words_in_idx[i*batch_size:(i+1)*batch_size]
    context = []
    y = []
    for j in range(len(batch_idx)):
      word = batch_idx[j]
      if j - window_size > 0:
        start_context_index = j - window_size
      else:
         start_context_index = 0

      end_context_index = j + window_size + 1
      context.extend(batch_idx[start_context_index:j])
      y.extend([word]*len(batch_idx[start_context_index:j]))
      context.extend(batch_idx[j+1:end_context_index])
      y.extend([word]*len(batch_idx[j+1:end_context_index]))
    
    yield(context,y)
