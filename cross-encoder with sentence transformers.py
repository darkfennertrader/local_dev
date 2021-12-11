#### same task with sentence transformers ######################

from sentence_transformers.cross_encoder import CrossEncoder
import numpy as np

# Pre-trained cross encoder
model = CrossEncoder('cross-encoder/stsb-distilroberta-base')

# We want to compute the similarity between the query sentence
query = "Who is Diego Maradona? Diego Maradona was the greatest soccer player of all times. How many goals did he score in his career? he scored many a goal. Can you tell the same for Pele?, Pele scored many goals throughout his career. Who is John Lennon?"

query = "Pele scored many goals throughout his career. Who is John Lennon?"

# With all sentences in the corpus
corpus = ['What about soccer?',
          'Where did Maradona play in his career?',
          'The girl is carrying a baby.',
          'A man is riding a horse.',
          'A woman is playing violin.',
          'Two men pushed carts through the woods.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'Pele scored many goals throughout his career.',
          "he scored many a goal",
          "Pele was a great player",
          "John Lennon was a great singer"
          ]

corpus = ['Pele scored many goals throughout his career.',
          "he scored many a goal",
          "Pele was a great player",
          "John Lennon was a great singer"
          ]

# So we create the respective sentence combinations
sentence_combinations = [[query, corpus_sentence] for corpus_sentence in corpus]

# Compute the similarity scores for these combinations
similarity_scores = model.predict(sentence_combinations)

# Sort the scores in decreasing order
sim_scores_argsort = reversed(np.argsort(similarity_scores))

# Print the scores
print("Query:", query)
for idx in sim_scores_argsort:
    print("{:.2f}\t{}".format(similarity_scores[idx], corpus[idx]))