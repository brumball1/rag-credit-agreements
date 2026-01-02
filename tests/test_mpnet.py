from sentence_transformers import SentenceTransformer
#from sklearn.metrics.pairwise import cosine_similarity
from pprint import pprint as pp

model = SentenceTransformer("all-mpnet-base-v2")
sentences = ["What is the time", "I went to the shop", "I want to go to TK Max later"]
sentence_embeddings = model.encode(sentences)
#print(sentence_embeddings)
print(len(sentence_embeddings))
print(len(sentence_embeddings[1]))

pp("The similarity between {} and {} is {} ".format(sentences[1],
                                                       sentences[2],
                                                       cosine_similarity(sentence_embeddings[1].reshape(1,-1), sentence_embeddings[2].reshape(1,-1))[0][0]))


