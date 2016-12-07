'''
This file is used for training a corpus of documents, collected from elasticsearch, using Gensim's implementation of Latent Dirichlet Allocation. The corpus of documents is under 'Science' category in 'Wikipedia', stored in Elasticsearch. The data fetched, is then tokenized, lower-cased and lemmatized. Finally, LDA model is trained over the collection of these documents.
'''

from elasticsearch import Elasticsearch, helpers
import gensim
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Building tokenizer object, stopwords set in english and a lemmatizer object from NLTK
tokenizer = RegexpTokenizer(r'\w+')
stopword_set = set(stopwords.words('english'))
w_lemmatizer = WordNetLemmatizer()

# This function does all cleaning of data using three objects above
def nlp_clean(data):

	token_data = []
	for d in data:
		new_d = d.lower()
		dlist = tokenizer.tokenize(new_d)
		dlist = list(set(dlist).difference(stopword_set))
		new_dlist = [w_lemmatizer.lemmatize(tok) for tok in dlist]
		token_data.append(new_dlist)

	return token_data
		
# Elasticsearch server connection
es = Elasticsearch('localhost:9200')

# Elasticsearch full-body search query
query = {
	'query':{
		'match_all':{}
	}
}

# 'data' is a list that stores all documents from elasticsearch
data = []

# 'result' is a generator that helps us iterate over all document results of query one-by-one
result = helpers.scan(es, query, scroll=u'5m', index='kdc', doc_type='science')
for res in result:
	data.append(res['_source']['Summary'])

# 'token_data' is new list that stores nlp-cleaned documents
token_data = nlp_clean(data)

# a dictionary is an id-to-word mapping that is built before training the model
dictionary = gensim.corpora.Dictionary(token_data)

# the dictionary is used to represent the documents in a bag-of-word fashion
corpus = [dictionary.doc2bow(d) for d in token_data]

# training step for '1000' iterations and clustering corpus into '100' topics
lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=100, id2word=dictionary, passes=100)

# testing model by printing top 10 words of topic index 1
print lda_model.get_topic_terms(1, topn=10)

# saving the created model
lda_model.save('science_lda.model')
