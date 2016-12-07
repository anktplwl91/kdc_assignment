'''
This file is used for training a corpus of documents, collected from elasticsearch, using Gensim's implementation of Document-to-Vector extension to Word2Vec. The corpus of documents is under 'Science' category in 'Wikipedia', stored in Elasticsearch. The data fetched, is then tokenized, lower-cased and lemmatized. Finally, Doc2Vec model is trained over the collection of these documents.
'''

import gensim
from elasticsearch import Elasticsearch, helpers
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Building tokenizer object, stopwords set in english and a lemmatizer object from NLTK
tokenizer = RegexpTokenizer(r'\w+')
stopword_set = set(stopwords.words('english'))
w_lemmatizer = WordNetLemmatizer()

# This function does all cleaning of data using three objects above
def nlp_clean(data):

	new_data = []
	for d in data:
		new_str = d.lower()
		dlist = tokenizer.tokenize(new_str)
		dlist = list(set(dlist).difference(stopword_set))
		new_dlist = [w_lemmatizer.lemmatize(tok) for tok in dlist]
		new_data.append(new_dlist)

	return new_data

# This class collects all documents from passed list doc_list and corresponding label_list and returns an iterator over those documents
class LabeledLineSentence(object):

	def __init__(self, doc_list, labels_list):

		self.labels_list = labels_list
		self.doc_list = doc_list

	def __iter__(self):

		for idx, doc in enumerate(self.doc_list):
			yield gensim.models.doc2vec.LabeledSentence(doc, [self.labels_list[idx]])

# "docLabels" stores unique labels for all documents and "data" stores the corresponding data of that document
docLabels = []
data = []

# Elasticsearch server connection
es = Elasticsearch('localhost:9200')

# Elasticsearch full-body search query
query = {
	'query':{
		'match_all':{}
	}
}

# 'result' is a generator that helps us iterate over all document results of query one-by-one
result = helpers.scan(es, query, scroll=u'5m', index='kdc', doc_type='science')
for res in result:
	data.append(res['_source']['Summary'])
	docLabels.append(res['_source']['Title'])

# 'new_data' is new list that stores nlp-cleaned documents
new_data = nlp_clean(data)

#iterator returned over all documents
it = LabeledLineSentence(data, docLabels)

#creation of Doc2Vec model and building of vocabulary - 'size' is number of features, 'alpha' is learning-rate, 'min_count' is for neglecting infrequent words 
model = gensim.models.Doc2Vec(size=300, min_count=0, alpha=0.025, min_alpha=0.025)
model.build_vocab(it)


#training of model
for epoch in range(100):
	print 'iteration '+str(epoch+1)
	model.train(it)
	model.alpha -= 0.002
	model.min_alpha = model.alpha
	model.train(it)


#saving the created model 
model.save('science_doc2vec.model')
