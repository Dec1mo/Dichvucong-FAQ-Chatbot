import torch
import json
import pickle
import numpy
import gensim
from gensim.utils import simple_preprocess
from pyvi import ViTokenizer
from scipy.spatial import distance
from spacy.language import Language
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from rank_bm25 import BM25Okapi
from fuzzywuzzy import fuzz

class Preprocessor():
    def __init__(self):
        pass

    @staticmethod
    def text_preprocess(text):
        return ' '.join(simple_preprocess(ViTokenizer.tokenize(text.lower())))

    def json_preprocess(self, json_file):
        data_dict = {}
        with open(json_file, encoding='utf8') as json_file:
            data_dict = json.load(json_file)
        for i, data in data_dict.items():
            data_dict[i] = data
            # data_dict[i]['tokenized_question'] = ViTokenizer.tokenize(data['question'].lower())
            data_dict[i]['tokenized_question'] = Preprocessor.text_preprocess(data['question'])
        return data_dict

# class PhobertModel():
#     def __init__(self):
#         self.phobert = AutoModel.from_pretrained("vinai/phobert-large")
#         self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large", use_fast=False)

#     def features_extractor(self, line):
#         try:
#             input_ids = torch.tensor([self.tokenizer.encode(line)])
#             with torch.no_grad():
#                 features = self.phobert(input_ids)
#         except IndexError:
#             words = line.split(' ')
#             small_words = words[-len(words)//2:]
#             line = ' '.join(small_words)

#             input_ids = torch.tensor([self.tokenizer.encode(line)])
#             with torch.no_grad():
#                 features = self.phobert(input_ids)
#         features = features[0][:, 0, :].numpy()
#         return features

#     def find_k_most_similar(self, query, base_features, sim='cosine', k=3):
#         tokenized_query = ViTokenizer.tokenize(query)
#         query_feature = self.features_extractor(tokenized_query)
#         distances = distance.cdist(query_feature, base_features, sim)[0]
#         k_indexes = numpy.argsort(distances)[:k]
#         return [(k_indexes[i], 1 - distances[k_indexes[i]]) for i in range(k)]

class VNword2vecModel():
    def __init__(self, threshold=0.85):
        self.nlp = Language()
        self.threshold = threshold

    def load_nlp(self, vectors_loc):
        with open(vectors_loc, 'rb') as file_:
            header = file_.readline()
            nr_row, nr_dim = header.split()
            self.nlp.vocab.reset_vectors(width=int(nr_dim))
            for line in file_:
                line = line.rstrip().decode('utf8')
                pieces = line.rsplit(' ', int(nr_dim))
                word = pieces[0]
                vector = numpy.asarray([float(v) for v in pieces[1:]], dtype='f')
                self.nlp.vocab.set_vector(word, vector)  # add the vectors to the vocab

    def find_k_most_similar(self, query, base_features, k=3):
        query = Preprocessor.text_preprocess(query)
        sims = [self.nlp(query).similarity(feature) for feature in base_features]
        sims = numpy.array(sims)
        k_indexes = numpy.argsort(-sims)[:k]
        return [(k_indexes[i], sims[k_indexes[i]]) for i in range(k) if sims[k_indexes[i]] > self.threshold]

class TFIDF_SVDModel():
    def __init__(self, threshold=0.85):
        self.vectorizer = TfidfVectorizer(max_features=20000)
        self.svd = TruncatedSVD(n_components=300, random_state=42)
        self.threshold = threshold

    def fit_transform(self, corpus):
        vector_corpus = self.vectorizer.fit_transform(corpus)
        return self.svd.fit_transform(vector_corpus)

    def find_k_most_similar(self, query, base_features, k=3, sim='cosine'):
        query = [Preprocessor.text_preprocess(query)]
        query_feature = self.svd.transform(self.vectorizer.transform(query))
        distances = distance.cdist(query_feature, base_features, sim)[0]
        k_indexes = numpy.argsort(distances)[:k]
        return [(k_indexes[i], 1-distances[k_indexes[i]]) for i in range(k) if 1-distances[k_indexes[i]] > self.threshold]

class BM25Model():
    def __init__(self, tokens_list, threshold=0):
        self.bm25 = BM25Okapi(tokens_list)
        self.threshold = threshold

    def find_k_most_similar(self, query, k=3):
        tokenized_query = Preprocessor.text_preprocess(query).split(' ')
        sims = self.bm25.get_scores(tokenized_query)
        k_indexes = numpy.argsort(-sims)[:k]
        return [(k_indexes[i], sims[k_indexes[i]]/100) for i in range(k) if sims[k_indexes[i]]/100 > self.threshold]

class Doc2VecModel():
    def __init__(self, threshold=0.85):
        self.model = gensim.models.doc2vec.Doc2Vec(vector_size=300, min_count=2, epochs=40)
        self.threshold = threshold

    def train_from_corpus(self, corpus):
        self.model.build_vocab(corpus)
        self.model.train(corpus, total_examples=self.model.corpus_count, epochs=self.model.epochs)

    def find_k_most_similar(self, query, k=3):
        query = Preprocessor.text_preprocess(query)
        query_vector = self.model.infer_vector(query.split(' '))
        sims = self.model.docvecs.most_similar([query_vector], topn=k)
        return [s for s in sims if s[1] > self.threshold]

class FuzzySearcher():
    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def find_k_most_similar(self, query, textbase, k=3):
        query = Preprocessor.text_preprocess(query)
        sims = [fuzz.partial_ratio(query, text) for text in textbase]
        sims = numpy.array(sims)
        k_indexes = numpy.argsort(-sims)[:k]
        return [(k_indexes[i], sims[k_indexes[i]]/100) for i in range(k) if sims[k_indexes[i]]/100 > self.threshold]

def train_and_save():
    preprocessor = Preprocessor()
    data_dict = preprocessor.json_preprocess('data/new_data_dict.json')
    with open("data/processed_data_dict.json", "w", encoding='utf8') as f:
        json.dump(data_dict, f, indent=4, ensure_ascii=False)

    # # Preprocess and save data
    # ## Phobert
    # pb_model = PhobertModel()
    # phobert_base_features = pb_model.features_extractor(data_dict['0']['tokenized_question'])
    # for i in range(1, len(data_dict)):
    #     this_features = pb_model.features_extractor(data_dict[str(i)]['tokenized_question'])
    #     phobert_base_features = numpy.vstack((phobert_base_features, this_features))
    # with open('saved/phobert_base_features.pkl', 'wb') as f:
    #     pickle.dump(phobert_base_features, f)

    ## Word2Vec
    w2v_model = VNword2vecModel()
    w2v_model.load_nlp(("saved/wiki.vi.vec"))
    w2v_base_features = [w2v_model.nlp(data['tokenized_question']) for data in data_dict.values()]
    with open('saved/w2v_base_features.pkl', 'wb') as f:
        pickle.dump(w2v_base_features, f)
    
    ## TFIDF + SVD
    tfidf_svd_model = TFIDF_SVDModel()
    X = [data['tokenized_question'] for data in data_dict.values()]
    tfidf_svd_base_features = tfidf_svd_model.fit_transform(X)
    with open('saved/tfidf_svd_base_features.pkl', 'wb') as f:
        pickle.dump(tfidf_svd_base_features, f)
    # TFIDF_SVDModel.__module__ = "thing"
    with open('saved/tfidf_svd_model.pkl', 'wb') as f:
        pickle.dump(tfidf_svd_model, f)
    
    tokens_list = [data['tokenized_question'].split(' ') for data in data_dict.values()]

    # ## BM25
    bm25_model = BM25Model(tokens_list)
    with open('saved/bm25_model.pkl', 'wb') as f:
        pickle.dump(bm25_model, f)

    ## Doc2Vec
    corpus = [gensim.models.doc2vec.TaggedDocument(tokens_list[i], [i]) for i in range(len(tokens_list))]
    doc2vec_model = Doc2VecModel()
    doc2vec_model.train_from_corpus(corpus)
    with open('saved/doc2vec_model.pkl', 'wb') as f:
        pickle.dump(doc2vec_model, f)

    ## FuzzySearcher

def load():
    # Load data
    with open('data/processed_data_dict.json', encoding='utf8') as json_file:
        data_dict = json.load(json_file)

    w2v_model = VNword2vecModel(0.9)
    w2v_model.load_nlp(("saved/wiki.vi.vec"))
    with open('saved/w2v_base_features.pkl', 'rb') as f:
        w2v_base_features = pickle.load(f)

    with open('saved/tfidf_svd_model.pkl', 'rb') as f:
        tfidf_svd_model = pickle.load(f)
    tfidf_svd_model.threshold = 0.8
    with open('saved/tfidf_svd_base_features.pkl', 'rb') as f:
        tfidf_svd_base_features = pickle.load(f)

    with open('saved/bm25_model.pkl', 'rb') as f:
        bm25_model = pickle.load(f)
    bm25_model.threshold = 0.17

    with open('saved/doc2vec_model.pkl', 'rb') as f:
        doc2vec_model = pickle.load(f)
    doc2vec_model.threshold = 0.75

    # textbase = [data['tokenized_question'] for data in data_dict.values()]
    fuzzy_searcher = FuzzySearcher(0.9)

    return data_dict, w2v_model, w2v_base_features,\
        tfidf_svd_model, tfidf_svd_base_features,\
        bm25_model, doc2vec_model, fuzzy_searcher


def load_and_run():
    # Load data
    with open('data/processed_data_dict.json', encoding='utf8') as json_file:
        data_dict = json.load(json_file)

    # pb_model = PhobertModel()
    # with open('saved/phobert_base_features.pkl', 'rb') as f:
    #     phobert_base_features = pickle.load(f)

    # w2v_model = VNword2vecModel(0.9)
    # w2v_model.load_nlp(("saved/wiki.vi.vec"))
    # with open('saved/w2v_base_features.pkl', 'rb') as f:
    #     w2v_base_features = pickle.load(f)

    # with open('saved/tfidf_svd_model.pkl', 'rb') as f:
    #     tfidf_svd_model = pickle.load(f)
    # tfidf_svd_model.threshold = 0.8
    # with open('saved/tfidf_svd_base_features.pkl', 'rb') as f:
    #     tfidf_svd_base_features = pickle.load(f)

    # with open('saved/bm25_model.pkl', 'rb') as f:
    #     bm25_model = pickle.load(f)
    # bm25_model.threshold = 0.17

    # with open('saved/doc2vec_model.pkl', 'rb') as f:
    #     doc2vec_model = pickle.load(f)
    # doc2vec_model.threshold = 0.75

    textbase = [data['tokenized_question'] for data in data_dict.values()]
    fuzzy_searcher = FuzzySearcher(0.9)

    while True:
        query = input('Nhập câu hỏi: ')
        # res = pb_model.find_k_most_similar(query, phobert_base_features)
        # res = w2v_model.find_k_most_similar(query, w2v_base_features)
        # res = tfidf_svd_model.find_k_most_similar(query, tfidf_svd_base_features)
        # res = bm25_model.find_k_most_similar(query)
        # res = doc2vec_model.find_k_most_similar(query)
        res = fuzzy_searcher.find_k_most_similar(query, textbase)

        print('Bạn có thể tìm thấy kết quả mong muốn trong những câu hỏi tương tự sau:\n')
        for (id, prob) in res:
            print('Câu hỏi thuộc "{}" (Độ tương đồng {:.2f}%): '.format(data_dict[str(id)]['domain'], prob*100))
            print(data_dict[str(id)]['question'])
            print('-'*80)
            print('Trả lời: ')
            print(data_dict[str(id)]['answer'])
            print('-'*80)
        print('='*80)

def main():
    # train_and_save()
    load_and_run()

if __name__ == '__main__':
    main()

