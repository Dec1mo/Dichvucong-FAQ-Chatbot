import torch
import json
import pickle
import numpy
from scipy.spatial import distance

from underthesea import word_tokenize
from gensim.utils import simple_preprocess
from spacy.language import Language
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

class Preprocessor():
    def __init__(self):
        pass

    @staticmethod
    def text_preprocess(text):
        return ' '.join(simple_preprocess(' '.join(word_tokenize(text))))

    def json_preprocess(self, json_file):
        data_dict = {}
        with open(json_file, encoding='utf8') as json_file:
            data_dict = json.load(json_file)
        for i, data in data_dict.items():
            data_dict[i] = data
            data_dict[i]['tokenized_question'] = ' '.join(word_tokenize(data['question'].lower()))
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
#         tokenized_query = ViTokenizer.tokenize(query.lower())
#         query_feature = self.features_extractor(tokenized_query)
#         distances = distance.cdist(query_feature, base_features, sim)[0]
#         k_indexes = numpy.argsort(distances)[:k]
#         return [(k_indexes[i], 1 - distances[k_indexes[i]]) for i in range(k)]

class VNword2vecModel():
    def __init__(self):
        self.nlp = Language()

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
        return [(k_indexes[i], sims[k_indexes[i]]) for i in range(k)]

class TFIDFModel():
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit_transform(self, corpus):
        return self.vectorizer.fit_transform(corpus)

    def find_k_most_similar(self, query, base_features, k=3, sim='cosine'):
        query = [Preprocessor.text_preprocess(query)]
        query_feature = self.vectorizer.transform(query).toarray()  
        distances = distance.cdist(query_feature, base_features, sim)[0]
        k_indexes = numpy.argsort(distances)[:k]
        return [(k_indexes[i], 1 - distances[k_indexes[i]]) for i in range(k)]

class BM25Model():
    def __init__(self, tokens_list):
        self.bm25 = BM25Okapi(tokens_list)

    def find_k_most_similar(self, query, k=3):
        tokenized_query = Preprocessor.text_preprocess(query).split(' ')
        sims = self.bm25.get_scores(tokenized_query)
        k_indexes = numpy.argsort(-sims)[:k]
        return [(k_indexes[i], sims[k_indexes[i]]) for i in range(k)]

# def main():
#     preprocessor = Preprocessor()
#     data_dict = preprocessor.json_preprocess('data/data_dict.json')
#     print(preprocessor.text_preprocess('Hôm nay tôi000 đi học. d.sad.sa.'))
# 
#     # # Preprocess and save data
#     # ## Phobert
#     # pb_model = PhobertModel()
#     # phobert_base_features = pb_model.features_extractor(data_dict['0']['tokenized_question'].lower())
#     # for i in range(1, len(data_dict)):
#     #     this_features = pb_model.features_extractor(data_dict[str(i)]['tokenized_question'].lower())
#     #     phobert_base_features = numpy.vstack((phobert_base_features, this_features))
    
#     # with open('pkl/phobert_base_features.pkl', 'wb') as f:
#     #     pickle.dump(phobert_base_features, f)

#     # ## w2v
#     # w2v_model = VNword2vecModel()
#     # w2v_model.load_nlp(("wiki.vi.vec"))
#     # w2v_base_features = []
#     # for data in data_dict.values():
#     #     string = Preprocessor.text_preprocess(data['question'])
#     #     w2v_base_features.append(w2v_model.nlp(string))
#     # with open('pkl/w2v_base_features.pkl', 'wb') as f:
#     #     pickle.dump(w2v_base_features, f)
    
#     # ## TFIDF
#     # tfidf_model = TFIDFModel()
#     # X = []
#     # for data in data_dict.values():
#     #     X.append(Preprocessor.text_preprocess(data['question']))
#     # tfidf_base_features = tfidf_model.fit_transform(X).toarray()
#     # with open('pkl/tfidf_base_features.pkl', 'wb') as f:
#     #     pickle.dump(tfidf_base_features, f)
#     # with open('pkl/tfidf_model.pkl', 'wb') as f:
#     #     pickle.dump(tfidf_model, f)
    
#     # ## BM25
#     # tokens_list = []
#     # for data in data_dict.values():
#     #     tokens_list.append(Preprocessor.text_preprocess(data['question']).split(' '))
#     # bm25_model = BM25Model(tokens_list)
#     # print(bm25_model.find_k_most_similar('Ai là người phải trực tiếp nộp hồ sơ đề nghị trợ cấp xã hội hàng tháng, hồ sơ đề nghị hỗ trợ kinh phí chăm sóc hàng tháng?'))
#     # with open('pkl/bm25_model.pkl', 'wb') as f:
#     #     pickle.dump(bm25_model, f)


#     # Load data
#     # pb_model = PhobertModel()
#     # with open('pkl/phobert_base_features.pkl', 'rb') as f:
#     #     phobert_base_features = pickle.load(f)

#     w2v_model = VNword2vecModel()
#     w2v_model.load_nlp(("wiki.vi.vec"))
#     with open('pkl/w2v_base_features.pkl', 'rb') as f:
#         w2v_base_features = pickle.load(f)

#     with open('pkl/tfidf_model.pkl', 'rb') as f:
#         tfidf_model = pickle.load(f)
#     with open('pkl/tfidf_base_features.pkl', 'rb') as f:
#         tfidf_base_features = pickle.load(f)

#     with open('pkl/bm25_model.pkl', 'rb') as f:
#         bm25_model = pickle.load(f)

#     while True:
#         query = input('Nhập câu hỏi: ')

#         # pb_res = pb_model.find_k_most_similar(query, phobert_base_features)
#         w2v_res = w2v_model.find_k_most_similar(query, w2v_base_features)
#         tfidf_res = tfidf_model.find_k_most_similar(query, tfidf_base_features)
#         bm25_res = bm25_model.find_k_most_similar(query)

#         print('Bạn có thể tìm thấy kết quả mong muốn trong những câu hỏi tương tự sau:\n')
#         # print('pb_res = ', pb_res)
#         # for (id, prob) in pb_res:
#         #     print('Câu hỏi thuộc "{}" (Độ tương đồng {:.2f}%): '.format(data_dict[str(id)]['domain'], prob*100))
#         #     print(data_dict[str(id)]['question'])
#         #     print('-'*80)
#         #     print('Trả lời: ')
#         #     print(data_dict[str(id)]['answer'])
#         #     print('-'*80)
#         # print('='*80)

#         # print('w2v_res = ', w2v_res)
#         for (id, prob) in w2v_res:
#             print('Câu hỏi thuộc "{}" (Độ tương đồng {:.2f}%): '.format(data_dict[str(id)]['domain'], prob*100))
#             print(data_dict[str(id)]['question'])
#             print('-'*80)
#             print(data_dict[str(id)]['answer'])
#             print('-'*80)
#         print('='*80)
#         # print('tfidf_res = ', tfidf_res)
#         for (id, prob) in tfidf_res:
#             print('Câu hỏi thuộc "{}" (Độ tương đồng {:.2f}%): '.format(data_dict[str(id)]['domain'], prob*100))
#             print(data_dict[str(id)]['question'])
#             print('-'*80)
#             # print('Trả lời: ')
#             print(data_dict[str(id)]['answer'])
#             print('-'*80)
#         # print('bm25_res = ', bm25_res)
#         for (id, prob) in bm25_res:
#             print('Câu hỏi thuộc "{}" (Độ tương đồng {:.2f}%): '.format(data_dict[str(id)]['domain'], prob*100))
#             print(data_dict[str(id)]['question'])
#             print('-'*80)
#             print(data_dict[str(id)]['answer'])
#             print('-'*80)
#         print('='*80)
#         print('*'*80)
#         # print()
#         # print()
# 
# if __name__ == '__main__':
#     main()

