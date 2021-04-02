# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

from models.qa_model import *

# train_and_save()
data_dict, w2v_model, w2v_base_features, \
tfidf_svd_model, tfidf_svd_base_features,\
bm25_model, doc2vec_model, fuzzy_searcher = load()

class ActionAsk(Action):
    def name(self) -> Text:
        return "action_ask"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        query = tracker.latest_message['text']
        if query == '/ask':
            query = tracker.slots["question"]
        w2v_res = w2v_model.find_k_most_similar(query, w2v_base_features, 2)
        tfidf_svd_res = tfidf_svd_model.find_k_most_similar(query, tfidf_svd_base_features, 2)
        bm25_res = bm25_model.find_k_most_similar(query, 2)
        doc2vec_res = doc2vec_model.find_k_most_similar(query, 2)
        fuzzy_res = fuzzy_searcher.find_k_most_similar(query, [data['tokenized_question'] for data in data_dict.values()],2)

        res_set = set([r[0] for r in w2v_res] + \
                    [r[0] for r in tfidf_res] + \
                    [r[0] for r in bm25_res] + \
                    [r[0] for r in doc2vec_res] + \
                    [r[0] for r in fuzzy_res])

        if not res_set:
            elements = []
            for id in list(res_set):
                a_dict = {}
                a_dict["title"] = 'Câu hỏi #'+str(id)
                # a_dict["subtitle"] = data_dict[str(id)]['question']
                index = str(id)
                if 'pid' in data_dict[index]:
                    index = data_dict[index]['pid']
                a_dict["subtitle"] = data_dict[index]['question']

                a_dict["image_url"] = "https://scontent.fhan2-6.fna.fbcdn.net/v/t1.6435-9/67968165_518806932190832_6941292360135344128_n.png?_nc_cat=100&ccb=1-3&_nc_sid=09cbfe&_nc_ohc=LL7UIid9dr4AX8cMOvO&_nc_ht=scontent.fhan2-6.fna&oh=18f3d8714478c20d7177ff851aa1696b&oe=6089388B"
                a_dict["buttons"] = [
                    {
                        "type": "postback",
                        "payload": "/ask_detail{" + str(id) + "}",
                        "title": "Xem câu trả lời"
                    }
                ]
                elements.append(a_dict)
            attachment = {}
            attachment["type"] = "template"
            attachment["payload"] = {"template_type":"generic", "elements":elements}

            dispatcher.utter_message(text='Các câu hỏi tương tự:')
            dispatcher.utter_message(json_message = {"attachment":attachment})
        else:
            dispatcher.utter_message(text='Chúng tôi không tìm thấy câu hỏi nào tương ứng. Xin hãy thay đổi câu hỏi hoặc cách hỏi sang trực tiếp.')
        return [SlotSet("question", query)]

class ActionAskDetail(Action):

    def name(self) -> Text:
        return "action_ask_detail"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        event_text = ''
        for event in (list(reversed(tracker.events)))[:3]: # latest 5 messages
            if event.get("event") == "user":
                event_text = event.get("text")
        data = data_dict[event_text.split('{')[-1].split('}')[0]]
        ans = 'Đây là câu hỏi thuộc {}.\n{}'.format(data["domain"], data["answer"])
        dispatcher.utter_message(text=ans)
        dispatcher.utter_message(text='Anh/Chị có hài lòng với câu trả lời này không?')
        return []
