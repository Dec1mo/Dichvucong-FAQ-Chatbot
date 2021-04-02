# chatbot
## Requirements
- Python >= 3.6
- [rasa](https://rasa.com/): Chatbot core
- [torch](https://pytorch.org/)
- [scipy](https://www.scipy.org/)
- [pyvi](https://pypi.org/project/pyvi/)
- [gensim](https://pypi.org/project/gensim/)
- [spacy](https://spacy.io/)
- [fuzzywuzzy](https://pypi.org/project/fuzzywuzzy/)
- [rank_bm25](https://pypi.org/project/rank-bm25/)

The implementations below are tested on Window 10

## Background
- For building chatbot: You have to know about Rasa and how to implement a chatbot using it.
- For text similarity search: You have to know about Document Embedding (TFIDF, BM25, Word2Vec) + Cosine Similarity and Fuzzy matching.

## Implementation
Note: should use virtualenv during system installation
```
virtualenv env -p 3.7
env/Scripts/activate
```

On Ubuntu:
```
virtualenv env -p 3.7
source env/bin/activate
```

### Download the repository
```
git clone https://github.com/Dec1mo/chatbot
```

Cài thư viện từ file requirements.txt

`pip install -r requirements.txt`

### Download the saved data
Download saved data [here](https://drive.google.com/drive/folders/1aUzwo-Ty2YsxY_tRk95gUuoGg3GrBEFN?usp=sharing) then add to root directory.

chatbot:

|_ pkl

|_wiki.vi.vec

|_command.sh

### Start actions server
To handle some complex behaviors of the chatbot, we need to customize some rasa actions in ```chatbot/actions/actions.py```. Then we need to define ```action_endpoint``` in ```chatbot/endpoints.yml``` for rasa to know where is actions server. To start actions server, run:

```rasa run actions```

### Start rasa chatbot server
To connect rasa to Facebook Messenger: You can refer [here](https://www.miai.vn/2020/03/23/rasa-series-5-ket-noi-rasa-voi-facebook-messenger-phan-1-2/)

TL;DR (Too long; Didn't read): 
- Create a Facebook Page.
- At ```facebook``` area in ```chatbot/credentials.yml```, define ```verify```, ```secret```, ```page-access-token``` based on that facebook page.
- Use ngrok to expose port 5005 (by default) of rasa chatbot server: ```ngrok http 5005```.
- Facebook requires using https in Webhooks!!! So add https generated by ngrok to Webhooks in Facebook App.
- Start rasa chatbot server: ```rasa run --endpoints endpoints.yml --credentials credentials.yml```


