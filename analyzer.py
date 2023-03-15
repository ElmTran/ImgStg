import nltk
import time
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim
import pandas as pd
import networkx as nx
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from wordcloud import WordCloud


def get_func_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print('Function {} took {} seconds'.format(func.__name__, end - start))
    return wrapper


class AirbnbAnalyzer:

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.stop_words = set(stopwords.words('english'))
        self.positive_texts = list()
        self.positive_words = set()
        self.negative_texts = list()
        self.negative_words = set()

    # Load data when called
    def load_data(self, nrows=100):
        self.data = pd.read_csv(self.file_path, nrows=nrows).dropna()

    def sentiment_analysis(self):
        sentiments = self.data['comments'].apply(
            lambda comment: 'positive' if SentimentIntensityAnalyzer(
            ).polarity_scores(comment)['compound'] >= 0 else 'negative'
        ).tolist()
        self.data['sentiment'] = sentiments

    def preprocess(self, text):
        words = [word.lower() for word in nltk.word_tokenize(text)]
        words = [word for word in words if word not in self.stop_words and word.isalpha(
        ) and len(word) > 2]
        return words

    def process_sentiment(self, sentiment_value):
        data = self.data[self.data['sentiment'] == sentiment_value]
        texts = data['comments'].drop_duplicates().apply(self.preprocess)
        words = set(word for text in texts for word in text)
        return texts, words

    def split(self):
        self.positive_texts, self.positive_words = self.process_sentiment(
            'positive')
        self.negative_texts, self.negative_words = self.process_sentiment(
            'negative')

    def word_cloud(self):
        _, axes = plt.subplots(1, 2, figsize=(16, 8))
        clouds = [
            (axes[i], set_of_words, title)
            for i, (set_of_words, title) in enumerate([
                (self.positive_words, 'Positive'),
                (self.negative_words, 'Negative')
            ])
        ]
        for ax, words, title in clouds:
            cloud = WordCloud(
                width=1600, height=800,
                background_color='White', stopwords=self.stop_words,
            ).generate(' '.join(words))
            ax.imshow(cloud)
            ax.axis("off")
            ax.set_title(title)
        plt.tight_layout(pad=0)
        plt.show()

    @get_func_time
    def topic_modelling(self, num_topics=5):
        positive_lda = self.get_model_params(
            self.positive_texts, num_topics)
        negative_lda = self.get_model_params(
            self.negative_texts, num_topics)

        self.network_analysis(
            positive_lda['dictionary'], positive_lda['model'])
        self.network_analysis(
            negative_lda['dictionary'], negative_lda['model'])
        # pyLDAvis.show(positive_lda['vis'], local=False)
        # pyLDAvis.show(negative_lda['vis'], local=False)

    def get_model_params(self, texts, num_topics):
        dictionary = Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        model = LdaModel(
            corpus, num_topics=num_topics, id2word=dictionary
        )
        # perplexity = model.log_perplexity(corpus)
        # score = model.bound(corpus)
        # coherence = CoherenceModel(
        #     model=model, texts=texts, dictionary=dictionary, coherence='c_v'
        # ).get_coherence()
        vis = pyLDAvis.gensim.prepare(model, corpus, dictionary)
        return {
            "dictionary": dictionary,
            "corpus": corpus,
            "model": model,
            # "perplexity": perplexity,
            # "score": score,
            # "coherence": coherence,
            "vis": vis
        }

    def network_analysis(self):
        pass

    def jaccard(self, list1, list2):
        set1 = set(item[0] for item in list1)
        set2 = set(item[0] for item in list2)
        return len(set1.intersection(set2)) / len(set1.union(set2))

    def choose_best_topic(self, max_topics=10):
        coherences = []
        for num_topics in range(2, max_topics):
            positive_lda = self.get_model_params(
                self.positive_texts, num_topics)
            negative_lda = self.get_model_params(
                self.negative_texts, num_topics)
            coherences.append(
                (positive_lda['coherence'], negative_lda['coherence'])
            )
        _, axes = plt.subplots(1, 2, figsize=(16, 8))
        for i, (coherence, title) in enumerate([
            (coherences, 'Coherence'),
        ]):
            axes[i].plot(range(2, max_topics), coherence)
            axes[i].set_title(title)
            axes[i].set_xlabel('Number of Topics')
            axes[i].set_ylabel('Coherence Score')
        plt.tight_layout(pad=0)
        plt.show()

    @get_func_time
    def run(self):
        self.load_data(1000)
        self.sentiment_analysis()
        self.split()
        # self.word_cloud()
        self.topic_modelling(5)
        # self.choose_best_topic()


if __name__ == '__main__':
    analyzer = AirbnbAnalyzer('./data/reviews.csv')
    analyzer.run()
