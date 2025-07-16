import os
import itertools
import nltk
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer, OnlineCountVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sentence_transformers import SentenceTransformer
import traceback
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary

BERTOPIC_MODEL_BASE_PATH = "models/bertopic_albertina_online"
ARTICLES_PATH = "data/articles/articles_body/articles.parquet"
START_DATE = datetime(2023, 1, 7)
END_DATE = datetime(2023, 4, 28)
CONTINUOUS_MODEL_NAME = "bertopic_continuous_model.pkl"
SENTENCE_MODEL = "PORTULAN/albertina-100m-portuguese-ptbr-encoder"

def _get_continuous_model_path() -> str:
    return os.path.join(BERTOPIC_MODEL_BASE_PATH, CONTINUOUS_MODEL_NAME)

def load_continuous_bertopic_model() -> BERTopic:
    path = _get_continuous_model_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Continuous model not found at {path}")
    print(f"Loading continuous model from {path}...")
    model = BERTopic.load(path)
    if not isinstance(model, BERTopic):
        raise ValueError(f"Loaded object is not BERTopic: {type(model)}")
    print("Continuous model loaded successfully.")
    return model

def save_continuous_bertopic_model(model: BERTopic) -> None:
    path = _get_continuous_model_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Saving continuous model to {path}...")
    model.save(path, serialization='pickle', save_ctfidf=True)
    print("Continuous model saved successfully.")

def load_articles() -> pd.DataFrame:
    articles = pd.read_parquet(ARTICLES_PATH)
    if articles.empty:
        raise ValueError(f"No articles found in {ARTICLES_PATH}")
    if 'url' in articles.columns:
        articles = articles.drop(columns=["url"])
    return articles

def get_document_corpus(df: pd.DataFrame) -> pd.Series:
    return df["title"]

def get_stopwords() -> list[str]:
    try:
        nltk.data.find("corpora/stopwords")
    except Exception:
        nltk.download("stopwords")
    return nltk.corpus.stopwords.words("portuguese") + ["g1", "explica", "veja", "vídeo"]

class TopicModelManager:
    def __init__(self, sentence_model: SentenceTransformer):
        self.sentence_model = sentence_model
        self.stopwords = get_stopwords()

    def _initialize_new_bertopic_model(self) -> BERTopic:
        umap_model = IncrementalPCA(n_components=5)
        cluster_model = MiniBatchKMeans(n_clusters=15, random_state=42, n_init='auto')
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        vectorizer_model = OnlineCountVectorizer(stop_words=self.stopwords, ngram_range=(1, 1))
        return BERTopic(
            embedding_model=self.sentence_model,
            umap_model=umap_model,
            hdbscan_model=cluster_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            verbose=True,
            nr_topics='auto',
            min_topic_size=5
        )

    def get_or_create_continuous_model(self) -> BERTopic:
        try:
            return load_continuous_bertopic_model()
        except Exception:
            print("Could not load continuous model. Creating a new one.")
            return self._initialize_new_bertopic_model()

class SectionGenerator:
    def __init__(self, start_date: datetime, end_date: datetime):
        self.start_date = start_date
        self.end_date = end_date
        self.sentence_model = SentenceTransformer(SENTENCE_MODEL)
        self.topic_model_manager = TopicModelManager(self.sentence_model)
        self.metrics: list[dict[str, float]] = []

    def process_day(self, current_date: datetime, articles: pd.DataFrame, model: BERTopic) -> pd.DataFrame | None:
        day = current_date.strftime("%Y-%m-%d")
        print(f"\n--- Processing articles for {day} ---")
        if articles.empty:
            print(f"No articles found for {day}. Skipping.")
            return None
        if 'newsId' not in articles.columns or articles['newsId'].isnull().all():
            print("Warning: 'newsId' column not found or is empty. Generating default document keys.")
            articles['newsId'] = [f"doc_{day}_{i}" for i in range(len(articles))]
        try:
            docs = get_document_corpus(articles).tolist()
            embeddings = self.sentence_model.encode(docs, show_progress_bar=False).astype(np.float64)
            model.partial_fit(docs, embeddings=embeddings)
            topics, _ = model.transform(docs, embeddings=embeddings)
            print(f"Model updated for {day} with {len(set(topics))} topics (including noise if present).")
            counts = pd.Series(topics).value_counts()
            unassigned = int(counts.get(-1, 0))
            valid = [t for t in set(topics) if t != -1]
            num_topics = len(valid)
            if num_topics == 0:
                print(f"No meaningful topics found for {day} after processing.")
                self.metrics.append({"day": day, "num_topics": 0, "num_unassigned": unassigned, "npmi": 0.0, "td": 0.0})
                return None
            df = pd.DataFrame({
                "timestamp": datetime.now(),
                "processing_date": day,
                "newsId": articles["newsId"].reset_index(drop=True),
                "doc": docs,
                "topic": [str(t) for t in topics],
                "topic_name_list": [[w for w, _ in model.get_topic(t)] if t != -1 else [] for t in topics]
            })
            topic_words = [[w for w, _ in model.get_topic(tid)] for tid in valid]
            tokens = [d.split() for d in docs]
            gensim_dict = Dictionary(tokens)
            cm = CoherenceModel(topics=topic_words, texts=tokens, dictionary=gensim_dict, coherence='c_npmi')
            npmi = cm.get_coherence()
            N = len(topic_words[0])
            all_w = list(itertools.chain.from_iterable(topic_words))
            td = len(set(all_w)) / (N * num_topics)
            self.metrics.append({"day": day, "num_topics": num_topics, "num_unassigned": unassigned, "npmi": npmi, "td": td})
        except Exception as e:
            print(f"Error processing articles for {day}: {e}")
            traceback.print_exc()
            print("Skipping this day due to error.")
            return None
        return df

    def run(self):
        df_all = pd.DataFrame()
        articles = load_articles()
        articles['date_col'] = pd.to_datetime(articles['publishDate'])
        model = self.topic_model_manager.get_or_create_continuous_model()
        date = self.start_date
        while date <= self.end_date:
            daily = articles[articles['date_col'].dt.date == date.date()].copy()
            res = self.process_day(date, daily, model)
            if res is not None and not res.empty:
                df_all = pd.concat([df_all, res], ignore_index=True)
            date += timedelta(days=1)
        if not df_all.empty:
            save_continuous_bertopic_model(model)
            topics_filename = f"topics_{self.start_date:%Y%m%d}_to_{self.end_date:%Y%m%d}_bertopic_online.parquet"
            df_all.to_parquet(topics_filename, index=False)
            print(f"\nAll generated sections saved to {topics_filename}")
        else:
            print("\nNo sections were generated for the entire date range.")
        if self.metrics:
            pd.DataFrame(self.metrics).to_csv("daily_topic_metrics_online.csv", index=False)
        return df_all

if __name__ == "__main__":
    gen = SectionGenerator(start_date=START_DATE, end_date=END_DATE)
    out = gen.run()
    if out is not None and not out.empty:
        print("\n--- Example of Generated Sections ---")
        print(out.head())
        print(f"\nTotal sections generated: {len(out)}")
