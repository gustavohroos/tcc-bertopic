import nltk
import pandas as pd
import numpy as np
from hdbscan import HDBSCAN
from datetime import datetime
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from umap import UMAP
import traceback
from tqdm import tqdm
import re

BERTOPIC_MODEL_BASE_PATH = "models/bertopic_v2"
ARTICLES_PATH = "data/articles/articles_body/articles.parquet"
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2023, 5, 2)
SENTENCE_MODEL = "PORTULAN/albertina-100m-portuguese-ptbr-encoder"

def load_articles() -> pd.DataFrame:
    try:
        articles = pd.read_parquet(ARTICLES_PATH)
        if articles.empty:
            raise ValueError(f"No articles found in {ARTICLES_PATH}")
        if 'url' in articles.columns:
            articles = articles.drop(columns=["url"])
        return articles
    except Exception as e:
        raise RuntimeError(f"Failed to load articles: {e}")

def get_document_corpus(docs_df: pd.DataFrame) -> pd.Series:
    return docs_df["title"].apply(lambda x: re.sub(r"[^\w\s]", "", str(x)))

def get_stopwords() -> list[str]:
    try:
        nltk.data.find("corpora/stopwords")
    except nltk.downloader.DownloadError:
        nltk.download("stopwords")
    return nltk.corpus.stopwords.words("portuguese") + ["g1", "explica", "veja", "vídeo"]

class TopicModelManager:
    def __init__(self):
        self.stopwords = get_stopwords()
        self.sentence_model = SentenceTransformer(SENTENCE_MODEL)

    def get_new_bertopic_model(self, pca_embeddings) -> BERTopic:
        umap_model = UMAP(
            n_neighbors=5,
            n_components=5,
            min_dist=0.0,
            metric="euclidean",
            init=pca_embeddings
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=10,
            min_samples=5,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True
        )
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        return BERTopic(
            embedding_model=self.sentence_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=CountVectorizer(stop_words=self.stopwords, ngram_range=(1, 1)),
            ctfidf_model=ctfidf_model,
        )

class SectionGenerator:
    def __init__(self, start_date: datetime, end_date: datetime):
        self.start_date = start_date
        self.end_date = end_date
        self.topic_model_manager = TopicModelManager()
        self.sentence_model = self.topic_model_manager.sentence_model

    def rescale(self, x: np.ndarray) -> np.ndarray:
            # Rescale a 2D embedding array to prevent convergence issues during dimensionality reduction (e.g., UMAP).

            x /= np.std(x[:, 0]) * 10000
            return x

    def process_day(self, current_date: datetime, articles_for_day: pd.DataFrame) -> pd.DataFrame | None:
        day_str = current_date.strftime("%Y-%m-%d")

        if articles_for_day.empty:
            print(f"No articles found for {day_str}. Skipping.")
            return None

        if 'newsId' not in articles_for_day.columns or articles_for_day['newsId'].isnull().all():
            articles_for_day['newsId'] = [f"doc_{day_str}_{i}" for i in range(len(articles_for_day))]

        try:
            corpus = get_document_corpus(articles_for_day).tolist()
            embeddings = self.sentence_model.encode(corpus, show_progress_bar=False)
            pca_embeddings = self.rescale(PCA(n_components=5).fit_transform(embeddings))
            topic_model = self.topic_model_manager.get_new_bertopic_model(pca_embeddings)
            topics, probs = topic_model.fit_transform(corpus, embeddings=pca_embeddings)

            valid_topic_ids = [t for t in set(topics) if t != -1]
            num_topics = len(valid_topic_ids)

            if num_topics == 0:
                print(f"No meaningful topics found for {day_str} after processing.")
                return None

            df_res = pd.DataFrame({
                "timestamp": datetime.now(),
                "processing_date": day_str,
                "newsId": articles_for_day["newsId"].reset_index(drop=True),
                "doc": corpus,
                "topic": [str(t) for t in topics],
                "topic_name_list": [
                    [word for word, _ in topic_model.get_topic(t)] if t != -1 else []
                    for t in topics
                ]
            })

        except Exception as e:
            print(f"Error processing articles for {day_str}: {e}")
            traceback.print_exc()
            print("Skipping this day due to error.")
            return None

        return df_res if not df_res.empty else None

    def run(self):
        all_articles = load_articles()
        if 'publishDate' not in all_articles.columns:
            raise ValueError("Required column 'publishDate' not found")
        all_articles['date_col'] = pd.to_datetime(all_articles['publishDate'])

        df_all_sections = pd.DataFrame()
        current_date = self.start_date

        date_range = pd.date_range(self.start_date, self.end_date, freq="D")
        for current_date in tqdm(date_range, desc="Processing days"):
            day_df = all_articles[all_articles['date_col'].dt.date == current_date.date()].copy()
            daily = self.process_day(current_date, day_df)
            if daily is not None and not daily.empty:
                df_all_sections = pd.concat([df_all_sections, daily], ignore_index=True)

        if not df_all_sections.empty:
            topics_filename = f"topics_{self.start_date:%Y%m%d}_to_{self.end_date:%Y%m%d}_bertopic_v2.parquet"
            df_all_sections.to_parquet(topics_filename, index=False)
            print(f"Saved topics to {topics_filename}")

        return df_all_sections

if __name__ == "__main__":
    generator = SectionGenerator(start_date=START_DATE, end_date=END_DATE)
    final_df = generator.run()
    if not final_df.empty:
        print("\n--- Example of Generated Sections ---")
        print(final_df.head())
        print(f"\nTotal sections generated: {len(final_df)}")
