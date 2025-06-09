import os
import time
import nltk
import numpy as np
import pandas as pd
from datetime import datetime
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.vectorizers import OnlineCountVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import IncrementalPCA
from sentence_transformers import SentenceTransformer

DEFAULT_NUM_WORDS = 30
BERTOPIC_MODEL_PATH = "bertopic_albertina_online"
ARTICLES_PATH = "data/articles/articles_body/articles.parquet"

def _load_bertopic_model(day: str = time.strftime("%Y-%m-%d")) -> BERTopic:
    """
    Load a BERTopic model from a GCS bucket for a specified configuration.
    """
    model_path = os.path.join(BERTOPIC_MODEL_PATH, day, "bertopic.pkl")
    print(f"Loading model from {model_path}...")
    model = BERTopic.load(model_path)
    if not isinstance(model, BERTopic):
        raise ValueError(f"Loaded model is not an instance of BERTopic: {type(model)}")
    print("Model loaded successfully.")
    return model

def _save_bertopic_model(model: BERTopic, day: str = time.strftime("%Y-%m-%d")) -> None:
    """
    Save a BERTopic model to a GCS bucket for a specified configuration.
    """
    model_path = os.path.join(BERTOPIC_MODEL_PATH, day, "bertopic.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print(f"Saving model to {model_path}...")
    model.save(model_path, serialization='pickle', save_ctfidf=True)
    print("Model saved successfully.")

def _load_articles() -> pd.DataFrame:
    """
    Load articles from a local or remote source.
    This function should be implemented to return a DataFrame with the articles.
    """
    articles = pd.read_parquet(ARTICLES_PATH)
    if articles.empty:
        raise ValueError("No articles found. Please check the ARTICLES_PATH.")
    return articles

class GenerateSections():

    def get_doc_corpus(self, docs_config):
        _type_table = ""
        _corpus = ""
        if docs_config["type_alg"].unique() == "complete":
            _corpus = docs_config["title"] + " " + docs_config["body"]
            _type_table = "complete"
        elif docs_config["type_alg"].unique() == "title_w_body":
            _corpus = (
                docs_config["title"]
                + " "
                + docs_config["body"]
                .str.split(" ")
                .str[:DEFAULT_NUM_WORDS]
                .str.join(" ")
            )
            _type_table = "title_w_body"
        else:
            _corpus = docs_config["title"]
            _type_table = "only_title"
        return _type_table, _corpus

    def get_number_sections(self, topics, topic_model):
        set_topics = list(set(topics))
        number_sections = {
            i: sorted(topic_model.get_topic(i), key=lambda x: x[1], reverse=True)[0]
            for i in set_topics
        }
        return number_sections

    def rescale(self, x: np.ndarray) -> np.ndarray:
            # Rescale a 2D embedding array to prevent convergence issues during dimensionality reduction (e.g., UMAP).

            x /= np.std(x[:, 0]) * 10000
            return x

    def get_similar_topic_ids(self, topic, all_topics, embeddings, top_n=3):
        topic_to_doc_embeddings = {}
        for topic_id, embedding in zip(all_topics, embeddings):
            if topic_id == -1:
                continue
            topic_to_doc_embeddings.setdefault(topic_id, []).append(embedding)

        topic_to_centroid = {
            topic: np.mean(np.vstack(embs), axis=0)
            for topic, embs in topic_to_doc_embeddings.items()
        }

        sorted_topics = sorted(topic_to_centroid.keys())
        topic_vectors = np.vstack([topic_to_centroid[t] for t in sorted_topics])
        topic_index_to_position = {t: i for i, t in enumerate(sorted_topics)}

        topic_sim_matrix = cosine_similarity(topic_vectors)

        if topic == -1 or topic not in topic_index_to_position:
            return []
        topic_pos = topic_index_to_position[topic]
        similarities = topic_sim_matrix[topic_pos]
        similar_idxs = np.argsort(similarities)[::-1][1:top_n+1]
        return [sorted_topics[i] for i in similar_idxs]

    def run(self):
        df_documents = _load_articles()
        df_docs_unique = df_documents.drop_duplicates()
        stopwords = self.get_stopwords()

        sentence_model = SentenceTransformer("PORTULAN/albertina-100m-portuguese-ptbr-encoder")

        res_timestamp = datetime.now()
        df_final_docs = pd.DataFrame()
        for config in list(df_docs_unique["alg_config"].unique()):
            docs_config = df_docs_unique.query("alg_config == @config")
            type_table, corpus = self.get_doc_corpus(docs_config)

            try:
                try:
                    print(f"Loading existing model from GCS for config {config}")
                    topic_model = _load_bertopic_model(config)
                except Exception as load_exc:
                    print(f"Could not load model from GCS for config {config}: {load_exc}")
                    print(f"Training new model for config {config}")

                    umap_model = IncrementalPCA(n_components=5)
                    cluster_model = MiniBatchKMeans(n_clusters=25, random_state=42)
                    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
                    vectorizer_model = OnlineCountVectorizer(stop_words=stopwords, ngram_range=(1, 2))
                    topic_model = BERTopic(
                        embedding_model=sentence_model,
                        umap_model=umap_model,
                        hdbscan_model=cluster_model,
                        vectorizer_model=vectorizer_model,
                        ctfidf_model=ctfidf_model,
                        verbose=True,
                        nr_topics='auto',
                    )

                embeddings = sentence_model.encode(corpus, show_progress_bar=False).astype(np.float64)
                model = topic_model.partial_fit(corpus, embeddings=embeddings)
                topics, probs = model.transform(corpus, embeddings=embeddings)
                umap_embeddings = topic_model.umap_model.transform(embeddings)
                pd.DataFrame(umap_embeddings).to_parquet(f"/tmp/{config}_umap_embeddings.parquet")
                if umap_embeddings.dtype != np.float64:
                    print(f"Converting umap_embeddings from {umap_embeddings.dtype} to float64")
                    umap_embeddings = umap_embeddings.astype(np.float64)
                print(f"Model trained for config {config} with {len(set(topics))} topics")
                _save_bertopic_model(model, config)
                print(f"Model saved to GCS for config {config}")
            except Exception as e:
                import traceback
                print(f"Error in config {config}: {e}")
                print("FULL TRACEBACK:")
                traceback.print_exc()
                print("Continuing to next config...")
                continue

            if not topics:  # empty list
                print(f"No topics found for config {config}")
                print("Continuing to next config...")
                continue

            number_sections = self.get_number_sections(topics, topic_model)
            df_res = pd.DataFrame(
                {
                    "timestamp": np.repeat(res_timestamp, len(corpus)),
                    "type_table": np.repeat(type_table, len(corpus)),
                    "alg_config": np.repeat(config, len(corpus)),
                    "documentKey": docs_config["documentKey"],
                    "doc": corpus,
                    "section": [str(topic) for topic in topics],
                    "section_name": [number_sections[topic][0] if topic != -1 else None for topic in topics],
                    "section_name_list": [
                        [word for word, _ in topic_model.get_topic(topic)]
                        if topic != -1
                        else []
                        for topic in topics
                    ],
                    "section_similar_list": [
                        [str(t) for t in self.get_similar_topic_ids(topic, topics, umap_embeddings, top_n=3)]
                        for topic in topics
                    ],
                }
            )
            df_final_docs = pd.concat([df_final_docs, df_res])

        if len(df_final_docs) > 0:
            df_final_docs = df_final_docs.reset_index(drop=True)
            sections_filename = f"sections_{res_timestamp.strftime('%Y%m%d_%H%M%S')}.parquet"
            df_final_docs.to_parquet(sections_filename, index=False)
            print(f"Sections saved to {sections_filename}")

        return df_final_docs

    def get_stopwords(self):
        nltk.download("stopwords")
        return nltk.corpus.stopwords.words("portuguese") + ["g1", "explica", "veja", "vídeo"]
