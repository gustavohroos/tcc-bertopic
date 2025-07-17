ARTICLES_W_BODY_DRIVE_ID = 1fNT_06whmSFq_tEsJyAiNZQ6fClH4S-M
DATA_DIR = data
ARTICLES_DIR = $(DATA_DIR)/articles
RECOMMENDATIONS_DIR = $(DATA_DIR)/recommendations/recommendations/articles
KAGGLE_ZIP = $(DATA_DIR)/news-portal-recommendations-npr-by-globo.zip
ARTICLES_ZIP = $(DATA_DIR)/articles_body.zip

.PHONY: setup clean

requirements:
	@echo "Installing required Python packages..."
	@pip install -r requirements.txt -qqq
	@echo "Required packages installed."

download-data:
	@echo "Creating necessary directories..."
	@mkdir -p $(ARTICLES_DIR)/articles_body
	@mkdir -p $(RECOMMENDATIONS_DIR)

	@echo "Checking if Kaggle data is already extracted..."
	@bash -c 'if [ ! -f "$(ARTICLES_DIR)/articles/articles_wobody_wourl.parquet" ]; then \
		echo "Downloading dataset from Kaggle..."; \
		curl -s -L -o $(KAGGLE_ZIP) https://www.kaggle.com/api/v1/datasets/download/joelpl/news-portal-recommendations-npr-by-globo; \
		echo "Unzipping dataset..."; \
		unzip -oq $(KAGGLE_ZIP) -d $(DATA_DIR); \
		rm -f $(KAGGLE_ZIP); \
	else \
		echo "Kaggle data already exists, skipping download."; \
	fi'

	@echo "Checking if article bodies are already extracted..."
	@bash -c 'if [ ! -f "$(RECOMMENDATIONS_DIR)/articles_small.parquet" ]; then \
		echo "Downloading articles body from Google Drive..."; \
		gdown --id $(ARTICLES_W_BODY_DRIVE_ID) -O $(ARTICLES_ZIP) --quiet; \
		echo "Unzipping articles body..."; \
		unzip -qq -o $(ARTICLES_ZIP) -d $(ARTICLES_DIR); \
		rm -f $(ARTICLES_ZIP); \
		find $(ARTICLES_DIR) -type d -name "__MACOSX" -exec rm -rf {} +; \
		echo "Moving parquet file..."; \
		mv $(ARTICLES_DIR)/articles_body/articles_small.parquet $(RECOMMENDATIONS_DIR)/articles_small.parquet; \
	else \
		echo "Articles body already exists, skipping download."; \
	fi'

	@echo "Setup complete."

setup: requirements download-data
	@echo "All setup tasks completed successfully."

clean:
	@rm -rf $(DATA_DIR)
	@echo "Cleaned up data directory."

run:
	@echo "Running the main script..."
	@python bertopic_v1.py
	@python bertopic_v2.py
	@python bertopic_online.py
	@python metrics.py
	@echo "All scripts executed successfully."