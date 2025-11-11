# GRASP-ChoQ: Political Stance Detection in Tweets

This repository contains a collection of Python scripts designed for a comprehensive pipeline to analyze political tweets, specifically focusing on detecting stances related to the Awami League in Bangladesh. The project leverages Natural Language Processing (NLP), Large Language Models (LLMs), and Knowledge Graphs to preprocess data, classify stances, and build a rich, queryable knowledge base.

The core of this project is the **GRASP-ChoQ** (Graph-based Relational and Semantic Prompting with Chain-of-Question) method, a novel prompting technique for enhanced political stance detection.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Pipeline Architecture](#pipeline-architecture)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Tweet Translation](#2-tweet-translation)
  - [3. Entity Extraction and Word Cloud](#3-entity-extraction-and-word-cloud)
  - [4. Knowledge Graph Construction](#4-knowledge-graph-construction)
  - [5. Stance Classification](#5-stance-classification)
- [License](#license)

## Project Overview

This project provides a complete workflow for political text analysis:

1.  **Preprocessing**: Cleaning and filtering raw tweet datasets to retain relevant, high-quality data.
2.  **Translation**: Translating non-English tweets into English for consistent processing.
3.  **Entity Analysis**: Identifying key entities (people, organizations) from the text, analyzing their frequency, and visualizing them as a word cloud.
4.  **Knowledge Graph**: Building a Neo4j knowledge graph from the extracted entities using Wikipedia data to create structured relational information.
5.  **Stance Detection**: Classifying the political stance of tweets (in favor of or against a target entity) using various LLM-based prompting strategies, including Zero-Shot, Few-Shot, and the advanced GRASP-ChoQ method.

## Features

-   **Data Cleaning**: Robust preprocessing script to filter tweets by content length, user bio, and media attachments.
-   **LLM-Powered Translation**: Translates text while preserving proper nouns.
-   **Entity Recognition (NER)**: Identifies and tags named entities in the text corpus.
-   **Knowledge Graph Builder**: Automatically constructs a Neo4j graph from entities using Wikipedia as a data source.
-   **Vector-based Retrieval**: Implements hybrid search (semantic and keyword-based) on the knowledge graph.
-   **Multiple Stance Detection Models**:
    -   **Zero-Shot**: Classify stance without examples.
    -   **Few-Shot**: Improve accuracy with in-prompt examples.
    -   **Few-Shot with RAG**: Enhance classification by providing external context.
    -   **GRASP-ChoQ**: A sophisticated model using chain-of-question prompts and multiple information sources for nuanced stance detection.

## Pipeline Architecture

The project is structured as a modular pipeline. Here is the typical workflow:

1.  **`preprocess.py`**: Start with a raw Twitter dataset (`.xlsx` or `.csv`) and generate a cleaned `preprocessed_dataset.xlsx`.
2.  **`translation.py`**: Take the preprocessed data and translate the tweets into English, saving the result to `translated_dataset.xlsx`.
3.  **`entities.py`**: Analyze the translated text to generate a `wordcloud.png` and an `entities.csv` file containing top words and their NER tags.
4.  **`knowledge_graph_builder.py`**: Use the `entities.csv` to populate a Neo4j database with a knowledge graph.
5.  **Stance Detection Scripts**: Use one of the classification scripts (`zero_shot.py`, `few_shot.py`, `grasp_choq.py`) on the translated data. These scripts can leverage the knowledge graph via `retrieve_from_graph.py` to get contextual information.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Programming-Dude/GRASP-ChoQ.git
    cd GRASP-ChoQ
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
    *Note: You will need to create a `requirements.txt` file.*
    ```text name=requirements.txt
    pandas
    wordcloud
    matplotlib
    nltk
    spacy
    langchain-openai
    langchain-community
    langchain-experimental
    langchain
    python-dotenv
    openai
    neo4j
    langchain-huggingface
    torch
    sentence-transformers
    openpyxl
    ```

3.  **Download NLP models:**
    Run the following in a Python interpreter to download necessary models for `nltk` and `spacy`:
    ```python
    import nltk
    import spacy

    nltk.download('stopwords')
    nltk.download('punkt')
    spacy.cli.download("en_core_web_sm")
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the root directory and add your API keys and database credentials. This project uses [OpenRouter.ai](https://openrouter.ai/) for LLM access and Neo4j for the knowledge graph.

    ```env name=.env
    # Get your key from https://openrouter.ai/keys
    OPENROUTER_API_KEY="your-openrouter-api-key"

    # Neo4j database credentials
    NEO4J_URI="bolt://localhost:7687"
    NEO4J_USERNAME="neo4j"
    NEO4J_PASSWORD="your-neo4j-password"
    ```

5.  **Set up Neo4j:**
    Ensure you have a running Neo4j instance. You can use Neo4j Desktop or a Docker container.

## Usage

Each script can be run individually. Follow the pipeline steps for a full workflow.

### 1. Data Preprocessing

Place your raw dataset (e.g., `BPDisC_with_stance.xlsx`) in the root folder and run:
```bash
python preprocess.py
```
This script filters tweets based on criteria like word count and user bio, saving the output to `preprocessed_BPDisC_dataset.xlsx`.

### 2. Tweet Translation

Translate the preprocessed tweets into English:
```bash
python translation.py
```
This will create a `BPDisC_translated.xlsx` file with a new `translation` column.

### 3. Entity Extraction and Word Cloud

Analyze the translated text to find key entities:
```bash
python entities.py
```
This script reads the translated Excel file, generates a `wordcloud.png` image, and saves the top 100 words and their NER tags to `stop_words.csv`.

### 4. Knowledge Graph Construction

Build the knowledge graph from the extracted entities. Make sure your Neo4j database is running.
```bash
python knowledge_graph_builder.py
```
This will populate the graph with nodes and relationships based on Wikipedia articles for the top entities.

### 5. Stance Classification

You can classify tweet stances using different methods. Each script includes an example in its `if __name__ == "__main__":` block.

#### Zero-Shot Classification
For simple, direct classification:
```bash
python zero_shot.py
```

#### Few-Shot Classification
For improved accuracy using examples:
```bash
python few_shot.py
```

#### GRASP-ChoQ (with RAG)
For the most advanced classification using retrieval-augmented generation and chain-of-question prompting:
1.  Use `retrieve_from_graph.py` to fetch context for a given tweet.
2.  Feed the tweet and the retrieved context into the `detect_stance_grasp_choq` function in `grasp_choq.py`.

Example:
```bash
python grasp_choq.py
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
