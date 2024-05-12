# LDA Topic Labeler

Description
This repository contains Python scripts aimed at performing advanced text analysis on collections of documents over time. The scripts handle the extraction of topics from documents using Latent Dirichlet Allocation (LDA), automatically label these topics using various NLP techniques, and analyze the similarity of topics across consecutive years. This toolkit is especially useful for longitudinal studies in domains like content analysis, trend detection, and thematic studies in large textual datasets.

Components
Topic Extraction and Labeling: Extracts topics from documents using LDA and labels them using methods such as TF-IDF, word vector clustering, and YAKE! keyword extraction.

Topic Similarity Analysis: Analyzes the similarity between topics of consecutive years to understand thematic continuity or evolution using spaCy's semantic similarity features.

Detailed Script Breakdown
topic_modeling_and_labeling.py:

Extracts topics using LDA from documents organized by year.
Labels the topics using TF-IDF, word vectors, and YAKE!.
Outputs a CSV file containing the topics and their labels.
topic_similarity_analysis.py:

Loads topics from the CSV file.
Calculates semantic similarities between topics of consecutive years using spaCy.
Outputs a CSV file containing the similarity scores.
Visualizing Data
To visualize trends or similarities, additional scripts or Jupyter notebooks can be used to plot the data from CSV files, providing insights into thematic changes over time.
