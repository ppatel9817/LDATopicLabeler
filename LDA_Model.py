#!/usr/bin/env python
# coding: utf-8


import os
import pandas as pd
from gensim import corpora, models
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import yake
from tqdm import tqdm
import numpy as np



'''
This part of the code is defining function to perform Automatic labelling of the topics

Automatic labeling of topics in topic modeling, such as with LDA, can be challenging because the algorithm doesn't understand the context or meaning behind the groupings of words—it only knows which words tend to co-occur frequently. However, there are strategies you can employ to automate the labeling process to some extent. Here are a few methods you could consider:

1. Top-N Words
The simplest approach is to label each topic by the most frequent or salient words within that topic. Typically, the top 2-3 words are used to form a label. This can often provide a good enough indication of what the topic might represent, although it may not always yield intuitive or meaningful labels.

2. Word Pairing and Phrasing
Instead of using single words, you can look for common phrases or pairs of words that frequently appear together within the topics. This might involve additional NLP processing to identify common bigrams or trigrams (sequences of two or three words) that can serve as more descriptive labels.

3. TF-IDF to Identify Distinctive Words
You can use the TF-IDF (Term Frequency-Inverse Document Frequency) score to find words that are uniquely representative of a topic compared to other topics. This method can help in identifying words that not only appear frequently in a topic but are also distinctive to that topic, providing potentially more meaningful labels.

4. Clustering of Word Vectors
By using word embeddings (like Word2Vec or GloVe), you can represent the top words of each topic in a vector space and perform clustering (e.g., K-means) on these words to find the 'centroid' or most central word of a cluster that might serve as a label.

5. Automated Keyword Extraction Tools
Utilize automated keyword extraction tools that apply algorithms designed to extract the most important words or phrases from a text. Tools or libraries like YAKE!, RAKE, or Gensim’s summarization.keywords can be used to extract keywords from the aggregation of texts in a topic and use these as labels.

'''

def label_with_tfidf(texts, top_n=3):
    """Extract distinctive words using TF-IDF."""
    vectorizer = TfidfVectorizer(stop_words='english')
    texts_joined = [' '.join(text) for text in texts]
    X = vectorizer.fit_transform(texts_joined)
    sorted_indices = np.argsort(vectorizer.idf_)[::-1]
    features = vectorizer.get_feature_names_out()
    top_features = [features[i] for i in sorted_indices[:top_n]]
    return top_features

def label_with_word_vectors(texts, top_n=3):
    """Use word vectors and K-means clustering to identify central words as labels."""
    model = Word2Vec(texts, vector_size=100, window=5, min_count=5, workers=4)
    word_vectors = model.wv
    kmeans = KMeans(n_clusters=10, random_state=0)
    kmeans.fit(word_vectors.vectors)
    common_words = kmeans.cluster_centers_.argsort()[:, -1:-top_n-1:-1]
    return [" ".join([word_vectors.index_to_key[index] for index in word_indices]) for word_indices in common_words]

def label_with_yake(texts):
    """Extract key phrases using YAKE! keyword extraction."""
    kw_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, dedupFunc='seqm', windowsSize=1, top=20, features=None)
    all_text = " ".join([" ".join(text) for text in texts])
    keywords = kw_extractor.extract_keywords(all_text)
    return [kw[0] for kw in keywords[:5]]  # return top 5 keywords


'''
This part of the code is used to execute LDA and calling Automated Labelling functions as defined above to see if they are able to label the given topics

'''

yearly_topics = []

for year in sorted(os.listdir(processed_text_directory)):
    year_dir = os.path.join(processed_text_directory, year)
    if os.path.isdir(year_dir):
        documents = []
        for root, dirs, files in tqdm(os.walk(year_dir), desc=f"Processing Texts for {year}"):
            for filename in files:
                if filename.endswith(".txt"):
                    file_path = os.path.join(root, filename)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        documents.append(file.read())

        # Tokenize and prepare texts
        texts = [simple_preprocess(doc) for doc in documents]
        dictionary = corpora.Dictionary(texts)
        dictionary.filter_extremes(no_below=5, no_above=0.7)

        if len(dictionary) == 0:
            print(f"No valid words for topic modeling in year {year}. Skipping.")
            continue

        corpus = [dictionary.doc2bow(text) for text in texts]
        
        if not corpus:
            print(f"No valid corpus for topic modeling in year {year}. Skipping.")
            continue

        print(f"{year} is being processed")
        lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
        
        top_topics = lda_model.print_topics(num_words=4)
        top_5_topics = [topic[1] for topic in top_topics[:5]]

        # Generate labels
        tfidf_labels = label_with_tfidf(texts)
        vector_labels = label_with_word_vectors(texts)
        yake_labels = label_with_yake(texts)

        # Append to the yearly data
        yearly_topics.append([year] + top_5_topics + [tfidf_labels, vector_labels, yake_labels])

# Create DataFrame and save to CSV
df_columns = ['Year', 'Topic 1', 'Topic 2', 'Topic 3', 'Topic 4', 'Topic 5', 'TF-IDF Labels', 'Vector Labels', 'YAKE Labels']
df = pd.DataFrame(yearly_topics, columns=df_columns)
df.to_csv('/Users/poojanpatel/Desktop/Final_attempt/topics_by_year.csv', index=False)
print("Topic modeling and CSV saving complete.")


'''
This code is for visualisation of the topics that we see over the years.

The code  performs several steps to analyze and visualize data from a CSV file containing topics extracted from text documents by year. Here's a breakdown of what each part of the script does:

Imports Libraries:
It imports pandas for data manipulation and matplotlib.pyplot for plotting, which are essential Python libraries for data analysis and visualization.

Load the CSV File:
The DataFrame df is created by loading data from a CSV file located at /Users/poojanpatel/Desktop/Final_attempt/topics_by_year.csv. This file is expected to contain yearly data about topics derived from text analysis.

Review the Data Structure:
print(df.head()) displays the first few rows of the DataFrame. This is typically used to get a quick overview of the data format, column names, and types of data included in the DataFrame.

Quantitative Analysis - Count Specific Words:
A new column, count_specific_word, is added to the DataFrame. This column is populated by counting occurrences of specific keywords ("Employee" and "eco") in the Topic 1 column of each row.
The lambda function inside the apply method iterates over each row of the DataFrame. For each row, it checks if the words "Employee" or "eco" appear in the Topic 1 column and sums their occurrences. This is a simple form of content analysis to quantify the focus on certain topics across documents.

Plotting a Trend Line:
A trend line is plotted to visualize changes in the frequency of the specified words in Topic 1 over time.
plt.figure(figsize=(10, 5)) sets up the figure size for the plot.
plt.plot(df['Year'], df['count_specific_word'], marker='o', linestyle='-') creates a line plot with years on the x-axis and the counts of specific words on the y-axis, using circles as markers and a solid line as the connector.
The plot is titled and labeled with appropriate axis names, and a grid is added for better readability.

Visualization:
plt.show() displays the plot. This visual representation helps in understanding trends or patterns in how frequently the chosen keywords appear in the topics over the years, potentially indicating shifts in focus or interest in these areas within the dataset.

Additional Considerations:
The code assumes that the CSV file is well-structured and that the year and topic data are correctly formatted. Errors in data formatting might cause this code to fail, especially in the keyword counting and plotting sections.
The specific choice of keywords ("Employee" and "eco") suggests a focus on sustainability and employee-related topics, possibly indicating the user's interest in analyzing trends related to environmental and workforce-related discussions in the dataset.
Overall, this script provides a basic framework for quantitative textual analysis and trend visualization based on predefined keywords, which can be further expanded for more sophisticated analyses or to include additional keywords and topics.

'''


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('/Users/poojanpatel/Desktop/Final_attempt/topics_by_year.csv')


# Example of quantitative analysis: Count the appearance of a specific word
df['count_specific_word'] = df.apply(lambda row: sum(word in row['Topic 1'] for word in ['Employee', 'eco']), axis=1)

# Plotting a trend line for a specific topic keyword
plt.figure(figsize=(10, 5))
plt.plot(df['Year'], df['count_specific_word'], marker='o', linestyle='-')
plt.title('Trend of Sustainability Topics Over Time')
plt.xlabel('Year')
plt.ylabel('Frequency of Word in Topics')
plt.grid(True)
plt.show()

# More sophisticated analysis and visualizations can be implemented as needed


# In[25]:


'''
This code is for similarity between topics over years

The code is designed to analyze the similarity between topics of successive years in a dataset of topics extracted from documents, presumably from annual reports or similar texts. It uses the spaCy library's language model to compute semantic similarities between topics, which are textual descriptions of thematic content for each year. Here’s a detailed breakdown of what the code does:

Step-by-Step Explanation:

Import Libraries:
pandas for data handling.
spaCy for natural language processing, specifically to leverage its semantic similarity capabilities.

Load spaCy Model:
The medium-sized English language model (en_core_web_md) is loaded. This model includes word vectors, which are necessary for computing semantic similarity.

Load the CSV File:
The CSV file, presumably containing yearly topics data, is loaded into a pandas DataFrame (df). This DataFrame is expected to have columns representing different topics for each year.
Initialize a DataFrame for Storing Similarities:
A new DataFrame (similarity_df) is created with predefined columns to store the year pairs (Year1, Year2) and their respective topic similarities (Similarity Topic 1 to Similarity Topic 5).

Calculate Similarities:
The script iterates through years from 2003 to 2022 (inclusive) and calculates the similarity between topics of consecutive years (e.g., 2003 and 2004, 2004 and 2005, etc.).
For each pair of years, it retrieves the topic descriptions from the df DataFrame and checks if they exist.
If both topics exist for the year pair, it uses spaCy to convert the topic strings into Doc objects, which encapsulate the processed text segments.
It then calculates the semantic similarity between these Doc objects using spaCy's built-in .similarity() method, which computes cosine similarity between the vectors of the documents.
If a topic is missing for either year in the pair, it appends None for that topic's similarity.

Store and Append Results:
The similarities for each topic, along with the year pair, are appended to the similarity_df DataFrame.
Save Results:
Optionally, the resulting similarities DataFrame is saved to a new CSV file, yearly_topic_similarities.csv, for further analysis or record-keeping.

Output:
Finally, the similarity_df DataFrame is printed, showing the computed similarities between topics of consecutive years.

Practical Use:
This approach is useful for longitudinal studies where understanding the evolution or consistency of topics over time is crucial. By measuring how similar topics are from one year to the next, one can gauge shifts in thematic focus, continuity in discourse, or abrupt changes in content emphasis, which might reflect underlying shifts in organizational strategy, regulatory environment, or industry dynamics.

Caveats:
The accuracy and effectiveness of this analysis heavily rely on the quality and consistency of topic descriptions in the dataset.
Semantic similarity measures like those provided by spaCy are based on the angles between word vectors in high-dimensional space and might not always capture nuanced differences in meaning or context.

'''


import pandas as pd
import spacy

# Load spaCy model
nlp = spacy.load('en_core_web_md')  # This assumes the model is already installed

# Load the CSV file
df = pd.read_csv('/Users/poojanpatel/Desktop/Final_attempt/topics_by_year.csv')

# Initialize a DataFrame to store the similarities
columns = ['Year1', 'Year2'] + [f'Similarity Topic {i}' for i in range(1, 6)]
similarity_df = pd.DataFrame(columns=columns)

# Calculate similarity for each year from 2003 to 2023
for year1 in range(2003, 2023):
    year2 = year1 + 1
    similarities = [year1, year2]
    
    for i in range(1, 6):  # Assuming 5 topics per year
        topic1_values = df.loc[df['Year'] == year1, f'Topic {i}'].values
        topic2_values = df.loc[df['Year'] == year2, f'Topic {i}'].values
        
        if topic1_values.size > 0 and topic2_values.size > 0:
            topic1 = topic1_values[0]
            topic2 = topic2_values[0]
            
            # Create spaCy doc objects for each topic
            doc1 = nlp(topic1)
            doc2 = nlp(topic2)
            
            # Calculate similarity and store it
            similarity = doc1.similarity(doc2)
            similarities.append(similarity)
        else:
            similarities.append(None)  # Append None if no data available
    
    # Append the year and similarities to the DataFrame
    new_row = pd.Series(similarities, index=columns)
    similarity_df = pd.concat([similarity_df, pd.DataFrame([new_row])], ignore_index=True)

# Optionally, save the results to a CSV file
similarity_df.to_csv('yearly_topic_similarities.csv', index=False)

print(similarity_df)
