
## Text Classification with PAN12 and PAN13 Datasets

This project focuses on downloading, extracting, and performing text classification on datasets from the PAN12 and PAN13 source retrieval tasks. The project involves data preprocessing, feature extraction, training various classifiers, and visualizing the results.

### Table of Contents
1. [Installation](#installation)
2. [Dataset Download and Extraction](#dataset-download-and-extraction)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Extraction](#feature-extraction)
5. [Model Training and Evaluation](#model-training-and-evaluation)
6. [Visualization](#visualization)
7. [Usage](#usage)
8. [Models and Vectorizers](#models-and-vectorizers)
9. [Results](#results)
10. [License](#license)

### Installation

To run this project, ensure you have the following packages installed:

```bash
pip install wget pandas numpy scikit-learn joblib matplotlib seaborn
```

### Dataset Download and Extraction

The script downloads and extracts the PAN12 and PAN13 datasets from provided URLs. It supports handling both `.tar.gz` and `.zip` file formats.

#### URLs and Directories

```python
urls = {
    'pan2012': 'https://zenodo.org/records/3250135/files/pan-wikipedia-quality-flaw-corpus-2012.tar.gz?download=1',
    'pan2013': 'https://zenodo.org/records/3715980/files/pan13-text-alignment-test-and-training.zip?download=1'
}

dataset_dirs = {
    'pan2012': '/content/PAN12-Source-Retrieval',
    'pan2013': '/content/PAN13-Source-Retrieval'
}
```

The script creates the necessary directories and downloads the datasets using the `wget` library. It then extracts the contents based on the file type.

#### Download and Extract Function

```python
def download_and_extract(url, file_path, dataset_dir, is_zip=False):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f'Downloading {key} dataset...')
            wget.download(url, file_path)
            print(f'\nExtracting {key} dataset...')
            if is_zip:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(dataset_dir)
            else:
                with tarfile.open(file_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(dataset_dir)
            break  # Exit loop if successful
        except Exception as e:
            print(f'Error: {e}')
            if attempt < max_retries - 1:
                print(f'Retrying in 5 seconds... ({attempt + 1}/{max_retries})')
                time.sleep(5)
            else:
                print('Failed to download after several attempts.')
```

### Data Preprocessing

The script reads text files from the extracted directories and assigns labels based on the directory structure.

#### Read Texts Function

```python
def read_texts_from_dir(dir_path):
    texts = []
    labels = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".txt"):
                label = os.path.basename(root)
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                    labels.append(label)
    return texts, labels
```

### Feature Extraction

The script uses `TfidfVectorizer` from scikit-learn to transform the text data into TF-IDF feature vectors.

```python
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(all_texts).toarray()
y = np.array(all_labels)
```

### Model Training and Evaluation

The script trains three models: K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Logistic Regression. It evaluates each model using classification metrics.

#### KNN Training

```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("KNN Classification Report:\n", classification_report(y_test, y_pred))
print("KNN Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(knn, 'knn_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
```

#### SVM and Logistic Regression Training

```python
svm = SVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
print("SVM Classification Report:\n", classification_report(y_test, svm_pred))
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict(X_test)
print("Logistic Regression Classification Report:\n", classification_report(y_test, log_reg_pred))
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_reg_pred))
```

### Visualization

The script visualizes the distribution of text lengths and the most common words in the dataset.

#### Text Length Distribution

```python
text_lengths = [len(text.split()) for text in all_texts]
plt.figure(figsize=(10, 6))
sns.histplot(text_lengths, bins=50, kde=True)
plt.title("Distribution of Text Lengths")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()
```

#### Most Common Words

```python
all_words = ' '.join(all_texts).split()
word_counts = Counter(all_words)
common_words = word_counts.most_common(20)

words, counts = zip(*common_words)
plt.figure(figsize=(10, 6))
sns.barplot(x=list(counts), y=list(words))
plt.title("Most Common Words")
plt.xlabel("Count")
plt.ylabel("Word")
plt.show()
```

### Usage

1. Ensure you have the required packages installed.
2. Run the script to download, extract, preprocess, and classify the datasets.
3. Use the saved models and vectorizers for further predictions.

### Models and Vectorizers

The trained models and vectorizers are saved using `joblib`:

- `knn_model.pkl`
- `tfidf_vectorizer.pkl`

### Results

The script outputs classification reports and accuracy scores for the trained models, which help in evaluating the performance of each classifier on the test dataset.
