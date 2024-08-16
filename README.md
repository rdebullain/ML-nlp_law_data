# final_project

# **Human Rights Tribunal Decision Prediction Using Machine Learning**

This project leverages **Natural Language Processing (NLP)** and **machine learning** models to predict the outcomes of Human Rights Tribunal decisions using data from case summaries, protected grounds, and other associated metadata. The process involved multiple stages, including data preprocessing, exploratory data analysis (EDA), feature engineering with TF-IDF, and training several machine learning models, including **Logistic Regression**, **Random Forest**, **XGBoost**, and **Support Vector Machines (SVM)**.

## **Table of Contents**
- [Overview](#overview)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering with TF-IDF](#feature-engineering-with-tf-idf)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)

---

## **Overview**

This project focuses on building machine learning models that predict whether Human Rights Tribunal decisions will favor the complainant, respondent, or be dismissed. The project follows a structured approach from data collection and preprocessing to model building and evaluation.

### **Technologies Used:**
- **Python** for data analysis and modeling
- **Pandas**, **NumPy** for data manipulation
- **Seaborn**, **Matplotlib** for visualization
- **Scikit-learn**, **Imbalanced-learn**, **XGBoost** for modeling
- **SMOTE** for handling class imbalance
- **TF-IDF Vectorizer** for text processing

---

## **Data Collection**

1. **Source:**
   - The dataset was collected from public Human Rights Tribunal decisions.
   - The case data includes protected grounds (e.g., race, age, disability), reasoning, and metadata.

2. **API Usage and Web Scraping:**
   - Data was pulled from the **CanLII API** for specific case IDs.
   - Case content (reasoning) was scraped using **BeautifulSoup** for textual data.

3. **Data Format:**
   - The dataset contains metadata (case details, protected grounds) and a large text field with case summaries (`reasoning`).

---

## **Data Preprocessing**

1. **Handling Missing Data:**
   - Columns such as `reasoning` with missing data were filled with empty strings.
   - Missing values in other fields were handled using appropriate strategies (e.g., filling or dropping).

2. **One-Hot Encoding:**
   - Categorical columns like `protected grounds` and `decisionType` were one-hot encoded for model input.

3. **TF-IDF Vectorization:**
   - The `reasoning` column (text data) was transformed using **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text into numeric features.
   - A maximum of 1000 TF-IDF features were extracted.

4. **Final Dataset:**
   - The final dataset consists of one-hot encoded features combined with TF-IDF features, making the data ready for model training.

---

## **Exploratory Data Analysis (EDA)**

1. **Descriptive Statistics:**
   - Summary statistics and basic information about the dataset were computed using **Pandas**.
   
2. **Distribution of Target Labels:**
   - The distribution of decision outcomes (`labels`) was analyzed to understand class imbalance.

3. **Protected Grounds and Areas:**
   - Pie charts and heatmaps were created to visualize the distribution of protected grounds (e.g., race, age, sex) and protected areas (e.g., employment, goods, services).

4. **Correlation Analysis:**
   - The relationship between protected grounds and areas was examined using heatmaps to reveal potential interactions.

---

## **Feature Engineering with TF-IDF**

1. **Text Processing:**
   - The `reasoning` column was cleaned, removing punctuation, numbers, and stopwords.
   - The text was lemmatized to reduce words to their base forms.
   
2. **TF-IDF Vectorization:**
   - The cleaned text was transformed using **TfidfVectorizer** with a maximum of 1000 features.
   - The resulting TF-IDF matrix was combined with the existing features from one-hot encoding.

---

## **Model Training and Evaluation**

Multiple models were trained to predict tribunal decisions. Here's a breakdown of the models and techniques used:

### **1. Logistic Regression**
   - A **Logistic Regression** model was trained using a pipeline that scaled the data using **StandardScaler**.
   - SMOTE (Synthetic Minority Over-sampling Technique) was used to balance the dataset due to class imbalance.

### **2. Random Forest**
   - A **Random Forest Classifier** was trained on the balanced dataset.
   - Feature importance was examined to understand the model's decision-making process.

### **3. XGBoost**
   - The **XGBoost** model was trained after encoding the labels using **LabelEncoder**.
   - This model was evaluated for its high performance and interpretability.

### **4. Support Vector Machine (SVM)**
   - Several **SVM** kernels (linear, polynomial, radial basis function (RBF), and sigmoid) were evaluated.
   - The best kernel was selected based on the highest **F1 score** from cross-validation on the training data.

### **Performance Metrics:**
   - **Accuracy**
   - **Precision**
   - **Recall**
   - **F1 Score**
   
   These metrics were used to evaluate the models on both the training and testing datasets.

---

## **Results**

The key results from the test data evaluation include:

| Model                | Accuracy | Precision | Recall  | F1 Score |
|----------------------|----------|-----------|---------|----------|
| Logistic Regression   | 88.77%   | 88.14%    | 88.40%  | 88.14%   |
| Random Forest         | 86.95%   | 85.72%    | 85.92%  | 85.61%   |
| XGBoost               | 86.16%   | 85.61%    | 85.84%  | 85.61%   |
| SVM (Best Kernel: RBF)| 89.30%   | 89.37%    | 89.10%  | 89.37%   |

The **SVM model** with the RBF kernel performed the best, with an **accuracy** of **89.30%** and an **F1 score** of **89.37%** on the test data.

---

## **Conclusion**

This project demonstrated the use of **Natural Language Processing (NLP)** and **machine learning** models to predict legal outcomes from tribunal decisions. By employing **TF-IDF** for text processing and a variety of machine learning models, we were able to achieve high accuracy in predictions, with **SVM** emerging as the best-performing model.

### **Key Takeaways:**
- **TF-IDF** is highly effective for transforming legal text into features for classification tasks.
- **Class imbalance** was handled using **SMOTE**, which improved the model's performance.
- **SVM** with the **RBF kernel** provided the best results, highlighting the importance of trying different kernels for complex datasets like tribunal cases.

---

## **Future Work**
- Further hyperparameter tuning for models like **XGBoost** and **SVM** could lead to even better results.
- Expanding the dataset to include more case decisions will improve the model's generalizability.
- Incorporating more advanced NLP techniques, such as **BERT embeddings**, could enhance text feature extraction.
