# Analyzing-COVID-19-Search-Trends-and-Hospitalizations

## Introduction
The objective of the project was to explore different supervised and unsupervised learning frameworks for two COVID-19
related datasets, to tackle real-world datasets. The frst dataset consists of aggregated, anonymous Google search trends
for the United States. This dataset contains various symptom counts respective of each state. The second dataset consists
of aggregate public COVID-19 hospitalizations, cases, deaths, and other attributes. The unsupervised learning models
explored in this report are Principal Component Analysis and K-Means clustering to visualize and further understand the
data. The supervised learning frameworks explored are K-nearest-neighbours regression and Decision Trees regression.
The goal is to train these models to predict hospitalization cases given the search trends data. In our results we have
KNN and Decision Tree models that achieve 5-fold-cross-validation errors of 0.0057 & 0.0133 respectively.

## Tasks/Sections
 - Task 1: Clean & Pre-process Data:
      * The Processing of the Data involves removing zero or NaN 'Missing' entries or data, normalization, resolution shifting, and mean filling.
 - Task 2: Dataset Visualization:
      * To visualize the Data we view the evolution of the symptoms over time for each region. We then perform Principal component reduction analysis (PCA). Finally we deploy K-Means to cluster the data. 
- Task 3: Supervised Learning:
      * The supervised learning task consists of training a regression K-Nearest-Neighbours & regression Decision Tree algorithms. Furthermore, we analyze the performance of our models.
- For more detail, please find the report in the repository.

## Calling Convention:
### Examples: 
loader = Loader()  
dataset_ready = loader.get_merged_datasets()  
loader.print_basic_stats()  
loader.run_knn_crossval()  
loader.run_decision_tree_crossval()  
loader.run_knn_by_date()  
loader.run_decision_tree_by_date()  
loader.run_pca()  
loader.run_Kmeans(k = 5, kmax = 10)  
loader.plotAllRegionsForComparison(['Hawaii','Montana','Idaho','Maine','Wyoming'])  

## Main Methods:
  - The main class is our 'Loader()' class which includes all the methods deployed. Methods in this class include:
      * get_merged_datasets: Performs pre-processing on the two datasets and merges them to be ready for use.
      * plotAllRegionsForComparison: Plots the evolution of symptoms for specified regions over time.
      * run_PCA: Performs the Principal component reduction.
      * run_Kmeans : Performs K-Means clustering.
      * run_decision_tree_by_date & run_decision_tree_crossval: Constructs the decision tree and segments by date for 5-fold-cross-validation.
      * run_knn_by_date & run_knn_crossval: Constructs KNN and segments by date for 5-fold-cross-validation.

## Libraries Utilized
  - In our project we utilize the following Python Libraries:
        * Sci-Kit-Learn, Numpy, Pandas, & Matplotlib.  
