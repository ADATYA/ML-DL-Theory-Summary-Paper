#Machine Learning and Deep Learning Summary <br>
 * ML Source GFG: https://www.geeksforgeeks.org/machine-learning/?ref=shm
 * Machine Learning Mathematics: https://www.geeksforgeeks.org/machine-learning-mathematics/?ref=lbp
 * Java Tutorial Points: https://www.javatpoint.com/machine-learning

## What are Supervised and Unsupervised Algorithm/Model Lists At a glance:

 ![image](https://github.com/ADATYA/Research-Paper-/assets/97549431/cf2af118-0a47-44ef-98ac-c571d3fd2fcd)
 
 ## Supervised Models:
 * Source: https://www.geeksforgeeks.org/supervised-unsupervised-learning/

  ### Classification Models:
  * CNN
  * Support Vector Machine (SVM)
  * J48(C4.5) / Decision Tree
  * Naive Bayes
  * Logistic Regression
  * Random Forest
  * KNN (K-Nearest Neighbors)
  ### Regression Models:
  * Linear Regression
  * Lasso Regression
  * Ridge Regression
  * Gradient Boosting
    
## Unsupervised Models:
   ### 
  * k-means Clustering
  * DBSCAN Clustering
  * Gaussian Matrix
  * BIRCH
  * Mean shift
  * Afiniting propagation
  * Agglometrativv Hierarchy clustering
  * Dimensionality Reduction
  * Anomaly Detection
  * Association Rule Learning
  * Topic Modeling
  * Clustering

## C4.5(J48) Decision Tree Algorithm :
  * References:<br>
    1.https://medium.com/@nilimakhanna1/j48-classification-c4-5-algorithm-in-a-nutshell-24c50d20658e<br>
    2.https://www.researchgate.net/figure/An-architecture-of-the-proposed-algorithm-using-the-IDS-approach_fig5_319716463<brr>
  * Introduce J48 model :<br>
    We're living in the era of Ai and it was very important for our daily life. Now a single day we cannot live without helping Ai. But there is a question, how does this Ai work why it is important and what is the purpose of working this? So now we discuss some algorithms, the backbone of Ai , there 
    are many types of algorithms for solving problems in real life.<br>
    J48 is an open-source implementation of the C4.5 decision tree algorithm, often used in the WEKA data mining tool.
    It's a supervised learning technique that builds a tree-like model to classify data points based on a set of features (attributes).
   
  * How J48 Works:
   
   1. Data Preparation: <br>
      * The training data consists of samples (data points) with features (attributes) and their corresponding class labels (categories).<br>
      * J48 can handle both categorical and continuous attributes.<br>
   
   2. Tree Construction: <br> 
       * The algorithm starts with the entire dataset at the root node.<br>
       * At each node:<br>
            * It selects the feature (attribute) that best separates the data points into classes. This is determined using a measure called information gain (or gain ratio).<br>
            * The feature with the highest information gain becomes the splitting criterion at that node.<br>
            * The data is divided into subsets based on the values of the chosen feature, creating child nodes.<br>
            * The process continues recursively until all data points are classified or a stopping criterion (e.g., maximum depth, minimum number of instances per node) is met.<br>
   
   3. Classification: <br>
      * To classify a new data point (not part of the training data):<br>
      * You start at the root node and follow the branches based on the values of the features in the new data point.<br>
      * You traverse the tree until you reach a leaf node, which represents the predicted class label.<br>
   
   4. Key Concepts:<br>
      
      * Information Gain: Measures how much a feature reduces uncertainty (entropy) about the class labels in a dataset. A higher gain indicates a better split.<br>
      * Gain Ratio: A variant of information gain that addresses potential biases toward features with more values.<br>
      * Pruning: A technique to prevent overfitting (the model memorizing training data too well, leading to poor performance on unseen data). J48 employs pruning strategies like reduced-error pruning.<br>
   
      Example:<br>
      
      Imagine you want to classify emails as spam or not spam based on features like the presence of certain words, sender address, etc.<br> J48 would build a decision tree that iteratively splits the data based on the most informative features to predict whether an email is likely spam or not.<br>
   
   5. Advantages of J48:<br>
   
      * Interpretability: The decision tree structure is readily understandable, allowing you to see the reasoning behind classifications.<br>
      * Efficiency: J48 can handle large datasets relatively quickly compared to some other algorithms.<br>
      * Ability to handle various data types: It can work with both categorical and continuous attributes.<br>
  
   6. Disadvantages of J48:<br>
   
      * Overfitting: J48 is susceptible to overfitting if not properly controlled (e.g., using pruning techniques).<br>
      * Sensitivity to noisy data: The performance of J48 can be affected by the presence of noise or irrelevant features in the dataset.<br>
   
   In Conclusion:<br>
   
   J48 is a powerful decision tree algorithm that provides a clear decision-making process for classification tasks. It offers advantages like interpretability and efficiency but requires careful consideration of overfitting and data quality. [Ref:Google Gemini]
  
   Some Image of C4.5(J48) Algorithm for Better understand the flow control:<br>
   
![image](https://github.com/ADATYA/Research-Paper-/assets/97549431/48ea1cfd-e00c-4b1e-8ce0-13224a58b5f0)

   ## J48 Algorithm image:
   ![image](https://github.com/ADATYA/ML-DL-Theory-Summary/assets/97549431/17609030-827a-4ef6-8068-ebb2717ea27d)<b>[Ref.ResearchGate]


## Naive Bayes Algorithm :<br>
Refarence:<br>
  * GFG : https://www.geeksforgeeks.org/naive-bayes-classifiers/<br>
  * Turing : https://www.turing.com/kb/an-introduction-to-naive-bayes-algorithm-for-beginners <br>
  * Tutorials : https://youtu.be/GBMMtXRiQX0?si=R71O6sPw-7CQdw4m (Youtube)<br>
  
  ![image](https://github.com/ADATYA/ML-DL-Theory-Summary/assets/97549431/b035355e-df0a-42ff-8b3f-063a7caeae24)


Naive Bayes is a family of supervised learning algorithms based on Bayes' theorem. It's a popular choice for classification tasks due to its simplicity, efficiency, and surprisingly good performance in many scenarios.

  1.Key Concepts:<br>

  * Classification: Assigning data points (emails, images, etc.) to predefined categories (spam/not spam, cat/dog).<br>
  * Probabilistic: Leverages probability calculations to determine the most likely class for a new data point.<br>
  * Bayes' Theorem: A mathematical formula for calculating conditional probability (the probability of event B occurring given that event A has already happened).<br>
  * Conditional Independence Assumption: Naive Bayes assumes that features (attributes of a data point) are independent of each other given the class label. While this assumption isn't always strictly true, it often works well in practice.<br><br>
2.How Naive Bayes Works:<br>

  * Training: Naive Bayes learns from a labeled dataset where each data point has a known class label. It calculates the probability of each feature value occurring for each class.<br>
  * Classification: When presented with a new, unlabeled data point, Naive Bayes applies Bayes' theorem to calculate the probability of the data point belonging to each possible class. It then assigns the class with the highest probability.<br>
 
    3. Example:<br>

   Imagine classifying emails as spam or not spam. Features might include word frequency, presence of certain keywords, or capital letter usage.<br>
   Naive Bayes would calculate the probability of each feature (e.g., "free" appearing in an email) occurring in both spam and non-spam emails. <br>
   Then, for a new email, it would combine these probabilities using Bayes' theorem to determine the most likely class (spam or not spam).<br>

  4. Advantages of Naive Bayes:<br>
  
   * Simple and easy to implement: Makes it a good choice for beginners.<br>
   * Fast: Training and prediction are computationally efficient.<br>
   * Effective for high-dimensional data: Handles data with many features well.<br>
   * Performs well with limited data: Can be effective even when training data is scarce.<br><br>
5. Disadvantages of Naive Bayes:<br>

  * Conditional independence assumption: May lead to inaccuracies if features are highly correlated.<br>
  * Sensitivity to outliers: Can be skewed by extreme values in the data.<br><br>
6. Applications of Naive Bayes in Machine Learning:<br>

  * Spam Filtering: Identifying spam emails based on keywords, sender information, etc.<br>
  * Text Classification: Classifying documents or emails by topic (news, sports, business).<br>
  * Sentiment Analysis: Determining the positive, negative, or neutral sentiment of text.<br>
  * Recommendation Systems: Recommending products or content based on user preferences.<br>
  * Fraud Detection: Identifying suspicious financial transactions.<br>
    
