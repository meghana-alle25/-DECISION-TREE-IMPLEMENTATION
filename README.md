# DECISION-TREE-IMPLEMENTATION

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: MEGHANA ALLE

*INTERN ID*: CT04DF435

*DOMAIN*: MACHINE LEARNING

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTHOSH

*Implementation of Decision Tree Using the Breast Cancer Dataset*
The Decision Tree algorithm is one of the most interpretable and widely used machine learning methods for classification tasks. In this project, we implement a Decision Tree classifier using the Breast Cancer Wisconsin dataset, a standard dataset used in medical research to predict whether a tumor is malignant or benign based on various features. This dataset is provided by the UCI Machine Learning Repository and is also conveniently available through libraries like Scikit-learn in Python.

*Dataset Description*

The Breast Cancer dataset consists of 30 numeric features computed from digitized images of fine needle aspirates (FNA) of breast masses. These features represent characteristics of the cell nuclei present in the image, such as radius, texture, perimeter, area, smoothness, and symmetry. The dataset includes 569 instances, with 357 benign and 212 malignant cases. The target variable is binary, indicating the presence (malignant) or absence (benign) of breast cancer.

*Preprocessing*

Before training the Decision Tree, the dataset needs preprocessing. The features are extracted, and the target variable is encoded. Typically, no missing values exist in this dataset, which simplifies the process. However, feature scaling is generally not required for Decision Trees since they are not distance-based algorithms. The dataset is split into training and test sets, usually in a 70:30 or 80:20 ratio, to evaluate the performance of the model on unseen data.

*Model Training*

The DecisionTreeClassifier from the Scikit-learn library is used for training the model. The classifier builds a tree-like structure, where each internal node represents a decision based on a specific feature, and the leaf nodes represent the classification outcomes. The splitting of nodes is determined by criteria such as Gini Impurity or Information Gain (Entropy). The model continues to split the nodes until all data points are perfectly classified or a stopping condition like maximum depth or minimum samples per leaf is met.

*Evaluation*

After training the model, predictions are made on the test set. The performance of the Decision Tree is evaluated using various classification metrics such as accuracy, precision, recall, and the F1-score. A confusion matrix is also used to understand the number of true positives, true negatives, false positives, and false negatives. Typically, Decision Trees perform well on this dataset, often achieving accuracy above 90%.

*Visualization*

One of the main advantages of using Decision Trees is their interpretability. The trained Decision Tree can be visualized using tools such as plot_tree in Scikit-learn or exporting to Graphviz format. This visual representation helps medical professionals understand how the model arrives at a decision, making it more trustworthy in clinical settings.

*Conclusion*

Implementing a Decision Tree classifier using the Breast Cancer dataset demonstrates how machine learning can be applied in the medical field to assist with diagnostic decisions. It is a clear example of how interpretable models can provide high accuracy while also offering transparency and ease of understanding. The approach is effective, efficient, and can serve as a foundation for developing more advanced ensemble methods like Random Forests or Gradient Boosted Trees.

*OUTPUT*

![Image](https://github.com/user-attachments/assets/053a2292-f454-4b53-9e71-0bf0878b8f36)
