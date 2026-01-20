# Machine Learning Basics

## What is Machine Learning?

Machine Learning (ML) is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. Instead of following hard-coded rules, ML algorithms identify patterns in data and make predictions or decisions.

## Types of Machine Learning

### 1. Supervised Learning
- **Definition**: Learning from labeled training data
- **Examples**: 
  - Image classification (cat vs dog)
  - Spam email detection
  - House price prediction
- **Common Algorithms**: Linear Regression, Decision Trees, Random Forest, SVM, Neural Networks

### 2. Unsupervised Learning
- **Definition**: Finding hidden patterns in unlabeled data
- **Examples**:
  - Customer segmentation
  - Anomaly detection
  - Topic modeling
- **Common Algorithms**: K-Means Clustering, PCA, DBSCAN, Autoencoders

### 3. Reinforcement Learning
- **Definition**: Learning through trial and error with rewards/penalties
- **Examples**:
  - Game playing (AlphaGo, chess)
  - Robotics control
  - Autonomous driving
- **Common Algorithms**: Q-Learning, Deep Q-Networks (DQN), PPO, A3C

## Key Concepts

### Training and Testing
- **Training Set**: Data used to train the model (typically 70-80%)
- **Validation Set**: Data used to tune hyperparameters (10-15%)
- **Test Set**: Unseen data used to evaluate final performance (10-15%)

### Overfitting vs Underfitting
- **Overfitting**: Model learns training data too well, fails on new data
  - Solution: Regularization, dropout, more training data
- **Underfitting**: Model is too simple, can't capture patterns
  - Solution: Use more complex model, add features

### Evaluation Metrics
- **Classification**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Regression**: MSE (Mean Squared Error), RMSE, MAE, R²

## Popular ML Libraries

### Python
- **Scikit-learn**: General-purpose ML library
- **TensorFlow**: Deep learning framework by Google
- **PyTorch**: Deep learning framework by Meta
- **XGBoost**: Gradient boosting library
- **Keras**: High-level neural network API

### Example Code (Python)

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
X, y = load_data()

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2%}")
```

## Common ML Workflows

1. **Data Collection**: Gather relevant data
2. **Data Preprocessing**: Clean, normalize, handle missing values
3. **Feature Engineering**: Create meaningful features from raw data
4. **Model Selection**: Choose appropriate algorithm
5. **Training**: Fit model to training data
6. **Evaluation**: Test on unseen data
7. **Hyperparameter Tuning**: Optimize model parameters
8. **Deployment**: Put model into production

## Best Practices

- **Start Simple**: Begin with basic models (logistic regression, decision trees)
- **Cross-Validation**: Use k-fold cross-validation to avoid overfitting
- **Feature Scaling**: Normalize/standardize features for better performance
- **Handle Imbalanced Data**: Use techniques like SMOTE, class weighting
- **Monitor Performance**: Track metrics over time in production
- **Version Control**: Track model versions, data versions, and code

## Resources

- **Courses**: Andrew Ng's ML Course (Coursera), Fast.ai
- **Books**: "Hands-On Machine Learning" by Aurélien Géron
- **Datasets**: Kaggle, UCI ML Repository, Google Dataset Search
- **Communities**: r/MachineLearning, Kaggle Forums, Stack Overflow
