
# Model Evaluation Results

This document summarizes the performance of various machine learning models tested on the Rotten Tomatoes Movies Rating Prediction dataset. The goal was to predict whether a movie’s `tomatometer_status` is `Certified-Fresh`, `Fresh`, or `Rotten`.

## Models Evaluated

The following models were trained and evaluated on the dataset:

1. **Random Forest**
2. **XGBoost**
3. **Support Vector Classifier (SVC)**
4. **K-Nearest Neighbors (KNN)**
5. **Multi-Layer Perceptron (MLP)**
6. **Decision Tree**

## Results Summary

| Model              | Accuracy | Certified-Fresh F1 | Fresh F1 | Rotten F1 | Macro Avg F1 | Weighted Avg F1 |
|--------------------|----------|--------------------|----------|-----------|--------------|-----------------|
| **Random Forest**  | 0.9123   | 0.95              | 0.88     | 0.91      | 0.91         | 0.91            |
| **XGBoost**        | 0.9091   | 0.95              | 0.87     | 0.91      | 0.91         | 0.91            |
| **SVC**            | 0.6731   | 0.73              | 0.58     | 0.69      | 0.67         | 0.67            |
| **KNN**            | 0.9107   | 0.94              | 0.88     | 0.91      | 0.91         | 0.91            |
| **MLP**            | 0.7136   | 0.75              | 0.58     | 0.76      | 0.70         | 0.70            |
| **Decision Tree**  | 0.9054   | 0.94              | 0.87     | 0.91      | 0.91         | 0.91            |

## Detailed Analysis

### 1. Random Forest
   - **Accuracy**: 91.2%
   - **Summary**: The Random Forest model achieved the highest accuracy and consistent F1-scores across all classes, indicating balanced performance.
   - **Strengths**: High recall and F1-score for `Certified-Fresh`, making it well-suited for this task.
   - **Weaknesses**: Minimal, as the model performs well across classes.

### 2. XGBoost
   - **Accuracy**: 90.9%
   - **Summary**: XGBoost closely follows Random Forest in performance, with high F1-scores for all classes. It’s slightly lower in recall for the `Fresh` class.
   - **Strengths**: Good generalization and consistent performance.
   - **Weaknesses**: Slightly lower recall for `Fresh`.

### 3. Support Vector Classifier (SVC)
   - **Accuracy**: 67.3%
   - **Summary**: The SVC model underperformed relative to other models, with lower scores across all metrics.
   - **Strengths**: Reasonable precision for `Fresh`.
   - **Weaknesses**: Low recall for `Fresh` (0.48), making it unsuitable without further tuning.

### 4. K-Nearest Neighbors (KNN)
   - **Accuracy**: 91.1%
   - **Summary**: KNN performed on par with Random Forest and achieved balanced F1-scores across classes.
   - **Strengths**: High recall for `Certified-Fresh` and `Rotten`.
   - **Weaknesses**: Slightly lower in precision for `Certified-Fresh` compared to Random Forest.

### 5. Multi-Layer Perceptron (MLP)
   - **Accuracy**: 71.4%
   - **Summary**: The MLP struggled, especially with the `Fresh` class, showing imbalanced recall and precision.
   - **Strengths**: High recall for `Rotten`.
   - **Weaknesses**: Low recall for `Fresh`, making it less suitable for this dataset.

### 6. Decision Tree
   - **Accuracy**: 90.5%
   - **Summary**: The Decision Tree model performed well, achieving an accuracy close to Random Forest and KNN. It has consistent F1-scores across all classes, though slightly lower than Random Forest.
   - **Strengths**: High precision and recall for `Certified-Fresh` and `Rotten`.
   - **Weaknesses**: Slightly lower accuracy compared to Random Forest and KNN, indicating it may not generalize as well.

## Conclusion

- **Top Models**: Random Forest, KNN, and Decision Tree achieved the best performance, making them strong candidates for deployment.
- **Further Optimization**: XGBoost may benefit from additional tuning to enhance recall for the `Fresh` class.
- **Underperforming Models**: SVC and MLP did not perform as well and may require different configurations or feature engineering to be competitive.

Based on these results, **Random Forest** remains the top recommended model due to its balanced performance across all metrics and ease of deployment.
