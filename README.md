# Credit_Risk_Analysis
#### Resources
  - pandas
  - scikit-learn
  - imbalanced-learn
  - Data: LoanStats_2019Q1.csv (became clean_loans.csv after cleaning/preprocessing)

## Overview
The purpose of this analysis was to analyze credit risk using various machine learning methods, and to effectively categorize high risk loans given the rest of the data. Because there was a heavy imbalance between loans classified as high risk and low risk, there was not much high risk data for the base model to train with. A baseline model was still utilized to serve as a gauge for the other methods.

The data had to be cleaned and preprocessed prior to the analysis - this process included using pd.get_dummies() to convert objects in columns into multiple columns with numeric values, since the methods used strictly require numeric data to predict with. An example of this would be the home_ownership column, which had values such as "RENT" or "MORTGAGE". After pd.get_dummies() was used, those values ("RENT") got their own column ("home_ownership_RENT") and either had a 0 for "no", or 1 for "yes". Another important step in the preprocessing was scaling the data so there wasn't as large of a range between the high and low numbers (such as loan amount versus interest rate). The "StandardScaler" method from sklearn.preprocessing was used for this step. By the end of the preprocessing, the data was encoded *and* scaled.

## Models & Results
#### Resampling Models
The following resampling models with a brief description were used to conduct the analysis with logistic regression:
 - Naive Oversampling
    - Scales the minority data to the size of the majority 
 - SMOTE (oversampling)
    - Minority sample size increased by interpolated data points
 - Cluster Centroid (undersampling)
    - Representative clusters undersampled down to minority sample size
 - SMOTEENN (hybrid / over & undersampling)
    - Oversamples minority sample and drops datapoints in close proximity belonging to different classes

#### Ensemble Models
The following models used ensemble methods:
  - Balanced Random Forest
    - Creates numerous simple decision trees, and bases analysis off of combined result
  - Easy Ensemble AdaBoost (Adaptive Boosting)
    - Each subsequent model weighs errors heavier

#### Confusion Matrices and Classification Report Results
Each method result was given a classification report and a confusion matrix to juxtapose the model's predictions to the actual data. The following image is the set of results. Take note that "pre" stands for *precision* and "rec" stands for the *recall*.

![TotalCM](https://user-images.githubusercontent.com/92493572/155912127-e6eb1e2a-ac32-4fb4-a21a-5bb303c772a0.png)
