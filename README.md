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

## Summary
Ideally, a credit risk classifier would correctly categorize high risk loans as high risk, and always correctly classify low risk loans as low risk - this has proven to be a tricky task, at least in this case. The baseline model only caught 10/104 high risk loans, but correctly categorized 17,097/17,101 low risk loans. The F1 score, a balanced metric between precision and recall, for the high risk class baseline was 0.17. The goal is increase the rate in which it detects high risk loans, and hopefully perform even better with the low risk ones, though that is setting a high standard. Doing so will inevitably drive up the F1 score as well, though the F1 score is debatably not as important as the recall values because catching high risk loans is key.

The Easy Ensemble AdaBoost model performed the best out of the six models:
  - The high risk recall was the highest of all models, including the baseline, at 0.92
  - The low risk recall was the second highest, behind the baseline, at 0.94
  - The high risk precision dropped drastically to 0.09, though it inappropriately flagged low risk loans as high risk at a greatly reduced rate compared to other models, exclusing the baseline

The results indicate that the AdaBoost method correctly categorized high risk loans nine times more often than the baseline, but incorrectly categorized low risk loans one in 17 times. More data to train with *could* help, but over 17,000 data points is rather robust as is. Ultimately, how low and high risk are defined may be what makes the difference - perhaps a medium risk option should be defined and included. Given what is available, the AdaBoost model used in this analysis is recommended for identifying high credit risk.
