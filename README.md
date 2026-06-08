# Engineering Graduate Salary Prediction

## Author

Sagar Biswas<br>
MSDS 422-DL<br>
2026 Spring<br>
Final Project<br>

## Project Overview

<p>Employment outcomes are used as a key performance indicator for higher education institutions. The employability and satisfaction of graduates has a major impact on a university's reputation, rankings, admissions recruitment, and relationships with employers. The ability to predict employment outcomes, such as post-graduate salary, would be valuable to universities.</p>
<p>This project aims to predict post-graduate salary and proactively identify students who may be at risk of less desirable outcomes. This uses historical data from engineering graduates from Indian universities, including academic records, university characteristics, and test scores. The objective is to evaluate whether salary outcomes can be predicted using information available prior to placement while also identifying attributes that most influence salary differences.</p>

## Research Objective

<p>Universities are always facing pressure to improve post-graduation outcomes for students. These outcomes are usually evaluated retrospectively, after students have already graduated. A predictive framework would help to understand which factors influence post-graduate outcomes the most and how they can be used to identify students while they are still enrolled to prepare them better for the best outcomes.</p>
<p>The primary research objectives are to determine whether post-graduate salary can be reliably predicted and which student indicators have the strongest influence on salary variation.</p>
<p>Although the dataset reflects the population of Indian university engineering students and graduates, the objective of understanding salary indicators and the analytical techniques used remain transferable to broader university contexts.

## Dataset

The dataset used in this analysis uses the Kaggle dataset <strong>Engineering Graduate Salary Prediction</strong>, published by Kaggle contributor Manish KC. The underlying data originates from 'Aspiring Minds Research', which captures salary information for engineering graduates in India.

Kaggle dataset link: https://www.kaggle.com/datasets/manishkc06/engineering-graduate-salary-prediction/

Observations: 2,998 

Response Variable: Salary (log-transformed for modeling)

## Methodology

* Exploratory Data Analysis (EDA)
* Data Cleaning and Preprocessing
* Feature Engineering
* Log Transformation of Salary
* Train/Test Split (80/20)
* Machine Learning Model Evaluation

Models Evaluated:

* Linear Regression
* Ridge Regression
* Random Forest
* Gradient Boosting

## Results

|     | Linear | Ridge | Random Forest | Gradient Boosting |
| --- | ------ | ----- | ------------- | ----------------- |
| MAE | 0.35   | 0.35  | 0.35          | 0.34              |
| MSE | 0.21   | 0.21  | 0.21          | 0.20              |
| R^2 | 0.24   | 0,24  | 0.24          | 0.27              |

Gradient Boosting achieved the strongest predictive performance.

## Key Findings

* Quantitative score was the strongest predictor of salary.
* Academic consistency was an important engineered feature.
* English proficiency and programming skills contributed to salary prediction.

## Future Work

* Obtain larger sample with more countries and disciplines represented.
* Incorporate more predictors like internship experience to explain a larger portion of salary variation.
* Evaluate deep learning models such as neural networks.
* Expand analysis to additional student success outcomes.
