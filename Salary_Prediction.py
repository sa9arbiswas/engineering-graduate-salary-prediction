# %% [markdown]
# # Final Project
# Sagar Biswas <br>
# MSDS 422-DL <br>
# June 7, 2026

# %% [markdown]
# ## Executive Summary

# %% [markdown]
# <p>Employment outcomes are used as a key performance indicator for higher education institutions. The employability and satisfaction of graduates has a major impact on a university's reputation, rankings, admissions recruitment, and relationships with employers. The ability to predict employment outcomes, such as post-graduate salary, would be valuable to universities.</p>
# <p>This project aims to predict post-graduate salary and proactively identify students who may be at risk of less desirable outcomes. This uses historical data from engineering graduates from Indian universities, including academic records, university characteristics, and test scores. The objective is to evaluate whether salary outcomes can be predicted using information available prior to placement while also identifying attributes that most influence salary differences.</p>

# %% [markdown]
# ## Research Objectives

# %% [markdown]
# <p>Universities are always facing pressure to improve post-graduation outcomes for students. These outcomes are usually evaluated retrospectively, after students have already graduated. A predictive framework would help to understand which factors influence post-graduate outcomes the most and how they can be used to identify students while they are still enrolled to prepare them better for the best outcomes.</p>
# <p>The primary research objectives are to determine whether post-graduate salary can be reliably predicted and which student indicators have the strongest influence on salary variation.</p>
# <p>Although the dataset reflects the population of Indian university engineering students and graduates, the objective of understanding salary indicators and the analytical techniques used remain transferable to broader university contexts.

# %% [markdown]
# ## Dataset Source

# %% [markdown]
# The dataset used in this analysis uses the Kaggle dataset <strong>Engineering Graduate Salary Prediction</strong>, published by Kaggle contributor Manish KC. The underlying data originates from 'Aspiring Minds Research', which captures salary information for engineering graduates in India.
# 
# Kaggle dataset link: https://www.kaggle.com/datasets/manishkc06/engineering-graduate-salary-prediction/

# %% [markdown]
# ## Exploratory Data Analysis

# %%
# multiple outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

# %%
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor as RF, GradientBoostingRegressor as GBR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# %%
# show full data width
pd.set_option('display.max_columns', None)
# suppress scientific notation
pd.set_option('display.float_format', '{:,.2f}'.format)

# %%
# load dataset
df = pd.read_csv('Engineering_graduate_salary.csv')
print(f'Full dataset shape: {df.shape}')

# %%
# inspect data
df.head()
df.info()

# %%
# drop IDs
df = df.drop('ID', axis = 1)
# separate response and predictors
response = 'Salary'
response_col = [col for col in df.columns if col == response]
predictors = [col for col in df.columns if col != response]
print(f'Shape: {df.shape[0]} observations, {df.shape[1]} columns')
print(f'Response variable: {response}')
print(f'Predictors: {len(predictors)} columns')

# %%
# check duplicates and missing values
duplicate_count = df.duplicated().sum()
missing_values = df.isna().sum().sum()
print(f'Missing values count: {missing_values}')
print(f'Duplicate row count: {duplicate_count}')

# %% [markdown]
# There are no duplicate observations or literal null values.

# %% [markdown]
# ### Response variable descriptive statistics

# %%
# summary stats
df[response].describe().round(2)

# %%
# distribution plot
fig, axes = plt.subplots(1, 2, figsize = (10, 4))
sns.histplot(df[response], bins = 50, ax = axes[0])
axes[0].set_title('Salary distribution')
axes[0].ticklabel_format(axis = 'x', style = 'plain')
axes[0].set_xlabel('Salary (INR)')
sns.boxplot(df[response], orient = 'h', ax = axes[1])
axes[1].set_title('Boxplot of Salary')
axes[1].ticklabel_format(axis = 'x', style = 'plain')
axes[1].set_xlabel('Salary (INR)')
plt.show();

# %%
# check skew and kurtosis
salary_mean = df[response].mean()
salary_med = df[response].median()
salary_skew = df[response].skew()
salary_kurt = df[response].kurt()
print(f'Mean salary = ₹{salary_mean:,.2f}, Median salary = ₹{salary_med:,.2f}')
print(f'Skew = {salary_skew:.2f}, Kurtosis = {salary_kurt:.2f}')

# %%
# calculate outliers
salary_q1 = df[response].quantile(0.25)
salary_q3 = df[response].quantile(0.75)
salary_iqr = salary_q3 - salary_q1
lower_bound = salary_q1 - 1.5 * salary_iqr
upper_bound = salary_q3 + 1.5 * salary_iqr
print(f'Lower bound: {lower_bound:.2f}')
print(f'Upper bound: {upper_bound:.2f}')
salary_outliers = df[(df[response] < lower_bound) | (df[response] > upper_bound)]
print(f'Number of outliers in salary: {len(salary_outliers)}')
print(f'Percentage of dataset: {len(salary_outliers) / len(df) * 100:.2f}%')

# %%
# log transformation
df['l_Salary'] = np.log(df['Salary'])

# %%
# compare distributions
fig, axes = plt.subplots(2, 2, figsize = (10, 7))
sns.histplot(df[response], bins = 50, ax = axes[0, 0])
axes[0, 0].set_title('Salary distribution')
axes[0, 0].ticklabel_format(axis = 'x', style = 'plain')
axes[0, 0].locator_params(axis = 'x', nbins = 6)
axes[0, 0].set_xlabel('Salary (INR)')
sns.boxplot(df[response], orient = 'h', ax = axes[0, 1])
axes[0, 1].set_title('Boxplot of Salary')
axes[0, 1].ticklabel_format(axis = 'x', style = 'plain')
axes[0, 1].locator_params(axis = 'x', nbins = 6)
axes[0, 1].set_xlabel('Salary (INR)')
sns.histplot(df['l_Salary'], bins = 50, ax = axes[1, 0])
axes[1, 0].set_title('ln(Salary) distribution')
axes[1, 0].ticklabel_format(axis = 'x', style = 'plain')
sns.boxplot(df['l_Salary'], orient = 'h', ax = axes[1, 1])
axes[1, 1].set_title('Boxplot of ln(Salary)')
plt.tight_layout()
plt.show();

# %%
# check skew and kurtosis
l_salary_mean = df['l_Salary'].mean()
l_salary_med = df['l_Salary'].median()
l_salary_skew = df['l_Salary'].skew()
l_salary_kurt = df['l_Salary'].kurt()
print(f'Mean ln(Salary) = {l_salary_mean:.2f}, Median ln(Salary) = {l_salary_med:.2f}')
print(f'Skew = {l_salary_skew:.2f}, Kurtosis = {l_salary_kurt:.2f}')

# %% [markdown]
# The original response variable Salary is heavily right skewed. 2.6% of the dataset is outliers based on the 1.5 * IQR of Salary. After applying natural log transformation, the skew greatly reduced from 6.96 to -0.13, as did the kurtosis from 94.13 to 0.96. The histogram also demonstrates a more symmetrical and uniform shape, suggesting a more normal distribution shape.

# %% [markdown]
# ### Predictors descriptive statistics

# %%
# data types
df[predictors].dtypes

# %%
# summary stats for predictors
df[predictors].describe().round(2).T

# %%
# count number of colleges in sample
num_colleges = df['CollegeID'].nunique()
print(f'{num_colleges} unique colleges in sample')

# %%
# check numeric values < 0
neg_cols = [col for col in df.select_dtypes(exclude = 'object').columns if (df[col] == -1).sum() != 0]
neg_vals = (df[neg_cols] == -1).sum().sort_values(ascending = False)
neg_vals

# %% [markdown]
# While there weren't any literal nulls in the dataset, there are predictor columns with many -1 values. -1 is likely not a real academic score, so it is more likely that it is functionally a missing value.

# %%
# group features together
categorical_vars = ['Gender', '10board', '12board', 'CollegeTier', 'Degree', 'Specialization', 'CollegeCityTier', 'CollegeState']
date_vars = ['DOB', '12graduation', 'GraduationYear']
academic_vars = ['10percentage', '12percentage', 'collegeGPA']
score_vars = ['English', 'Logical', 'Quant', 'Domain', 'ComputerProgramming', 'ElectronicsAndSemicon',
              'ComputerScience', 'MechanicalEngg', 'ElectricalEngg', 'TelecomEngg', 'CivilEngg']
personality_vars = ['conscientiousness', 'agreeableness', 'extraversion', 'nueroticism', 'openess_to_experience']
id_vars = ['CollegeID', 'CollegeCityID']
# confirm no missing features
print(f'All predictors accounted for: {len(categorical_vars + date_vars + academic_vars + score_vars + personality_vars + id_vars) == len(predictors)}')

# %% [markdown]
# #### Categorical features

# %%
# unique values in categorical features
df[categorical_vars].nunique().sort_values(ascending = False)

# %% [markdown]
# There is high dimensionality for the school board variables, Specialization, and CollegeState.

# %%
# frequency of categories
for col in categorical_vars:
  print('\n', df[col].value_counts().head(10))

# %% [markdown]
# Gender is moderately imbalanced. CollegeTier is extremely imbalanced, with almost all observations being type 2. Degree seems to be almost entirely in Bachelors degree, with relatively few observations in the other categories (could be regrouped). Specialization is highly fragmented, but there is high concentration in the top majors.

# %% [markdown]
# #### Continuous features

# %%
# distributions of continuous variables
df[academic_vars + score_vars].hist(figsize = (30, 30), bins = 50)
plt.show();

# %%
# plot concentration of -1 counts
diff = [col for col in neg_cols if col not in score_vars]
if len(diff) > 0:
  print(f'{len(diff)} non-score columns with -1 values.')
else:
  print('All columns with -1 values are score columns.')
sns.barplot(data = neg_vals, orient = 'h')
plt.title('Frequency of -1 placeholders across features')
plt.ylabel('Feature')
plt.xlabel('Count of -1 values')
plt.show;

# %% [markdown]
# Several features (all of them assessment scores) contain high concentrations of -1 placeholder values, which will not be functional for numeric analysis.

# %%
# distributions of personality scores
df[personality_vars].hist(figsize = (20, 20), bins = 50)
plt.show();

# %% [markdown]
# The personality score distributions all appear to be centered around 0.

# %% [markdown]
# ### Correlations and collinearity

# %%
# top correlations with salary
df_corr = df.select_dtypes(include = 'number')
corrmat = df_corr.corr().round(2)
f, ax = plt.subplots(figsize = (20, 20))
ax = sns.heatmap(corrmat, vmax = .8, square = True, annot = True, cmap = 'RdYlBu', linewidths = .5 )
ax.tick_params(axis='y', rotation = 0)
plt.title('Heatmap for Salary correlations (ln(Salary) included)')
plt.show();

# %%
# view highest correlated predictors with Salary and l_Salary
top_corr = df_corr.corr()[response].drop(['Salary', 'l_Salary']).sort_values(ascending = False).head()
l_top_corr = df_corr.corr()['l_Salary'].drop(['Salary', 'l_Salary']).sort_values(ascending = False).head()
fig, axes = plt.subplots(1, 2, figsize = (7, 5))
sns.barplot(x = top_corr.values, y = top_corr.index, ax = axes[0])
axes[0].set_title('Correlations with Salary')
axes[0].set_ylabel(None)
axes[0].set_xlabel('Correlation')
sns.barplot(x = l_top_corr.values, y = l_top_corr.index, ax = axes[1])
axes[1].set_title('Correlations with ln(Salary)')
axes[1].set_ylabel(None)
axes[1].set_xlabel('Correlation')
plt.tight_layout()
plt.show();

# %%
# quant vs salary scatterplot
fig, axes = plt.subplots(2, 1, figsize = (10, 10))
sns.regplot(data = df, 
            x = 'Quant', 
            y = response, 
            line_kws = {'color':'red'}, 
            scatter_kws = {'alpha': 0.7},
            ax = axes[0])
axes[0].set_title('Quant score vs. Salary')
axes[0].ticklabel_format(axis = 'y', style = 'plain')
axes[0].set_ylabel('Salary (INR)')
axes[0].set_xlabel('Quant score')
sns.regplot(data = df, 
            x = 'Quant', 
            y = 'l_Salary',
            line_kws = {'color':'red'}, 
            scatter_kws = {'alpha': 0.7}, 
            ax = axes[1])
axes[1].set_title('Quant score vs. ln(Salary)')
axes[1].set_xlabel('Quant score')
plt.show();

# %% [markdown]
# There are no individual features with strong correlations with Salary. Quant has a moderate positive correlation with log-transformed Salary. There are some moderate to strong correlations among predictors, particularly the assessment scores.

# %%
# boxplots of salary with low dimensional categories
fig, axes = plt.subplots(3, 2, figsize = (20, 25))
sns.boxplot(data = df, x = 'Gender', y = response, hue = 'Gender', ax = axes[0, 0])
axes[0, 0].set_title('Salary boxplot by Gender')
axes[0, 0].ticklabel_format(axis = 'y', style = 'plain')
axes[0, 0].set_ylabel('Salary (INR)')
sns.boxplot(data = df, x = 'Gender', y = 'l_Salary', hue = 'Gender', ax = axes[0, 1])
axes[0, 1].set_title('ln(Salary) boxplot by Gender')
sns.boxplot(data = df, x = 'CollegeTier', y = response, hue = 'CollegeTier', ax = axes[1, 0])
axes[1, 0].set_title('Salary boxplot by CollegeTier')
axes[1, 0].ticklabel_format(axis = 'y', style = 'plain')
axes[1, 0].set_ylabel('Salary (INR)')
sns.boxplot(data = df, x = 'CollegeTier', y = 'l_Salary', hue = 'CollegeTier', ax = axes[1, 1])
axes[1, 1].set_title('ln(Salary) boxplot by CollegeTier')
sns.boxplot(data = df, x = 'Degree', y = response, hue = 'Degree', ax = axes[2, 0])
axes[2, 0].set_title('Salary boxplot by Degree')
axes[2, 0].ticklabel_format(axis = 'y', style = 'plain')
axes[2, 0].set_ylabel('Salary (INR)')
sns.boxplot(data = df, x = 'Degree', y = 'l_Salary', hue = 'Degree', ax = axes[2, 1])
axes[2, 1].set_title('ln(Salary) boxplot by Degree')
plt.show();

# %% [markdown]
# The Salary (original and log-transformed) shows similar medians across categories, except for CollegeTier. There is a visible distinction in the boxplots between CollegeTiers 1 and 2. This could be an influential predictor, but the CollegeTier feature is also heavily imbalanced.

# %% [markdown]
# ## Data Preparation & Feature Engineering

# %% [markdown]
# ### Drop features

# %%
# drop IDs and boards
df_clean = df.copy()
df_clean = df_clean.drop(columns = id_vars)
df_clean = df_clean.drop(columns = ['10board', '12board'])
df_clean.shape

# %% [markdown]
# The CollegeID, CollegeCityID, 10board, and 12board features added a lot of levels to the dataset and likely have low practical signal.

# %% [markdown]
# ### Continuous feature preparation

# %%
# replace -1 values with Nan
df_clean[neg_cols] = df_clean[neg_cols].replace(-1, np.nan)

# %% [markdown]
# As discussed earlier, the -1 values are not functional. First step in dealing with these reclassifying them as missing values. These can then be imputed in preprocessing.

# %%
# handle date variables
df_clean['DOB'] = pd.to_datetime(df_clean['DOB'])
df_clean['GradAge'] = df_clean['GraduationYear'] - df_clean['DOB'].dt.year
df_clean = df_clean.drop('DOB', axis = 1)
date_vars.remove('DOB')
date_vars.append('GradAge')

# %% [markdown]
# Substituted the DOB feature with an age at graduation feature.

# %% [markdown]
# ### Categorical feature preparation

# %%
# regroup degree
df_clean['Degree'] = df_clean['Degree'].replace({'M.Tech./M.E.':'Other', 'M.Sc. (Tech.)':'Other'})
# regroup specializations
spec_count = df_clean['Specialization'].value_counts()
top_specs = spec_count[spec_count >= 100].index
df_clean['Specialization'] = df_clean['Specialization'].apply(lambda x: x if x in top_specs else 'Other')

# %% [markdown]
# The Masters degree classes in Degree were regrouped as Other because the observations were almost entirely in the Bachelors degree class. Specialization originally had 42 classes, so, the classes with fewer than 100 observations were regrouped as Other to reduce dimensionality.

# %% [markdown]
# ### Additional Feature Engineering

# %%
# create academic consistency feature
df_clean['AcademicConsistency'] = (
    df_clean['10percentage'] + df_clean['12percentage'] + df_clean['collegeGPA']
    ) / 3

# %%
# create academic growth feature
df_clean['AcademicGrowth'] = df_clean['collegeGPA'] - df_clean['12percentage']

# %% [markdown]
# In addition to the log-transformed response variable and GradAge feature, I have added a feature which averages the grade in class 10, class 12, and college GPA (which also appears to be a percentage) to capture academic consistency. I have also added a feature which takes the difference between college GPA and grade in class 12 to find any improvement in academic performance.

# %% [markdown]
# ### Preprocessing pipeline

# %%
# assign X and y (log transformed)
X = df_clean.drop(columns = ['Salary', 'l_Salary'])
y = df_clean['l_Salary']

# %%
# split numeric vs categorical
numeric_features = X.select_dtypes(include = ['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include = ['object']).columns.tolist()

# %%
# median impute and robust scaling
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy = 'median')),
    ('scaler', RobustScaler())])
# one hot encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown = 'ignore'))])

# %%
# build pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)])

# %% [markdown]
# <p>The log-transformed Salary variable is a preferable response variable because it demonstrates a more symmetric and normal shape than the original Salary variable and will be less sensitive to outlier observations. l_Salary will be the primary resposne variable going forward, and modeling predictions will be transformed back to the original INR scale.</p>
# <p>To continue preprocessing, the numeric features are imputed with the median (addresses the unusable -1 values) and scaled to reduce sensitivity to outliers. The categorical features are mode-imputed and one-hot encoded to create dummy variables.

# %% [markdown]
# ## Modeling

# %% [markdown]
# ### Design & model selection

# %%
# 80-20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print(f'Training set size: {len(X_train)}. Test set size: {len(X_test)}')

# %% [markdown]
# The models will be fit on the training set with 2398 observations and predictions will be made on the hold-out set with 600 observations.

# %%
# function for evaluation metrics
def eval_model(y_test, y_pred):
  return {
      'MAE': mean_absolute_error(y_test, y_pred),
      'MSE': mean_squared_error(y_test, y_pred),
      'R2': r2_score(y_test, y_pred)
      }

# %% [markdown]
# Defined the function eval_model to provide goodness of fit metrics for each model. Models will be evaluated on mean absolute error, mean squared error, and R^2.

# %% [markdown]
# ### Linear regression

# %%
# build linear regression model
linear_model = Pipeline([('preprocessor', preprocessor), ('model', LinearRegression())])

# %%
# fit linear model on training
linear_model.fit(X_train, y_train)
# make test predictions
lm_pred = linear_model.predict(X_test)

# %% [markdown]
# ### Ridge regression

# %%
# build ridge model
ridge_model = Pipeline([('preprocessor', preprocessor), ('model', Ridge(alpha = 1.0))])

# %%
# fit ridge model on training
ridge_model.fit(X_train, y_train)
# make test predictions
ridge_pred = ridge_model.predict(X_test)

# %% [markdown]
# ### Random forest

# %%
# build random forest model
rf_model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RF(random_state = 42))])

# %%
# hyperparameter tuning
rf_params = {'model__n_estimators': [100, 200, 300],
             'model__max_depth': [5, 10, None]}
rf_search = RandomizedSearchCV(
    rf_model,
    param_distributions = rf_params,
    n_iter = 20,
    cv = 5,
    scoring = 'r2',
    random_state = 42,
    n_jobs = -1)

rf_search.fit(X_train, y_train)
rf_best = rf_search.best_estimator_
# print best parameters
print('Best params: ', rf_search.best_params_)

# %%
# make test predictions
rf_pred = rf_best.predict(X_test)

# %% [markdown]
# ### Gradient boosting

# %%
# build gb model
gb_model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', GBR(random_state = 42))])

# %%
# hyperparameter tuning
gb_params = {'model__n_estimators': [100, 200, 300],
             'model__learning_rate': [0.01, 0.05, 0.1],
             'model__max_depth': [2, 3, 4]}
gb_search = RandomizedSearchCV(
    gb_model,
    param_distributions = gb_params,
    n_iter = 20,
    cv = 5,
    scoring = 'r2',
    n_jobs = -1,
    random_state = 42
)

gb_search.fit(X_train, y_train)
gb_best = gb_search.best_estimator_
# print best parameters
print('Best params: ', gb_search.best_params_)

# %%
# make test predictions
gb_pred = gb_best.predict(X_test)

# %% [markdown]
# Performed some hyperparameter tuning on the Random Forest and Gradient Boosting models using RandomizedSearchCV. The best models from the tuning were the ones fit and used to make predictions.

# %% [markdown]
# ### Model performance

# %%
# goodness of fit metrics
log_results = pd.DataFrame({
    'Linear': eval_model(y_test, lm_pred),
    'Ridge': eval_model(y_test, ridge_pred),
    'RF': eval_model(y_test, rf_pred),
    'GBR': eval_model(y_test, gb_pred)
})
log_results.round(3)

# %% [markdown]
# Gradient Boosting is the best performing model at predicting the log-transformed salary. The Ridge model seems to add no gains to the linear model.The low R^2 scores across all models indicate that salary may be inherently difficult to predict given the available feature set. A larger data sample, more features such as internship experience, or different ML techniques may explain more variation in salary and yield better predictions.

# %%
# convert to original scale
y_test_exp = np.exp(y_test)
lm_pred_exp = np.exp(lm_pred)
ridge_pred_exp = np.exp(ridge_pred)
rf_pred_exp = np.exp(rf_pred)
gb_pred_exp = np.exp(gb_pred)
# goodness of fit metrics
results = pd.DataFrame({
    'Linear': eval_model(y_test_exp, lm_pred_exp),
    'Ridge': eval_model(y_test_exp, ridge_pred_exp),
    'RF': eval_model(y_test_exp, rf_pred_exp),
    'GBR': eval_model(y_test_exp, gb_pred_exp)
})
results

# %% [markdown]
# After converting back to the original salary units, Gradient Boosting still performs better. The mean absolute error of the Gradient Boosting model is 96,982.51 rupees.

# %%
# prediction vs actual plot
plt.figure(figsize = (8, 8))
sns.scatterplot(x = y_test_exp, y = gb_pred_exp)
plt.plot([y_test_exp.min(), y_test_exp.max()], [y_test_exp.min(), y_test_exp.max()], color = 'red')
plt.ticklabel_format(axis = 'both', style = 'plain')
plt.xlabel('Actual Salary (INR)')
plt.ylabel('Predicted Salary (INR)')
plt.title('Actual Salary vs. Predicted Salary (Gradient Boosting)')
plt.show();

# %%
# residual plot
sns.residplot(x = gb_pred,
              y = y_test,
              lowess = True,
              line_kws = {'color':'red'},
              scatter_kws = {'alpha': 0.7})
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residual Analysis (Gradient Boosting)')
plt.show();

# %%
# feature importance
feature_importance = (gb_best.named_steps['model'].feature_importances_)
feature_names = (gb_best.named_steps['preprocessor'].get_feature_names_out())
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values('Importance', ascending = False)
importance_df['Feature'] = importance_df['Feature'].str.replace('num__','')
importance_df.head(10)

# %%
# feature importance bar chart
plt.figure(figsize = (10, 10))
sns.barplot(data = importance_df.head(), x = 'Feature', y = 'Importance', hue = 'Feature')
plt.title('Top 5 Features (Gradient Boosting)')
plt.show();

# %% [markdown]
# ### Findings & Conclusions
# <p>Partially predicting salary is possible, but these models only explain a portion of the variation. Improving predictive power and accuracy would likely involve using a larger sample, obtaining more predictor features in the data like internship experience, and testing other learning techniques like neural networks.</p>
# <p>The most important features from the Gradient Boosting model were Quant, AcademicConsistency, English, and ComputerProgramming. EDA supports that variation in median salary does show up in Quant scores, so it could be interpreted that investing in quantitative training could improve overall salary outcomes. AcademicConsistency was one of the engineered features measuring the average grades from 10th grade through college. Its importance could mean that a lower college GPA compared to high school grades could be an early indicator that the student needs support. English scores are also an important feature, indicating that employers value this skill and it could be emphasized at the university level.

# %%



