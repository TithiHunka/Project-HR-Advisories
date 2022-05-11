# Project-HR-Advisories
## 1) Project discription:
### My client for this project is the HR Department at a software company.

- They want to try a new initiative to retain employees.
- The idea is to use data to predict whether an employee is likely to leave.
- Once these employees are identified, HR can be more proactive in reaching out to them before it's too late.
- They only want to deal with the data that is related to permanent employees.

### Current Practice
Once an employee leaves, he or she is taken an interview with the name “exit interview” and shares reasons for leaving. The HR Department then tries and learns insights from the interview and makes changes accordingly.

### This suffers from the following problems:

- This approach is that it's too haphazard. The quality of insight gained from an interview depends heavily on the skill of the interviewer.
- The second problem is these insights can't be aggregated and interlaced across all employees who have left.
- The third is that it is too late by the time the proposed policy changes take effect.

The HR department has hired you as data science consultants. They want to supplement their exit interviews with a more proactive approach.

## 2) consulting Goals

### Your Role
You are given datasets of past employees and their status (still employed or already left).
Your task is to build a classification model using the datasets.
Because there was no machine learning model for this problem in the company, you don’t have quantifiable win condition. You need to build the best possible model.

### Problem Specifics
- Deliverable: Predict whether an employee will stay or leave.
- Machine learning task: Classification
- Target variable: Status (Employed/Left)
- Win condition: N/A (best possible model)

## 3) Data Discription
The Business Intelligence Analysts of the Company provided you three datasets that contain information about past employees and their status (still employed or already left).

### department_data
This dataset contains information about each department. The schema of the dataset is as follows:

- dept_id – Unique Department Code
- dept_name – Name of the Department
- dept_head – Name of the Head of the Department

### employee_details_data
This dataset consists of Employee ID, their Age, Gender and Marital Status. The schema of this dataset is as follows:

- employee_id – Unique ID Number for each employee
- age – Age of the employee
- gender – Gender of the employee
- marital_status – Marital Status of the employee

### employee_data
This dataset consists of each employee’s Administrative Information, Workload Information, Mutual Evaluation Information and Status.

#### |) Target variable

- status – Current employment status (Employed / Left)
#### ||) Administrative information

- department – Department to which the employees belong(ed) to
- salary – Salary level with respect to rest of their department
- tenure – Number of years at the company
- recently_promoted – Was the employee promoted in the last 3 years?
- employee_id – Unique ID Number for each employee
#### |||) Workload information

- n_projects – Number of projects employee has worked on
- avg_monthly_hrs – Average number of hours worked per month
- Mutual evaluation information
- satisfaction – Score for employee’s satisfaction with the company (higher is better)
- last_evaluation – Score for most recent evaluation of employee (higher is better)
- filed_complaint – Has the employee filed a formal complaint in the last 3 years?
## 4) Steps of Model Building
### 1) Installing and importing liberaries
### 2) establishing connection with SQL Server for data fetching
### 3) Data Discription and info for all the 3 datasets
### 4) Auto EDA with sweetviz and pandas profiling to understand the anamolies present in datasets
### 5) Data Cleaning 
### 6) EDA on various points to find out which feature is contributing more for employees to leave
### 7) Data Encoding:
Here we convert the Categorical Variables including the TARGET Variable into numbers by applying on among the below list: 
##### a)	Label Encoding
##### b)	One-Hot Encoding
##### c)	Ordinal Encoding
In this case we are taking only 3 categorical variables viz. 'department', 'status', 'salary' as the unseen test data has only these Categorical Columns and the Target Variable status as well
The target variable should be 1 for the Left and 0 for the Employed

### 8) Split your dataset into train and test data

-  The Train data set after split has 11172 Rows/Records and 9 Columns
- The Test data set has 2794 Rows/Records and 9 Columns.

### 9) Apply Normalizer Scaling to the Data Set (both Train and Test
- Normalizer is a scaling Process applied on rows
-	We did not find Considerable Difference in Results in the ML O/P after Scaling

### 10)  Analyze which Algorithm Classifier predicts the BEST RESULT and also check the Computation Time (Before SMOTE)
We applied the following 7 ML Classifiers on the Data Set:
- a) LogisticRegression
-  b) DecisionTreeClassifier
- c) RandomForestClassifier
- d) KNeighborsClassifier
-  e) GaussianNB(), 
- f) GradientBoostingClassifier
- g) XGBClassifier
###### We got the best result for Basic Model of Random Forest Classifier as compared to other Classifiers with the following results:
###### [Processing Time]: 2.325659990310669 seconds  ,[F1_SCORE]: 0.9589041095890412, [ACCURACY SCORE]: 0.98067287043665


### 11) The Data Set was highly IMBALANCED with TARGET variable status having the following counts
- Employed:    10631
- Left :        3335


