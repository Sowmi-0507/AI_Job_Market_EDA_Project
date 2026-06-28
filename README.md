# AI Jobs Market 2025–2026: Exploratory Data Analysis (EDA)

## Project Overview

This project performs Exploratory Data Analysis (EDA) on an AI Job Market dataset containing information about salaries, job roles, experience levels, demand scores, remote work opportunities, and other job-related attributes.

The objective is to uncover trends in the AI job market, identify factors affecting salaries, and visualize relationships between different variables.

---

## Dataset

Dataset: `ai_jobs_market_2025_2026.csv`

The dataset contains information such as:

* Job ID
* Job Title
* Annual Salary (USD)
* Years of Experience
* Demand Score
* Remote Friendly Status
* Seniority Status
* LLM Role Indicator
* Salary Tier
* Company Information

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn

---

## Project Workflow

### 1. Data Collection

* Load dataset using Pandas
* Explore dataset structure
* Check dimensions and data types
* Generate descriptive statistics

### 2. Data Cleaning

* Remove unnecessary columns
* Extract numeric values from Job IDs
* Split salary tier information
* Remove duplicates
* Check for missing values

### 3. Data Preprocessing

* Detect outliers using IQR method
* Count outliers in numeric columns
* Display salary outliers
* Cap outliers using upper and lower bounds

### 4. Exploratory Data Analysis

The following visualizations were created:

#### Chart 1: Salary Distribution

* Understand salary distribution
* Identify skewness

#### Chart 2: Experience Level Distribution

* Analyze employee experience levels

#### Chart 3: Top Job Titles

* Identify the most common AI roles

#### Chart 4: Remote vs Non-Remote Jobs

* Understand remote work adoption

#### Chart 5: Salary vs Experience

* Analyze relationship between salary and experience

#### Chart 6: Demand Score vs Salary

* Study impact of market demand on salaries

#### Chart 7: Average Salary by Job Title

* Identify highest-paying roles

#### Chart 8: Senior vs Non-Senior Salary

* Compare salary based on seniority

#### Chart 9: LLM vs Non-LLM Role Salary

* Compare salaries of LLM-focused roles

#### Chart 10: Correlation Heatmap

* Identify relationships between numerical variables

---

## Key Insights

### Salary Distribution

* Salary distribution is right-skewed.
* A few high-paying positions create salary outliers.

### Experience Impact

* Salaries generally increase with years of experience.

### Senior Roles

* Senior professionals earn significantly higher salaries.

### LLM Roles

* LLM-related positions tend to offer competitive salaries.

### Demand Score

* Demand score shows a positive relationship with salary.

---

## Project Structure

AI-Jobs-Market-EDA/

├── eda_analysis.py

├── ai_jobs_market_2025_2026.csv

├── requirements.txt

├── README.md

└── images/

---

## Future Improvements

* Interactive dashboards using Power BI
* Advanced statistical analysis
* Machine Learning salary prediction model
* Deployment using Streamlit

---

## Author

Sowmiya Mohan



