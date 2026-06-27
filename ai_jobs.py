import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Data Collection
data = pd.read_csv("ai_jobs_market_2025_2026.csv")
#print(data)
print("1. Data Collection")
print(data.head())
print(data.tail())
print(data.shape)
print(data.columns.to_list())
print(data.describe())
data['years_of_experience'].describe()  #using describe for one column
print(data.dtypes)
print(data.info())

# 2. data preparation
print("2. Data Preparation & data cleaning")

#modifying job id
data['job_id_number'] = data['job_id'].str.extract('(\d+)').astype(int)

#remove columns
data = data.drop(columns=["job_id"])  #removing the old job_id column
data = data.drop(columns=["experience_level"])
data = data.drop(columns=["company_size"])

#split salary_tier column
data[['salary_category', 'salary_range']] = data['salary_tier'].str.extract(r'(.+?)\s*\((.+)\)')  
data.drop(columns=["salary_tier"])
#axis = 0 --> row, axis = 1 --> column
#inplace = True --> modifies the dataframe directly

data.drop('salary_tier', axis=1, inplace=True)
data = data.drop(columns=["benefits_score_10"])  #drop 'benefits_score_10' column


print(data.isnull().sum())  #no null values available
data = data.drop_duplicates()  #remove duplicates
print(data.shape)



# 3. data preprocessing
#detect outliers
plt.boxplot(data["annual_salary_usd"])
plt.title("Salary Outliers")
plt.savefig("Salary Outliers.png")
plt.show()                     #found outliers using boxplot

# # 1. calculate Q1 (25th percentile) and Q3 (75th percentile)
# q1 = data["annual_salary_usd"].quantile(0.25)
# q3 = data["annual_salary_usd"].quantile(0.75)

# # 2. calculate the Interquartile Range
# iqr = q3 - q1

# # 3. Define the upper boundary (no lower outliers are visible)
# upper_bound = q3 + (1.5 * iqr)

# # 4. Filter and display the outlier rows
# outliers = data[data["annual_salary_usd"] > upper_bound]
# print(outliers["annual_salary_usd"])                       # 8 outliers found




# # Select numeric columns only
# numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns

# for col in numeric_cols:
#     q1 = data[col].quantile(0.25)
#     q3 = data[col].quantile(0.75)

#     iqr = q3 - q1

#     lower_bound = q1 - (1.5 * iqr)
#     upper_bound = q3 + (1.5 * iqr)

#     outlier_count = ((data[col] < lower_bound) |
#                      (data[col] > upper_bound)).sum()

#     print(f"{col}: {outlier_count} outliers")

# #cap data to handle outliers
# capped_data = data.copy()

print("3. Data Preprocessing & data cleaning")
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns

# Exclude binary columns
numeric_cols = [col for col in numeric_cols if data[col].nunique() > 2]  #>2 is to remove categorical data as it doesn't have outliers

for col in numeric_cols:
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    # Count outliers before capping
    outlier_count = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()

    print(f"{col}: {outlier_count} outliers")

    # Cap outliers
    data[col] = data[col].clip(
        lower=lower_bound,
        upper=upper_bound
    )

print("Outliers detected and capped.")

#charts
#How are salaries distributed?
#Is the data skewed ---> skewed right
print("Chart 1. Salary Distribution (Histogram)")
plt.hist(data['annual_salary_usd'], bins=30)
plt.title('Salary Distribution')
plt.xlabel('Salary')
plt.ylabel('Count')
plt.savefig("Salary_Distribution.png")
plt.show()


print("Chart 2. Experience Level Distribution (Bar Chart)")
data['years_of_experience'].value_counts().sort_index().plot(kind='bar')
plt.title('Experience Level Distribution')
plt.xlabel("Years of Experience")
plt.ylabel("Count")
plt.savefig("Experience_Level_Distribution.png")
plt.show()

#Which job roles are most common?
print("Chart 3. Job Title Distribution (Bar Chart)")
plt.figure(figsize=(6,4))
data['job_title'].value_counts().head(10).plot(kind='bar')
plt.title('Top 10 Job Titles')
plt.xlabel("Job_title")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("Job_Title_Distribution.png")
plt.show()


#What percentage of jobs are remote?
print("Chart 4. Remote vs Non-Remote Jobs (Pie Chart)")
data['is_remote_friendly'].replace({1:'Remote_Friendly', 0:'Not_Remote_Friendly'}).value_counts().plot(kind='pie',autopct='%1.1f%%')
plt.title('Remote vs Non-Remote Jobs')
plt.savefig("Remote vs Non-remote jobs.png")
plt.show()

#Does salary increase with experience?
print("Chart 5. Salary vs Experience (Scatter Plot)")
plt.scatter(data['years_of_experience'],data['annual_salary_usd'])
plt.title("Salary vs Experience")
plt.xlabel('Years Experience')
plt.ylabel('Salary')
plt.savefig("Salary vs Experience.png")
plt.show()

#Are high-demand jobs paid more?
print("Chart 6. Demand score vs salary (scatter plot)")
plt.scatter(data['demand_score'],data['annual_salary_usd'])
plt.title("Salary vs Demand Score")
plt.xlabel('Demand Score')
plt.ylabel('Salary')
plt.savefig("Demand Score vs Salary.png")
plt.show()

#Which roles pay the most?
print("Chart 7. Average Salary by Job Title")
plt.figure(figsize=(8,6))
top_salary = data.groupby('job_title')['annual_salary_usd'].mean().sort_values(ascending=False)
top_salary.head(10).plot(kind='bar')
plt.title("Average Salary by job title")
plt.xlabel("Job Title")
plt.ylabel("Annual_Salary")
plt.tight_layout()
plt.savefig("Average Salary by Job Title.png")
plt.show()

#How much more do senior roles earn?
print("Chart 8. Senior vs Non-Senior Salary")
data.groupby(data['is_senior'].map({1:'Senior',0:'Not Senior'}))['annual_salary_usd'].mean().plot(kind='bar', color=["green","red"])
plt.title("Senior vs Non Senior Salary")
plt.xlabel("is_senior")
plt.ylabel("Annual_Salary")
plt.tight_layout()
plt.savefig("Senior vs non-senior salary.png")
plt.show()

#Do LLM jobs pay more?
print("Chart 9. LLM vs Non-LLM Role Salary")
data.groupby(data['is_llm_role'].map({1:'LLM_Role',0:'Non LLM_role'}))['annual_salary_usd'].mean().plot(kind='bar',color=["green","blue"])
plt.title("LLM vs Non LLM Role Salary")
plt.xlabel("is_LLM_Role")
plt.ylabel("Annual_salary")
plt.tight_layout()
plt.savefig("LLM vs Non-LLM Role Salary.png")
plt.show()

#Which variables are strongly related?
print("Chart 10. Correlation Heatmap")
corr = data.corr(numeric_only=True)
plt.figure(figsize=(10,6))
sns.heatmap(corr,annot=True,cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("Correlation Heatmap.png")
plt.show()



