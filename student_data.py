import numpy as np
import pandas as pd

#Generates random values with seed num of 42 for consistency 
rng = np.random.default_rng(42)

#Number of students
num_students = 500

#Student Ids
student_ids = [
    f"S{i:03d}" for i in range(1, num_students + 1)
]

#Data for students generated via randomnisation 
study_hours = rng.uniform(0, 30, num_students)
quiz_participation = rng.integers(0, 101, num_students)
past_performance = rng.integers(0, 101, num_students)

course_completed = rng.choice(
    [True, False], 
    size=num_students
)

#DataFrame created containing student data

# Create the student DataFrame
df = pd.DataFrame({
    "Student_ID": student_ids,
    "Study_Hours": study_hours,
    "Quiz_Participation": quiz_participation,
    "Past_Performance": past_performance,
    "Course_Completed": course_completed
})

# Check that the column names are correct
print("Column names:", df.columns.tolist())



#This introduces missing values on the 20 selected rows
missing_hours = rng.choice(
    df.index,
    size=20,
    replace=False
)

#replaces the existing values within those 20 rows with nan
df.loc[missing_hours, "Study_Hours"] = np.nan


#This introduces missing values on 15 selected rows for quiz participation
missing_quiz = rng.choice(
    df.index,
    size=15,
    replace=False
)

#replaces the existing quiz participation values with nan
df.loc[missing_quiz, "Quiz_Participation"] = np.nan


#This introduces missing values on 10 selected rows for past performance
missing_performance = rng.choice(
    df.index,
    size=10,
    replace=False
)

#replaces the existing past performance values with nan
df.loc[missing_performance, "Past_Performance"] = np.nan


#This selects rows where study hours are not missing
valid_hours_index = df[
    df["Study_Hours"].notna()
].index

invalid_hours = rng.choice(
    valid_hours_index,
    size=5,
    replace=False
)

df.loc[invalid_hours, "Study_Hours"] = -5.0


#This selects rows where quiz participation is not missing
valid_quiz_index = df[
    df["Quiz_Participation"].notna()
].index

#This selects 5 rows to introduce quiz participation above 100% used to simulate data entry errors
invalid_quiz = rng.choice(
    valid_quiz_index,
    size=5,
    replace=False
)

#This adds the scores over 100% to the 5 rows
df.loc[invalid_quiz, "Quiz_Participation"] = 150





# Validate dataset structure
expected_columns = [
    "Student_ID",
    "Study_Hours",
    "Quiz_Participation",
    "Past_Performance",
    "Course_Completed"
]

assert df.shape == (500, 5), "Incorrect dataset shape"
assert df.columns.tolist() == expected_columns, "Incorrect column names"
assert df["Student_ID"].is_unique, "Student IDs are not unique"

print("Dataset structure validated successfully!")

print("\nMissing values:")
print(df.isnull().sum())

print("\nNegative study hours:")
print((df["Study_Hours"] < 0).sum())

print("\nQuiz participation above 100:")
print((df["Quiz_Participation"] > 100).sum())

