
import pandas as pd
import matplotlib.pyplot as plt


#This loads the raw student dataset created in Part 1
df = pd.read_csv("students_raw.csv")


#This creates a separate dataset with rows containing missing values removed
df_dropped = df.dropna()

#This compares the number of rows before and after removing missing values
print("\nRows before dropping:", len(df))
print("Rows after dropping:", len(df_dropped))


#This creates a copy of the original dataset so that the raw data remains unchanged
df_clean = df.copy()


#This corrects negative study hours and prevents values above 30
df_clean["Study_Hours"] = df_clean["Study_Hours"].clip(
    lower=0,
    upper=30
)

#This ensures quiz participation stays between 0 and 100
df_clean["Quiz_Participation"] = df_clean["Quiz_Participation"].clip(
    lower=0,
    upper=100
)

#This ensures past performance stays between 0 and 100
df_clean["Past_Performance"] = df_clean["Past_Performance"].clip(
    lower=0,
    upper=100
)


#This replaces missing study hours with the median study hours
df_clean["Study_Hours"] = df_clean["Study_Hours"].fillna(
    df_clean["Study_Hours"].median()
)

#This replaces missing quiz participation with the median quiz participation
df_clean["Quiz_Participation"] = df_clean["Quiz_Participation"].fillna(
    df_clean["Quiz_Participation"].median()
)

#This replaces missing past performance with the median past performance
df_clean["Past_Performance"] = df_clean["Past_Performance"].fillna(
    df_clean["Past_Performance"].median()
)


#This checks whether missing values remain after cleaning
print("\nMissing values after cleaning:")
print(df_clean.isnull().sum())


#This checks whether invalid study hours remain
print("\nNegative study hours after cleaning:")
print((df_clean["Study_Hours"] < 0).sum())

#This checks whether quiz participation exceeds 100
print("\nQuiz participation above 100 after cleaning:")
print((df_clean["Quiz_Participation"] > 100).sum())


#This normalises study hours to a range between 0 and 1 using the min-max normalisation formula
min_hours = df_clean["Study_Hours"].min()
max_hours = df_clean["Study_Hours"].max()

#This prevents division by zero during normalisation
assert max_hours > min_hours, "Study hours cannot be normalised because all values are equal"

df_clean["Study_Hours_Normalised"] = (
    (df_clean["Study_Hours"] - min_hours) /
    (max_hours - min_hours)
)


#This checks the minimum and maximum normalised values
print("\nNormalised study hours:")
print(df_clean["Study_Hours_Normalised"].describe())


#This converts quiz participation percentages to a 0-1 scale
df_clean["Quiz_Normalised"] = (
    df_clean["Quiz_Participation"] / 100
)


#This calculates engagement using 60% study hours and 40% quiz participation
df_clean["Engagement_Score"] = (
    0.6 * df_clean["Study_Hours_Normalised"] +
    0.4 * df_clean["Quiz_Normalised"]
)


#This displays the engagement scores for the first 5 students
print("\nStudent engagement scores:")
print(df_clean[[
    "Student_ID",
    "Study_Hours_Normalised",
    "Quiz_Normalised",
    "Engagement_Score"
]].head())


#This validates that the cleaned dataset has no missing values
assert df_clean.isnull().sum().sum() == 0

#This validates that engagement scores are between 0 and 1
assert df_clean["Engagement_Score"].between(0, 1).all()

#This validates the normalised study hours
assert df_clean["Study_Hours_Normalised"].between(0, 1).all()

#This validates that study hours are between 0 and 30
assert df_clean["Study_Hours"].between(0, 30).all()

#This validates that quiz participation is between 0 and 100
assert df_clean["Quiz_Participation"].between(0, 100).all()

#This validates that past performance is between 0 and 100
assert df_clean["Past_Performance"].between(0, 100).all()


#Here we calculate the mean, median and standard deviation of study hours and quiz participation
statistics = df_clean[
    ["Study_Hours", "Quiz_Participation"]
].agg(["mean", "median", "std"])


#This displays the calculated statistics
print("\nDescriptive statistics:")
print(statistics)


#We use the pandas groupby function to compare the average engagement scores
#of students based on whether they completed their course
completion_groups = df_clean.groupby(
    "Course_Completed"
)["Engagement_Score"].mean()


#This displays the average engagement score for each group
print("\nAverage engagement by course completion:")
print(completion_groups)


#This counts students based on whether they completed their course
completion_counts = df_clean["Course_Completed"].value_counts()

print("\nNumber of students by course completion:")
print(completion_counts)


#This filters out students who have an engagement score greater than 0.7
high_engagement = df_clean[
    df_clean["Engagement_Score"] > 0.7
]


#This displays the number of students with engagement score greater than 0.7
print("\nNumber of students with engagement above 0.7:")
print(len(high_engagement))


#This displays the first 5 highly engaged students
print("\nStudents with high engagement:")
print(high_engagement[[
    "Student_ID",
    "Study_Hours",
    "Quiz_Participation",
    "Engagement_Score"
]].head())


#This function classifies students based on their past performance scores
df_clean["Performance_Category"] = df_clean["Past_Performance"].apply(
    lambda score: "High" if score >= 70
    else "Medium" if score >= 40
    else "Low"
)


#This displays the number of students in each performance category
print("\nStudent performance categories:")
print(df_clean["Performance_Category"].value_counts())


#This displays the first 5 students and their performance categories
print("\nStudent performance classification:")
print(df_clean[[
    "Student_ID",
    "Past_Performance",
    "Performance_Category"
]].head())


#This saves the new cleaned data set
df_clean.to_csv("students_clean.csv", index=False)

print("\nCleaned dataset saved successfully!")
print("Cleaned dataset shape:", df_clean.shape)





#Scatter plot below

#This creates a scatter plot comparing study hours and past performance
plt.figure(figsize=(10, 6))


#This plots students who completed the course
completed = df_clean[df_clean["Course_Completed"] == True]

plt.scatter(
    completed["Study_Hours"],
    completed["Past_Performance"],
    color="green",
    alpha=0.6,
    label="Completed"
)


#This plots students who did not complete the course
not_completed = df_clean[df_clean["Course_Completed"] == False]

plt.scatter(
    not_completed["Study_Hours"],
    not_completed["Past_Performance"],
    color="red",
    alpha=0.6,
    label="Not Completed"
)


#This adds a title, axis labels and legend
plt.title("Relationship Between Study Hours and Past Performance")
plt.xlabel("Study Hours Per Week")
plt.ylabel("Past Performance (%)")
plt.legend()
plt.grid(alpha=0.3)


#This saves the scatter plot as an image
plt.savefig("study_hours_scatter.png", dpi=300, bbox_inches="tight")



#Histogram below

#This creates a histogram showing the distribution of quiz participation
plt.figure(figsize=(10, 6))


#This groups quiz participation percentages into 10 intervals
plt.hist(
    df_clean["Quiz_Participation"],
    bins=10,
    color="skyblue",
    edgecolor="black",
    alpha=0.7,
    label="Quiz Participation"
)


#This adds a title, axis labels and legend
plt.title("Distribution of Student Quiz Participation")
plt.xlabel("Quiz Participation (%)")
plt.ylabel("Number of Students")
plt.legend()
plt.grid(axis="y", alpha=0.3)


#This saves the histogram as an image
plt.savefig("quiz_participation_histogram.png", dpi=300, bbox_inches="tight")



#Bar chart

#This calculates the average engagement score for students based on course completion
average_engagement = df_clean.groupby(
    "Course_Completed"
)["Engagement_Score"].mean()


#This creates a bar chart to compare the average engagement scores
plt.figure(figsize=(10, 6))

plt.bar(
    ["Not Completed", "Completed"],
    [
        average_engagement.loc[False],
        average_engagement.loc[True]
    ],
    color=["red", "green"],
    edgecolor="black",
    width=0.5,
    label="Average Engagement"
)


#This adds a title and labels to the bar chart
plt.title("Average Student Engagement by Course Completion")
plt.xlabel("Course Completion Status")
plt.ylabel("Average Engagement Score")



#This sets the engagement scale between 0 and 1
plt.ylim(0, 1)


#This adds grid lines to make comparison easier
plt.grid(axis="y", alpha=0.3)


#This saves the bar chart as an image
plt.savefig("engagement_bar_chart.png", dpi=300, bbox_inches="tight")


#This displays all three visualisations together
plt.show()


#This closes all figures after displaying them
plt.close("all")
