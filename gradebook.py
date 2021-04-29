import os
import math
import scipy.stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from mysql.connector import connect, Error
from sqlalchemy import create_engine

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

db_user = os.environ.get("DB_USER")
db_pass = os.environ.get("DB_PASS")
db_name = "gradebook_db"

try:
    with connect(
            host="localhost",
            user=db_user,  # input("Enter username: "),
            password=db_pass,  # getpass("Enter password: "),
            database=db_name) as connection:
        print(connection)
except Error as e:
    print(e)

engine = create_engine("mysql+mysqlconnector://{user}:{pw}@localhost/{db}"
                       .format(
                           user=db_user,
                           pw=db_pass,
                           db=db_name))

roster = pd.read_sql_table("roster",
                           con=engine,
                           columns=["Section", "Email Address", "NetID"])
roster["NetID"] = roster["NetID"].str.lower()
roster["Email Address"] = roster["Email Address"].str.lower()
roster.set_index("NetID", inplace=True)

hw_exam_grades = pd.read_sql_table("hw_exam_grades",
                                   con=engine)
hw_exam_grades["SID"] = hw_exam_grades["SID"].str.lower()
index = ~hw_exam_grades.columns.str.contains("Submission")
hw_exam_grades = hw_exam_grades.loc[:, index]
hw_exam_grades.set_index("SID", inplace=True)

data_path = Path(__file__).parent / "Data"
quiz_grades = pd.DataFrame()

for file_path in data_path.glob("quiz_*_grades.csv"):
    quiz_name = " ".join(file_path.stem.title().split("_")[:2])
    table_name = file_path.stem

    quiz = pd.read_sql_table(
        table_name,
        con=engine,
        columns=["Email", "Grade"]
    ).rename(columns={"Grade": quiz_name})

    quiz["Email"] = quiz["Email"].str.lower()
    quiz.set_index("Email", inplace=True)

    quiz_grades = pd.concat([quiz_grades, quiz], axis=1)

final_data = pd.merge(
    roster,
    hw_exam_grades,
    left_index=True,
    right_index=True,
)
final_data = pd.merge(
    final_data, quiz_grades, left_on="Email Address", right_index=True
)
final_data = final_data.fillna(0)

n_exams = 3
for n in range(1, n_exams + 1):
    final_data[f"Exam {n} Score"] = (
        final_data[f"Exam {n}"] / final_data[f"Exam {n} - Max Points"]
    )

homework_scores = final_data.filter(regex=r"^Homework \d\d?$", axis=1)
homework_max_points = final_data.filter(regex=r"^Homework \d\d? -", axis=1)

sum_of_hw_scores = homework_scores.sum(axis=1)
sum_of_hw_max = homework_max_points.sum(axis=1)
final_data["Total Homework"] = sum_of_hw_scores / sum_of_hw_max

hw_max_renamed = homework_max_points.set_axis(homework_scores.columns, axis=1)
average_hw_scores = (homework_scores / hw_max_renamed).sum(axis=1)
final_data["Average Homework"] = average_hw_scores / homework_scores.shape[1]

final_data["Homework Score"] = final_data[
    ["Total Homework", "Average Homework"]
].max(axis=1)

quiz_scores = final_data.filter(regex=r"^Quiz \d$", axis=1)
quiz_max_points = pd.Series(
    {"Quiz 1": 11, "Quiz 2": 15, "Quiz 3": 17, "Quiz 4": 14, "Quiz 5": 12}
)

sum_of_quiz_scores = quiz_scores.sum(axis=1)
sum_of_quiz_max = quiz_max_points.sum()
final_data["Total Quizzes"] = sum_of_quiz_scores / sum_of_quiz_max

average_quiz_scores = (quiz_scores / quiz_max_points).sum(axis=1)
final_data["Average Quizzes"] = average_quiz_scores / quiz_scores.shape[1]

final_data["Quiz Score"] = final_data[
    ["Total Quizzes", "Average Quizzes"]
].max(axis=1)

weightings = pd.Series(
    {
        "Exam 1 Score": 0.05,
        "Exam 2 Score": 0.1,
        "Exam 3 Score": 0.15,
        "Quiz Score": 0.30,
        "Homework Score": 0.4,
    }
)

final_data["Final Score"] = (final_data[weightings.index] * weightings).sum(
    axis=1
)
final_data["Ceiling Score"] = np.ceil(final_data["Final Score"] * 100)

grades = {
    90: "A",
    80: "B",
    70: "C",
    60: "D",
    0: "F",
}


def grade_mapping(value):
    """Map numerical grade to letter grade."""
    for key, letter in grades.items():
        if value >= key:
            return letter


letter_grades = final_data["Ceiling Score"].map(grade_mapping)
final_data["Final Grade"] = pd.Categorical(
    letter_grades, categories=grades.values(), ordered=True
)


final_data["Floor Score"] = np.floor(final_data["Ceiling Score"] / 10) * 10

weightings = pd.Series(
    {
        "Exam 1 Score": 0.05,
        "Exam 2 Score": 0.1,
        "Exam 3 Score": 0.15,
    }
)

final_data["Exam Score"] = (final_data[weightings.index] * weightings).sum(
    axis=1
)

X = final_data["Exam Score"].to_numpy().reshape(-1, 1)
y = final_data["Ceiling Score"].to_numpy().reshape(-1, 1)


# Split the data into training/testing sets
X_train = X[:-75]
X_test = X[-75:]

# Split the targets into training/testing sets
y_train = y[:-75]
y_test = y[-75:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# Plot outputs
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


# features = np.column_stack((x, y))
# scaler = MinMaxScaler()
# features = scaler.fit_transform(features)

# kmeans = KMeans(
#     init="random",
#     n_clusters=3,
#     n_init=10,
#     max_iter=300
# )
# y_km = kmeans.fit_predict(features)

# # plot the 3 clusters
# plt.scatter(
#     features[y_km == 0, 0], features[y_km == 0, 1],
#     s=50, c='lightgreen',
#     marker='s', edgecolor='black',
#     label='cluster 1'
# )

# plt.scatter(
#     features[y_km == 1, 0], features[y_km == 1, 1],
#     s=50, c='orange',
#     marker='o', edgecolor='black',
#     label='cluster 2'
# )

# plt.scatter(
#     features[y_km == 2, 0], features[y_km == 2, 1],
#     s=50, c='lightblue',
#     marker='v', edgecolor='black',
#     label='cluster 3'
# )

# # plot the centroids
# plt.scatter(
#     kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
#     s=250, marker='*',
#     c='red', edgecolor='black',
#     label='centroids'
# )
# plt.legend(scatterpoints=1)
# plt.grid()
# plt.show()


for section, table in final_data.groupby("Section"):
    section_file = data_path / f"Section {section} Grades.csv"
    num_students = table.shape[0]
    print(
        f"In Section {section} there are {num_students} students saved to "
        f"file {section_file}."
    )
    table.sort_values(by=["Last Name", "First Name"]).to_csv(section_file)

grade_counts = final_data["Final Grade"].value_counts().sort_index()
grade_counts.plot.bar()
plt.show()

final_data["Final Score"].plot.hist(bins=20, label="Histogram")
final_data["Final Score"].plot.density(
    linewidth=4, label="Kernel Density Estimate"
)

final_mean = final_data["Final Score"].mean()
final_std = final_data["Final Score"].std()
x = np.linspace(final_mean - 5 * final_std, final_mean + 5 * final_std, 200)
normal_dist = scipy.stats.norm.pdf(x, loc=final_mean, scale=final_std)
plt.plot(x, normal_dist, label="Normal Distribution", linewidth=4)
plt.legend()
plt.show()

sns.set_style('darkgrid')
sns.displot(final_data["Final Score"], kind="hist", kde=True)
plt.show()
