import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

num_students = 500

student_ids = [
    f"S{i:03d}" for i in range(1, num_students + 1)
]
s
student_hours = rng.uniform(0, 30, num_students)
quiz_participation = rng.integers(0, 101, num_students)
past_performance = rng.integers(0, 101, num_students)

course_completed = rng.choice(
    [True, False], 
    size=num_students
)

df = pd.DataFrame({
    "Student_ID: students_id",
    "Stud"
})