import pandas as pd

def get_questions(df):
    #list of questions
    questions = list(df.columns.values)
    questions.remove("Name")
    return questions

def get_students(df):
    #list of individuals
    students = list(df["Name"])
    students.remove("Total Marks") 
    return students

def get_marks(df, qlist):
    total_marks = df.loc[df["Name"] == "Total Marks"].values.flatten().tolist()
    total_marks.remove('Total Marks')   #list of total marks for each question

    #dictionary relating test mark to question
    mark_dict = {}
    current_mark = 1
    for i in range(len(qlist)):
        mark_dict[qlist[i]] = [m for m in range(current_mark, current_mark + total_marks[i])]
        current_mark += total_marks[i]

    #list of marks
    marks = list(range(1, current_mark))
    return mark_dict, marks

def student_marks(df, mark_dict, student):
    #list of student's scores for each question
    s_scores = df.loc[df["Name"] == student].values.flatten().tolist()
    s_scores.remove(student)

    qlist = get_questions(df)

    #binary list indicating if individual got mark correct for all marks
    s_marks = []
    for index, q in enumerate(qlist):
        s_marks_q = [0 for i in range(len(mark_dict[q]))]
        for i in range(s_scores[index]):
            s_marks_q[i] = 1
        s_marks += s_marks_q
    return s_marks

def get_score_df(df):
    qlist = get_questions(df)
    student_list = get_students(df)
    mark_dict, mlist = get_marks(df, qlist)

    markdf = pd.DataFrame({"Marks": mlist}) #DataFrame first row
    for student in student_list:
        markdf_s = pd.DataFrame({student: student_marks(df, mark_dict, student)})
        markdf = pd.concat([markdf, markdf_s], axis=1) #Append columns to DataFrame
    return markdf

def get_skill_df(df, max_skills):
    qlist = get_questions(df)
    mark_dict = get_marks(df, qlist)[0]
    
    skill_df = [] #Create a nested list to be converted into a DataFrame
    cols = ["Mark", "Question", "Question Mark", "Skills"] + ["T" + str(i+1) for i in range(max_skills)]
    
    for q, q_marks in mark_dict.items():
        for index, m in enumerate(q_marks):
            skill_df.append([m, q, index+1, ""] + ["" for i in range(max_skills)])
    skill_df = pd.DataFrame(skill_df, columns = cols)
    return skill_df
