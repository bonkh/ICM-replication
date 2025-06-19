import pandas as pd 
data = pd.read_csv('proc_ILEA567.csv', index_col=0)
print(data['ExamScore'].mean())