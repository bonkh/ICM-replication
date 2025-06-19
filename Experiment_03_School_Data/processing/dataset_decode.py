# Data is originated from http://multilevel.ioe.ac.uk/intro/datasets.html
# However, the dataset is no longer available there.
# Current dataset can be downloaded from https://www.bristol.ac.uk/cmm/media/migrated/ilea567.zip

import pandas as pd
from itertools import accumulate


def split_by_pattern(s, pattern):
    ends = list(accumulate(pattern))
    starts = [0] + ends[:-1]
    return [s[start:end].strip() for start, end in zip(starts, ends)]

def process_file_to_df(filename, pattern):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            segments = split_by_pattern(line, pattern)
            data.append(segments)
    
    # Create column names like col1, col2, ...
    col_names = ['Year', 'School', 'ExamScore', '%FSM', '%VR1band', 'Gender', 'VRbandOfStudent', 'EthicGroup', 'SchoolGender', 'SchoolDenomination']
    return pd.DataFrame(data, columns=col_names)

# Example usage
pattern = [1,3, 2, 2, 2, 1,1,2,1,1]
df = process_file_to_df('ILEA567.DAT', pattern)
df.to_csv('proc_ILEA567.csv', index=False)
