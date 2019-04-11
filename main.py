import numpy as np
import pandas as pd
from cross_validation import CrossValidation


data = pd.read_csv(r'CSVs/exam_performance.csv', header=0, index_col=0)
dep_var = 'math_score'
ind_vars = ['prep_course', 'free_lunch', 'male',
            'eth_B', 'eth_C', 'eth_D', 'eth_E',
            'ped_shs', 'ped_hs', 'ped_sc', 'ped_ad', 'ped_md']
init_weights = [ 63.9,
                 4.97, -10.4, 5.48,
                 2.15,  1.22, 2.15, 9.36,
                -3.03, -4.50, 0.14, 1.27, 4.43]

exams = CrossValidation(data, 10, dep_var, ind_vars, init_weights)
exams.split_data()

for i in range(exams.chunk_count):
    exams.regress_and_test(exams.traintest_pairs[i], 0.001, 20000)

exams.results_to_csv(r'CSVs/CV_results.csv')
