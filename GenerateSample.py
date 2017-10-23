import _pickle as cPickle
import pickle
import pandas as pd

SAMPLES = pd.read_csv(r"C:\Users\ajinkuriakose\Desktop\Computational Biology\Project\p1_train.csv", sep=',', error_bad_lines=False)
labels = ['CEU','FIN','GBR','TSI','YRI']

SAMP = pd.DataFrame()

# SAMPLES_FILETERED = SAMPLES.query("label in @labels").sample(n=10)['accession']
# print(SAMPLES_FILETERED)
for lab in labels:
    SAMPLES_FILETERED = SAMPLES[SAMPLES.label == lab].sample(n=15)
    SAMP =SAMP.append(SAMPLES_FILETERED,ignore_index=False)
# print (SAMP['accession'])
# SAMP.to_csv('test.csv',index=False)
pickle.dump(SAMP,open("sample75_ids.pkl",'wb'))