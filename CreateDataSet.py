import pickle
import os
import pandas as pd

sample_set= pickle.load(open("sample75_ids.pkl",'rb'))
sample_set_id = sample_set['accession']
# sample_set_label = sample_set['label']
# sample_set_id = ['ERR188021','ERR188022','ERR188023']
sample_data = pd.DataFrame()
root = r"C:\Users\ajinkuriakose\Desktop\Computational Biology\train"
for sampleid in sample_set_id:
    try:
        quantified_data = pd.read_csv(root+"\\"+sampleid+"\\bias\\quant.sf",sep='\t',error_bad_lines=False)
        quantified_data = pd.DataFrame(quantified_data.set_index('Name').stack()).transpose()
        sample_data = sample_data.append(quantified_data)
    except KeyboardInterrupt:
        break
    pickle.dump(sample_data, open("data_75.pkl", 'wb'))



