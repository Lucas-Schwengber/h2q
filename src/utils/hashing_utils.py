import numpy as np
import csv

def save_sign_hashes(data, out_path):
    records_hash = np.sign(data)

    with open(out_path,"w", newline = '') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')

        for hash_vec in records_hash:
            hash_list = [int(hash_entry) for hash_entry in hash_vec]
            writer.writerow(hash_list)
                    
                        

                    

