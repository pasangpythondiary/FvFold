import h5py
import re
import os
#@title Read in file in fasta format. { display-mode: "form" }
def read_fasta( fasta_path, split_char="!", id_field=0):
    '''
        Reads in fasta file containing multiple sequences.
        Split_char and id_field allow to control identifier extraction from header.
        E.g.: set split_char="|" and id_field=1 for SwissProt/UniProt Headers.
        Returns dictionary holding multiple sequences or only single 
        sequence, depending on input file.
    '''
    
    seqs = dict()
    with open( fasta_path, 'r' ) as fasta_f:
        
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip().split(split_char)[id_field]
                # replace tokens that are mis-interpreted when loading h5
                uniprot_id = uniprot_id.replace("/","_").replace(".","_")
                seqs[ uniprot_id ] = ''
            else:
                # repl. all whie-space chars and join seqs spanning multiple lines, drop gaps and cast to upper-case
                seq= ''.join( line.split() ).upper().replace("-","")
                # repl. all non-standard AAs and map them to unknown/X
                seq = seq.replace('U','X').replace('Z','X').replace('O','X')
                seqs[ uniprot_id ] += seq 
    example_id=next(iter(seqs))
    print("Read {} sequences.".format(len(seqs)))
    print("Example:\n{}\n{}".format(example_id,seqs[example_id]))

    return seqs

def read_all_seq(path, filename,fasta_dir):  
    print(path+filename)
    
    with h5py.File(path+filename, "r") as f:
        # Print all root level object names (aka keys) 
        # these can be group or dataset names 
        print("Keys: %s" % f.keys())
        # get first object name/key; may or may NOT be a group
        a_group_key = list(f.keys())[5]

        # get the object type for a_group_key: usually group or dataset
        #print(type(f[a_group_key])) 
        #print(f[a_group_key]) 
        

        # If a_group_key is a group name, 
        # this gets the object names in the group and returns as a list
        data = list(f[a_group_key])
        #print(data)
        

        # If a_group_key is a dataset name, 
        # this gets the dataset values and returns as a list
        data = list(f[a_group_key])
        #print(data)
        
        # preferred methods to get dataset values:
        ds_obj = f[a_group_key]      # returns as a h5py dataset object
        #print(ds_obj)
        ds_arr = f[a_group_key][()]  # returns as a numpy array
        #print(ds_arr)   
    
        train_data=ds_arr.tolist()
        

        
    
    t_data=[]
    for i in train_data:
        file=i.decode('utf-8')+".fasta"
        t_data.append(file)      
    
    all_seq = {}

    path = os.listdir(fasta_dir)


    for f in t_data:
        seqs = read_fasta(fasta_dir+ f)
        for k in seqs.keys():
            all_seq[k] = seqs[k]
            
    all_seq
            
    return all_seq