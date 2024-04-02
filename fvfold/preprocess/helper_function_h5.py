import h5py
import re
import os
import sys
sys.path.append('/home/pasang/all_experiment/FvFold')
import fvfold
project_path = os.path.abspath(os.path.join(fvfold.__file__, "../.."))
data_path = os.path.join(project_path, "data/")
def embedding():
    import h5py
    def read_embeddings(file_path):
        emb_dict = {}
        with h5py.File(file_path, 'r') as hf:
            group = hf['embeddings']
            for key in sorted(group.keys(), key=lambda x: int(x)):
                emb_dict[group[key].attrs['sequence_id']] = group[key][:]
        return emb_dict
    my_emb = read_embeddings(data_path+'per_residue_embeddings.h5')
    my_emb.keys()

    with h5py.File(data_path+"antibody.h5", "r") as f:
            # Print all root level object names (aka keys) 
            # these can be group or dataset names 
            # print("Keys: %s" % f.keys())
            # get first object name/key; may or may NOT be a group
            a_group_key = list(f.keys())[5]
            h = list(f.keys())[4]
            l= list(f.keys())[10]

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
            h_arr = f[h][()] 
            l_arr = f[l][()] 

            train_data=ds_arr.tolist()
            h_len=h_arr.tolist()
            l_len=l_arr.tolist()
            
    t_data=[]
    t_data_len={}
    for i,h,l in zip(train_data,h_len,l_len):
            file=i.decode('utf-8')
            t_data_len[file[:4]]=h+l
            t_data.append(file[:4])   
            
    t_data
    t_data_len

    import numpy as np
    name={}
    max_len={}
    def add_cancat(id,my_emb):
        h=id+":H"
        # h=":H"
        l=id+":L"
        # l=":L"
        name[id]=np.concatenate((my_emb[h],my_emb[l]), axis=0)
        # result = np.concatenate((array1, array2), axis=1)
        # key.startswith('your_character')
        
        
    for i in t_data:
        add_cancat(i,my_emb)
    name


    
    import torch
    # Define the input dictionary
    protein_embeddings = name

    # Create an empty dictionary to store the new embeddings
    new_embeddings = {}

    # Iterate over the keys and values in the input dictionary
    for protein_id, embedding in protein_embeddings.items():
        # Convert the embedding from numpy to PyTorch Tensor
        embedding = torch.from_numpy(embedding)
        
        # Add the new embedding to the output dictionary
        new_embeddings[protein_id] = embedding

    # Check the shapes of the new embeddings
    # for protein_id, embedding in new_embeddings.items():
    #     print(f"{protein_id} shape: {embedding.shape}")
        
        

    for key, value in new_embeddings.items():
        new_embeddings[key] = value.transpose(1,0)
        
    import torch
    from torch.utils.data import DataLoader

    class PadSequence:
        def __init__(self):
            pass

        def __call__(self, batch):
            key, data = zip(*batch)
            max_len = max([v.shape[1] for v in new_embeddings.values()])
            data = [torch.nn.functional.pad(t, (0, max_len - t.shape[1],0,0), mode='constant', value=0) for t in data]
            return key, torch.stack(data)

    class ProteinDataset:
        def __init__(self, new_embeddings):
            self.new_embeddings = new_embeddings

        def __len__(self):
            return len(self.new_embeddings)

        def __getitem__(self, idx):
            key = list(self.new_embeddings.keys())[idx]
            value = self.new_embeddings[key]
            return key, value

    # new_embeddings = {"1a6t": torch.randn(128, 224), "1ad9": torch.randn(128, 233), "1a3r": torch.randn(128, 232)}

    dataset = ProteinDataset(new_embeddings)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=PadSequence())
    
    
    return data_loader

