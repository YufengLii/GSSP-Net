import os
import sys
import numpy as np
import torch.utils.data as Data


class DatasetGSSU(Data.Dataset):
    def __init__(self, cfg):

        self.is_train=cfg.is_train
        self.Train_Data_Root = cfg.Train_Data_Root
        self.Test_Data_Root = cfg.Test_Data_Root
        self.file_list = []
        if self.is_train:
            data_filenames = os.listdir(self.Train_Data_Root)
            for i in range(len(data_filenames)):
                self.file_list.append(os.path.join( self.Train_Data_Root, data_filenames[i] ))
            
        else:
            data_filenames = os.listdir(self.Test_Data_Root)
            for i in range(len(data_filenames)):
                self.file_list.append(os.path.join( self.Test_Data_Root, data_filenames[i] ))
        
        
    def __getitem__(self, index):

        data_i = np.load(self.file_list[index])
        
        image = data_i['image']
        edge = data_i['edge']
        edge = np.expand_dims(edge, axis=0)
        jmap = data_i['jmap']
        joff = data_i['joff']
        filepath = str(data_i["filepath"])

        return image, edge, jmap, joff, filepath

    def __len__(self):
        
        return len(self.file_list)


# if __name__ == '__main__':

#     sys.path.append(r'./config/')
#     import cfg
#     cfg_data = cfg.parse()
#     dataset = DatasetGSSU(cfg_data)

#     for i in range(dataset.__len__()):
        
#         image, edge, jmap, joff, filepath = dataset.__getitem__(i)
