import numpy as np
import os
import glob
import cPickle as pickle
import os

class Checkpointer:
    def __init__(self, model_nm, cell_nm, attention_type):
        self.model_nm = model_nm
        self.cell_nm = cell_nm
        self.attention_type = attention_type
        self.last_ckpt = None
        self.last_id = 0
        self.step_save_location = 'steps.p'
        self.data_save_location = 'data'
        if model_nm == 'bidirectional' or model_nm ==  'stackedBidirectional':
            self.mapper_save_location = 'mapper_bi.p'
        else:
            self.mapper_save_location = 'mapper.p'
        # initialize the steps if not initialized
        if self.step_save_location not in os.listdir(self.get_checkpoint_location()):
            pickle.dump(0,open(self.get_step_file(), 'wb'))


    def steps_per_checkpoint(self, num_steps):
        self.steps_per_ckpt = num_steps

    def get_checkpoint_steps(self):
        return self.steps_per_ckpt

    def get_checkpoint_location(self):
        return 'checkpoint/' + self.model_nm + '/' + self.cell_nm + '/' + self.attention_type

    def get_last_checkpoint(self):
        '''
            Assumes that the last checpoint has a higher checkpoint id.
            Checkpoint will be saved in this exact format
            model_<checkpint_id>.ckpt
            Eg - model_100.ckpt
        '''
        self.present_checkpoints = glob.glob(self.get_checkpoint_location() + '/*.ckpt')
        if(len(self.present_checkpoints) != 0):
            present_ids = [self.__get_id(ckpt) for ckpt in self.present_checkpoints]
            # sort the ID's and return the model for the last ID
            present_ids.sort()
            self.last_id = present_ids[-1]
            self.last_ckpt = self.get_checkpoint_location() + '/model_' \
                            + str(self.last_id) + '.ckpt'


        return self.last_ckpt


    def __get_id(self, ckpt_file):
        return int(ckpt_file.split('.')[0].split('_')[1])

    def delete_previous_checkpoints(self, num_previous=5):
        '''
            Deletes all previous checkpoints that are <num_previous> before
            the present checkpoint.
            This is done to prevent blowing out of memory due to too many
            checkpoints
        '''
        self.present_checkpoints = glob.glob(self.get_checkpoint_location() + '/*.ckpt')
        if(len(self.present_checkpoints) > num_previous):
            present_ids = [self.__get_id(ckpt) for ckpt in self.present_checkpoints]
            present_ids.sort()
            ids_2_delete = present_ids[0:len(present_ids) - num_previous]
            for ckpt_id in ids_2_delete:
                ckpt_file_nm = self.get_checkpoint_location() + '/model_' \
                                + str(ckpt_id) + '.ckpt'
                os.remove(ckpt_file_nm)


    def get_save_address(self):
        _ = self.get_last_checkpoint()
        next_id = self.last_id + 1
        return self.get_checkpoint_location()+'/model_'+ str(next_id)+'.ckpt'

    def is_checkpointed(self):
        return self.last_id > 0

    def get_data_file_location(self):
        return  'checkpoint/' + self.data_save_location

    def get_mapper_file_location(self):
        return  'checkpoint/' + self.data_save_location + '/' + self.mapper_save_location

    def get_step_file(self):
        return self.get_checkpoint_location() + '/' + self.step_save_location

    def is_data_checkpointed(self,type_op=None):
        if type_op is 'bidirectional':
            if 'X_trn_fwd.csv' in os.listdir(self.get_data_file_location()):
                return True
        elif type_op is None:
            if 'X_trn.csv' in os.listdir(self.get_data_file_location()):
                return True
        else:
            return False
