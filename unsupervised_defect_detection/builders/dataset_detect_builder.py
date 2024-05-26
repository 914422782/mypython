import os
import pickle
from torch.utils import data
from dataset.syringe import SyringeDataSet, SyringeTestDataSet

def build_dataset_train(dataset, input_size, batch_size, train_type, random_scale, random_mirror, num_workers, var=0.01):
    if dataset == 'syringe':
        data_dir = './dataset/syringe'
        train_data_list = './dataset/syringe/train'
        val_data_list = './dataset/syringe/valid'
    else:
        raise NotImplementedError(
            "This repository not supports this dataset" % dataset)


    if dataset == 'syringe':
        trainLoader = data.DataLoader(
            SyringeDataSet(root=train_data_list, var=var, crop_size=input_size),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True)

        valLoader = data.DataLoader(
            SyringeDataSet(root=val_data_list, var=var, crop_size=input_size),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True)

        return trainLoader, valLoader



def build_dataset_test(dataset, input_size, batch_size, num_workers, none_gt=False, var=0.01):

    if dataset == 'syringe':
        test_data_list = './dataset/syringe/test/defect'
        test_data_label_list = './dataset/syringe/test/defect_ground_truth'
    else:
        raise NotImplementedError(
            "This repository not supports this dataset" % dataset )

    if dataset == 'syringe':
        testLoader = data.DataLoader(
            SyringeTestDataSet(root=test_data_list, label_root=test_data_label_list, var=var, crop_size=input_size),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        return testLoader