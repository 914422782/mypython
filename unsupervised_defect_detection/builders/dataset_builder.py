import os
import pickle
from torch.utils import data
from dataset.cityscapes import CityscapesDataSet, CityscapesTrainInform, CityscapesValDataSet
from dataset.camvid import CamVidDataSet, CamVidValDataSet, CamVidTrainInform
from dataset.voc import  VocDataSet, VocTrainInform, VocValDataSet
from dataset.ade20k import Ade20kTrainInform, Ade20kDataSet, Ade20kValDataSet
from dataset.sunrgb import SunRGBValDataSet, SunRGBTrainInform, SunRGBDataSet

def     build_dataset_train(dataset, input_size, batch_size, train_type, random_scale, random_mirror, num_workers):

    if dataset == 'cityscapes':
        data_dir = os.path.join('./dataset/', dataset)
        dataset_list = dataset + '_train_list.txt'
        train_data_list = os.path.join(data_dir, dataset + '_' + 'train' + '_list.txt')
        val_data_list = os.path.join(data_dir, dataset + '_val' + '_list.txt')
        inform_data_file = os.path.join('./dataset/inform/', dataset + '_inform.pkl')
    elif dataset == 'camvid':
        data_dir = os.path.join('./dataset/', dataset)
        dataset_list = dataset + '_train_list.txt'
        train_data_list = os.path.join(data_dir, dataset + '_' + 'train' + '_list.txt')
        val_data_list = os.path.join(data_dir, dataset + '_val' + '_list.txt')
        inform_data_file = os.path.join('./dataset/inform/', dataset + '_inform.pkl')
    elif dataset == 'voc':
        data_dir = './dataset/VOCdevkit/VOC2012'
        dataset_list = './dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
        train_data_list = './dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
        val_data_list = './dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'
        inform_data_file = os.path.join('./dataset/inform/', dataset + '_inform.pkl')
    elif dataset == 'ade20k':
        data_dir = './dataset/ADEChallengeData2016'
        dataset_list = './dataset/ADEChallengeData2016/images/training'
        train_data_list = './dataset/ADEChallengeData2016/images/training'
        val_data_list = './dataset/ADEChallengeData2016/images/validation'
        inform_data_file = os.path.join('./dataset/inform/', dataset + '_inform.pkl')
    elif dataset == 'sunrgb':
        data_dir = './dataset/SUNRGB-D'
        dataset_list = './dataset/SUNRGB-D/train'
        train_data_list = './dataset/SUNRGB-D/train'
        val_data_list = './dataset/SUNRGB-D/test'
        inform_data_file = os.path.join('./dataset/inform/', dataset + '_inform.pkl')
    else:
        raise NotImplementedError(
            "This repository not supports this dataset" % dataset)


    # inform_data_file collect the information of mean, std and weigth_class
    if not os.path.isfile(inform_data_file):
        print("%s is not found" % (inform_data_file))
        if dataset == "cityscapes":
            dataCollect = CityscapesTrainInform(data_dir, 19, train_set_file=dataset_list,
                                                inform_data_file=inform_data_file)
        elif dataset == 'camvid':
            dataCollect = CamVidTrainInform(data_dir, 11, train_set_file=dataset_list,
                                            inform_data_file=inform_data_file)
        elif dataset == 'voc':
            dataCollect = VocTrainInform(data_dir, 21, train_set_file=dataset_list,
                                            inform_data_file=inform_data_file)
        elif dataset == 'ade20k':
            dataCollect = Ade20kTrainInform(data_dir, 150, train_set_file=dataset_list,
                                         inform_data_file=inform_data_file)
        elif dataset == 'sunrgb':
            dataCollect = SunRGBTrainInform(data_dir, 37, train_set_file=dataset_list,
                                            inform_data_file=inform_data_file)
        else:
            raise NotImplementedError(
                "This repository now supports two datasets: cityscapes and camvid, %s is not included" % dataset)

        datas = dataCollect.collectDataAndSave()
        if datas is None:
            print("error while pickling data. Please check.")
            exit(-1)
    else:
        print("find file: ", str(inform_data_file))
        datas = pickle.load(open(inform_data_file, "rb"))


    if dataset == "cityscapes":

        trainLoader = data.DataLoader(
            CityscapesDataSet(data_dir, train_data_list, crop_size=input_size, scale=random_scale,
                              mirror=random_mirror, mean=datas['mean']),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True)

        valLoader = data.DataLoader(
            CityscapesValDataSet(data_dir, val_data_list, crop_size=input_size, f_scale=1, mean=datas['mean']),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
            drop_last=True)

        return datas, trainLoader, valLoader

    elif dataset == 'ade20k':
        trainLoader = data.DataLoader(
            Ade20kDataSet(data_dir, train_data_list, crop_size=input_size, scale=random_scale,
                              mirror=random_mirror, mean=datas['mean']),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True)

        valLoader = data.DataLoader(
            Ade20kValDataSet(data_dir, val_data_list, crop_size=input_size, f_scale=1, mean=datas['mean']),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
            drop_last=True)

        return datas, trainLoader, valLoader
    elif dataset == 'sunrgb':
        trainLoader = data.DataLoader(
            SunRGBDataSet(data_dir, train_data_list, crop_size=input_size, scale=random_scale,
                          mirror=random_mirror, mean=datas['mean']),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True)

        valLoader = data.DataLoader(
            SunRGBValDataSet(data_dir, val_data_list, crop_size=input_size, f_scale=1, mean=datas['mean']),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
            drop_last=True)

        return datas, trainLoader, valLoader

    elif dataset == "camvid":
        trainLoader = data.DataLoader(
            CamVidDataSet(data_dir, train_data_list, crop_size=input_size, scale=random_scale,
                          mirror=random_mirror, mean=datas['mean']),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True)

        valLoader = data.DataLoader(
            CamVidValDataSet(data_dir, val_data_list, crop_size=input_size, f_scale=1, mean=datas['mean']),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        return datas, trainLoader, valLoader

    elif dataset == 'voc':
        trainLoader = data.DataLoader(
            VocDataSet(data_dir, train_data_list, crop_size=input_size, scale=random_scale,
                          mirror=random_mirror, mean=datas['mean']),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True)

        valLoader = data.DataLoader(
            VocValDataSet(data_dir, val_data_list, crop_size=input_size, f_scale=1, mean=datas['mean']),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        return datas, trainLoader, valLoader


def build_dataset_test(dataset, input_size, batch_size, num_workers, none_gt=False):

    if dataset == 'camvid':
        data_dir = os.path.join('./dataset/', dataset)
        dataset_list = dataset + '_train_list.txt'
        test_data_list = os.path.join(data_dir, dataset + '_test' + '_list.txt')
        inform_data_file = os.path.join('./dataset/inform/', dataset + '_inform.pkl')
    else:
        raise NotImplementedError(
            "This repository not supports this dataset" % dataset)

    # inform_data_file collect the information of mean, std and weigth_class
    if not os.path.isfile(inform_data_file):
        print("%s is not found" % (inform_data_file))

        if dataset == 'camvid':
            dataCollect = CamVidTrainInform(data_dir, 11, train_set_file=dataset_list,
                                            inform_data_file=inform_data_file)
        else:
            raise NotImplementedError(
                "This repository now supports two datasets: cityscapes and camvid, %s is not included" % dataset)

        datas = dataCollect.collectDataAndSave()
        if datas is None:
            print("error while pickling data. Please check.")
            exit(-1)
    else:
        print("find file: ", str(inform_data_file))
        datas = pickle.load(open(inform_data_file, "rb"))

    if dataset == "camvid":
        testLoader = data.DataLoader(
            CamVidValDataSet(data_dir, test_data_list, crop_size=input_size,  mean=datas['mean']),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        return datas, testLoader