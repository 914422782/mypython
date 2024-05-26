import os,sys
import time
import torch
import torchsummary
from torch import optim
import torch.nn as nn
import timeit
import math
import numpy as np
import matplotlib
import copy
from torchvision.utils import save_image
from torchsummary import summary
from sklearn import metrics
from sklearn.metrics import roc_auc_score,average_precision_score
import torch.nn.functional as F
import pytorch_ssim

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
# user
from builders.model_detect_builder import build_model
from utils.utils import setup_seed, init_weight, netParams
from utils.metric.metric import get_iou, Metrics
from utils.metric.score import SegmentationMetric
from utils.losses.loss import LovaszSoftmax, CrossEntropyLoss2d, CrossEntropyLoss2dLabelSmooth,\
    ProbOhemCrossEntropy2d, FocalLoss2d, DiceLoss
from utils.optim import RAdam, Ranger, AdamW
from utils.scheduler.lr_scheduler import WarmupPolyLR
from builders.dataset_detect_builder import build_dataset_train
from builders.dataset_detect_builder import build_dataset_test
from scipy.spatial import distance


sys.setrecursionlimit(1000000)  # solve problem 'maximum recursion depth exceeded'
# torch_ver = torch.__version__[:3]
# print("=====> torch version: ", torch_ver)
GLOBAL_SEED = 0

def parse_args():
    parser = ArgumentParser(description='Efficient semantic segmentation')
    # model and dataset
    parser.add_argument('--model', type=str, default="SegNet", help="model name: (default ENet)")
    parser.add_argument('--dataset', type=str, default="syringe", help="syringe")
    parser.add_argument('--input_size', type=str, default="120,350", help="input size of model")
    parser.add_argument('--input_channels', type=int, default=1, help='input channels of input image')
    parser.add_argument('--output_channels', type=int, default=1, help='output channels of model')
    parser.add_argument('--var', type=float, default=0.01, help='noise var')
    parser.add_argument('--num_workers', type=int, default=4, help=" the number of parallel threads")
    parser.add_argument('--train_type', type=str, default="train",
                        help="ontrain for training on train set, ontrainval for training on train+val set")
    parser.add_argument('--test_small', type=bool, default=False, help="test small model without zeors")
    parser.add_argument('--method', default='Train', type=str, help='pruning method')

    # training hyper params
    parser.add_argument('--max_epochs', type=int, default=300,
                        help="the number of epochs: 300 for train set, 350 for train+val set")
    parser.add_argument('--random_mirror', type=bool, default=False, help="input image random mirror")
    parser.add_argument('--random_scale', type=bool, default=False, help="input image resize 0.5 to 2")
    parser.add_argument('--loss', default='SSIM', type=str, help='training loss')
    parser.add_argument('--lr', type=float, default=0.001, help="initial learning rate")
    parser.add_argument('--batch_size', type=int, default=24, help="the batch size is set to 16 for 2 GPUs")
    parser.add_argument('--optim',type=str.lower,default='adam',choices=['sgd','adam','radam','ranger'],help="select optimizer")
    parser.add_argument('--lr_schedule', type=str, default='warmpoly', help='name of lr schedule: poly')
    parser.add_argument('--num_cycles', type=int, default=1, help='Cosine Annealing Cyclic LR')
    parser.add_argument('--poly_exp', type=float, default=0.9,help='polynomial LR exponent')
    parser.add_argument('--warmup_iters', type=int, default=500, help='warmup iterations')
    parser.add_argument('--warmup_factor', type=float, default=1.0 / 3, help='warm up start lr=warmup_factor*lr')
    parser.add_argument('--use_label_smoothing', action='store_true', default=False, help="CrossEntropy2d Loss with label smoothing or not")
    parser.add_argument('--use_ohem', action='store_true', default=False, help='OhemCrossEntropy2d Loss for cityscapes dataset')
    parser.add_argument('--use_lovaszsoftmax', action='store_true', default=False, help='LovaszSoftmax Loss for cityscapes dataset')
    parser.add_argument('--use_focal', action='store_true', default=False,help=' FocalLoss2d for cityscapes dataset')
    # cuda setting
    parser.add_argument('--cuda', type=bool, default=True, help="running on CPU or GPU")
    parser.add_argument('--gpus', type=str, default="0", help="default GPU devices (0,1)")

    # checkpoint and log
    parser.add_argument('--pretrain', type=str, default="",
                        help="pretrain model path")
    parser.add_argument('--savedir', default="./log/", help="directory to save the model snapshot")
    parser.add_argument('--logFile', default="log.txt", help="storing the training and validation logs")
    parser.add_argument('--compress_rate', type=float, default=1, help='pruning rate for model')
    parser.add_argument('--flag', default='test', type=str, help='iteration for experiment')
    args = parser.parse_args()

    return args


def train_model(args):
    """
    args:
       args: global arguments
    """
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    print("=====> input size:{}".format(input_size))
    print("=====> args:",args)

    if args.cuda:
        print("=====> use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")


    # set the seed
    setup_seed(GLOBAL_SEED)
    print("=====> set Global Seed: ", GLOBAL_SEED)

    cudnn.enabled = True
    print("=====> building network")

    # build the model
    if args.pretrain:
        model = torch.load(args.pretrain,  map_location=lambda storage, loc: storage)
    else:
        model = build_model(args.model, input=args.input_channels, output=args.output_channels)



    print("=====> computing network parameters and FLOPs")
    total_paramters = netParams(model)
    print("the number of parameters: %d ==> %.2f M" % (total_paramters, (total_paramters / 1e6)))

    # load data and data augmentation
    trainLoader, valLoader = build_dataset_train(args.dataset, input_size, args.batch_size, args.train_type,
                                                        args.random_scale, args.random_mirror, args.num_workers, args.var)


    args.per_iter = len(trainLoader)
    args.max_iter = args.max_epochs * args.per_iter

    # define loss function, respectively
    if args.loss == 'SSIM':
        criteria = pytorch_ssim.SSIM()
    elif args.loss == 'MSE':
        criteria = nn.MSELoss()
    else:
        raise NotImplementedError(
            "This repository now supports two datasets: cityscapes and camvid, %s is not included" % args.dataset)

    if args.cuda:
        criteria = criteria.cuda()
        model = model.cuda()  # 1-card data parallel

    #ccreate log folder
    args.logdir = (args.savedir + args.dataset + '/' + "Lock" + "_" +
                    args.model +  "_" + args.method + "_" +
                    str(GLOBAL_SEED) + 'Seed' + '_' + ''+
                    args.loss + "_" +
                    str(args.lr)  + 'lr' + '_' +
                    str(args.batch_size) + "Batch" +'_' +
                    str(args.var) + "Var" + "_" +
                    args.flag + "Flag" + "_" +
                    str(args.max_epochs) + "Epoch" + '/')
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    args.val_img_dir = args.logdir + 'valid_img/'
    if not os.path.exists(args.val_img_dir):
        os.makedirs(args.val_img_dir)

    start_epoch = 0

    model.train()
    cudnn.benchmark = True
    # cudnn.deterministic = True ## my add

    logFileLoc = args.logdir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s Seed: %s" % (str(total_paramters), GLOBAL_SEED))
        logger.write("\n%s\t%s\t%s\t%s" % ('Epoch', 'Loss(Tr)', 'Loss (Va)', 'lr'))
    logger.flush()


    # define optimization strategy
    # define optimization strategy
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    elif args.optim == 'adam':
        # optimizer = torch.optim.Adam(
        #     filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
        #     weight_decay=1e-4)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'radam':
        optimizer = RAdam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.90, 0.999), eps=1e-08,
            weight_decay=1e-4)
    elif args.optim == 'ranger':
        optimizer = Ranger(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.95, 0.999), eps=1e-08,
            weight_decay=1e-4)
    elif args.optim == 'adamw':
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
            weight_decay=1e-4)

    lossTr_list = []
    lossVa_list = []



    epoches = []

    best_val_loss = 100

    print('=====> beginning training')
    for epoch in range(start_epoch, args.max_epochs):
        # training
        lossTr, lr = train(args, trainLoader, model, criteria, optimizer, epoch)
        lossTr_list.append(lossTr)


        # validation
        epoches.append(epoch)
        lossVa = val(args, valLoader, model, criteria)
        lossVa_list.append(lossVa)


        # save best model
        if lossVa < best_val_loss:
            best_val_loss = lossVa
            save_path = args.logdir + 'best_model.pth'
            torch.save(model, save_path)

            # save_path = args.logdir + 'small_model.pth'
            # prune_model = mask.do_prune()
            # torch.save(prune_model, save_path)


        # record train information
        #logger.write("\n%s\t%s\t%s\t%s\t%s" % ('Epoch', 'Loss(Tr)', 'Loss (Va)','ROC(Zero)', 'ROC(Prune)'))
        logger.write("\n%d\t%.7f\t%.7f\t%.7f" % (epoch, lossTr, lossVa, lr))
        logger.flush()
        print("Epoch No.: %d\t Train Loss = %.7f\t Valid Loss = %.7f\t lr= %.7f" % (epoch,
                                                                                    lossTr,
                                                                                    lossVa, lr))

        # Plot the figures
        fig1, ax1 = plt.subplots(figsize=(11, 8))

        ax1.plot(range(start_epoch, epoch + 1), lossTr_list, label='Train Loss')
        ax1.plot(range(start_epoch, epoch + 1), lossVa_list, label='Valid Loss')
        ax1.set_title("Loss Curve")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Current Loss")
        plt.savefig(args.logdir + "loss_vs_epochs.png")
        plt.clf()

        plt.close('all')

    logger.close()


def train(args, train_loader, model, criterion, optimizer, epoch):
    """
    args:
       train_loader: loaded for training dataset
       model: model
       criterion: loss function
       optimizer: optimization algorithm, such as ADAM or SGD
       epoch: epoch number
    return: average loss, per class IoU, and mean IoU
    """

    model.train()
    epoch_loss = []
    total_batches = len(train_loader)

    for iteration, batch in enumerate(train_loader, 0):
        args.per_iter = total_batches
        args.max_iter = args.max_epochs * args.per_iter
        args.cur_iter = epoch * args.per_iter + iteration
        #learming scheduling
        # if args.lr_schedule == 'poly':
        #     lambda1 = lambda epoch: math.pow((1 - (args.cur_iter / args.max_iter)), args.poly_exp)
        #     scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        # elif args.lr_schedule == 'warmpoly':
        #     scheduler = WarmupPolyLR(optimizer, T_max=args.max_iter, cur_iter=args.cur_iter, warmup_factor=1.0 / 3,
        #                          warmup_iters=args.warmup_iters, power=0.9)
        lr = optimizer.param_groups[0]['lr']


        images, noises = batch
        images = images.cuda()
        noises = noises.cuda()

        output = model(noises)
        if args.loss == 'SSIM':
            loss = 1- criterion(output, images)
        else:
            loss = criterion(output, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step() # In pytorch 1.1 .0 and later, should call 'optimizer.step()' before 'lr_scheduler.step()'
        epoch_loss.append(loss.item()) #

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

    return average_epoch_loss_train, lr

def val(args, val_loader, model, criterion):
    """
    args:
      val_loader: loaded for validation dataset
      model: model
    return: mean IoU and IoU class
    """
    # evaluation mode
    model.eval()
    epoch_loss = []


    for iteration, batch in enumerate(val_loader, 0):
        with torch.no_grad():
            images, noises = batch
            images = images.cuda()
            noises = noises.cuda()

            output = model(noises)
            if args.loss == 'SSIM':
                loss = 1 - criterion(output, images)
                anomaly_map = 1 - pytorch_ssim.ssim_map(images, output)
            elif args.loss == 'MSE':
                loss = criterion(output, images)
                anomaly_map = torch.abs(images - output)

            epoch_loss.append(loss.item())  #

            save_image(images, args.val_img_dir + '/' + str("%03d" % iteration) + '_raw.png', normalize=False)
            save_image(output, args.val_img_dir + '/' + str("%03d" % iteration) + '_rec.png', normalize=False)
            save_image(noises, args.val_img_dir + '/' + str("%03d" % iteration) + '_noi.png', normalize=False)
            save_image(anomaly_map, args.val_img_dir + '/' + str("%03d" % iteration) + '_sub.png', normalize=False)





    average_epoch_loss_valid = sum(epoch_loss) / len(epoch_loss)

    return average_epoch_loss_valid

if __name__ == '__main__':
    start = timeit.default_timer()
    args = parse_args()

    if args.dataset == 'syringe':
        args.input_size = '120,350'
        args.input_channels = 1
        args.output_channels = 1
        GLOBAL_SEED = 0
        #args.pretrain = './test_model/model.pth'
    else:
        raise NotImplementedError(
            "This repository now supports two datasets: cityscapes and camvid, %s is not included" % args.dataset)

    train_model(args)
    # rename log folder name
    old_dir = args.logdir
    new_dir = old_dir.replace("Lock", "Finish")
    os.rename(old_dir, new_dir)
