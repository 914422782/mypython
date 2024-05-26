import os,sys
import time
import torch
import torchsummary
from torch import optim
import torch.nn as nn
import timeit
import math
from skimage import measure
from statistics import mean
import pandas as pd
import numpy as np
import cv2
import time
import matplotlib
import copy
from numpy import ndarray
from torchvision.tv_tensors import Mask
from torchvision.utils import save_image
from torchsummary import summary
from sklearn import metrics
from sklearn.metrics import roc_auc_score,average_precision_score,auc
from sklearn.metrics import precision_recall_curve, auc
import torch.nn.functional as F
import pytorch_ssim
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from thop import profile

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
torch_ver = torch.__version__[:3]
GLOBAL_SEED = 1234

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
    parser.add_argument('--method', default='SFP', type=str, help='pruning method')
    parser.add_argument('--result_dir', default='SFP', type=str, help='pruning method')

    # training hyper params
    parser.add_argument('--max_epochs', type=int, default=300,
                        help="the number of epochs: 300 for train set, 350 for train+val set")
    parser.add_argument('--random_mirror', type=bool, default=False, help="input image random mirror")
    parser.add_argument('--random_scale', type=bool, default=False, help="input image resize 0.5 to 2")
    parser.add_argument('--loss', default='SSIM', type=str, help='training loss')
    parser.add_argument('--lr', type=float, default=0.001, help="initial learning rate")
    parser.add_argument('--batch_size', type=int, default=16, help="the batch size is set to 16 for 2 GPUs")
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
        model = torch.load(args.pretrain)
    else:
        model = build_model(args.model, input=args.input_channels, output=args.output_channels)

    #make mask
    mask = Mask(model)
    mask.init_length()

    print("=====> computing network parameters and FLOPs")
    total_paramters = netParams(model)
    print("the number of parameters: %d ==> %.2f M" % (total_paramters, (total_paramters / 1e6)))

    # load data and data augmentation
    trainLoader, valLoader = build_dataset_train(args.dataset, input_size, args.batch_size, args.train_type,
                                                        args.random_scale, args.random_mirror, args.num_workers, args.var)

    # load the test set
    testLoader = build_dataset_test(args.dataset, input_size, 1, args.num_workers, args.var)


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
    args.logdir = (args.savedir + args.dataset + '/' + "Lock" + "_P_" +
                    args.model +  "_" + args.method + "_" +
                    args.loss + "_" +
                    str(args.compress_rate) + "Rate" + "_" +
                    str(args.var) + "Var" + "_" +
                    args.flag + "Flag" + "_" +
                    str(args.max_epochs) + "Epoch" + '/')
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    args.train_img_dir = args.logdir + 'train_img/'
    if not os.path.exists(args.train_img_dir):
        os.makedirs(args.train_img_dir)

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
        logger.write("\n%s\t%s\t%s\t%s\t%s" % ('Epoch', 'Loss(Tr)', 'Loss (Va)','AUROC_PX', 'AUROC_SP'))
    logger.flush()


    # define optimization strategy
    # define optimization strategy
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
            weight_decay=1e-4)
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

    auroc_px_zero_list = []
    auroc_sp_zero_list = []

    epoches = []

    best_auroc_px_zero = 0


    print('=====> beginning training')
    for epoch in range(start_epoch, args.max_epochs):
        # training
        lossTr, lr = train(args, trainLoader, model, criteria, optimizer, epoch)
        lossTr_list.append(lossTr)


        #execute sfp
        mask.model = model
        mask.init_mask(args.compress_rate)
        mask.do_mask()
        model = mask.model


        # validation
        epoches.append(epoch)
        lossVa = val(args, valLoader, model, criteria)
        lossVa_list.append(lossVa)

        #testing
        auroc_px_zero, auroc_sp_zero = test(args, testLoader, model)
        auroc_px_zero_list.append(auroc_px_zero)
        auroc_sp_zero_list.append(auroc_sp_zero)


        # save best model
        if auroc_px_zero > best_auroc_px_zero:
            best_auroc_px_zero = auroc_px_zero
            save_path = args.logdir + 'best_model'+'_aurocpx'+str(auroc_px_zero)+"_aurocsp"+str(auroc_sp_zero)+".pth"
            torch.save(model, save_path)

            save_path = args.logdir + 'small_model.pth'
            prune_model = mask.do_prune()
            torch.save(prune_model, save_path)



        # record train information
        #logger.write("\n%s\t%s\t%s\t%s\t%s" % ('Epoch', 'Loss(Tr)', 'Loss (Va)','ROC(Zero)', 'ROC(Prune)'))
        logger.write("\n%d\t%.7f\t%.7f\t%f\t%.7f" % (epoch, lossTr, lossVa, auroc_px_zero, auroc_sp_zero))
        logger.flush()
        print("Epoch No.: %d\t Train Loss = %.7f\t Valid Loss = %.7f\t AUROC_PX = %.7f\t AUROC_SP=%.7f\t lr= %.7f" % (epoch,
                                                                                    lossTr,
                                                                                    lossVa, auroc_px_zero, auroc_sp_zero, lr))

        # Plot the figures
        fig1, ax1 = plt.subplots(figsize=(11, 8))

        ax1.plot(range(start_epoch, epoch + 1), lossTr_list, label='Train Loss')
        ax1.plot(range(start_epoch, epoch + 1), lossVa_list, label='Valid Loss')
        ax1.set_title("Loss Curve")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Current Loss")
        plt.savefig(args.logdir + "loss_vs_epochs.png")
        plt.clf()

        fig2, ax2 = plt.subplots(figsize=(11, 8))
        ax2.plot(epoches, auroc_px_zero_list, label="AUROC_PX")
        ax2.plot(epoches, auroc_sp_zero_list, label="AUROC_SP")
        ax2.set_title("mIoU vs epochs")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Current mIoU")
        plt.legend(loc='lower right')

        plt.savefig(args.logdir + "auroc_vs_epochs.png")

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
        # learming scheduling
        if args.lr_schedule == 'poly':
            lambda1 = lambda epoch: math.pow((1 - (args.cur_iter / args.max_iter)), args.poly_exp)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        elif args.lr_schedule == 'warmpoly':
            scheduler = WarmupPolyLR(optimizer, T_max=args.max_iter, cur_iter=args.cur_iter, warmup_factor=1.0 / 3,
                                 warmup_iters=args.warmup_iters, power=0.9)
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
        scheduler.step() # In pytorch 1.1 .0 and later, should call 'optimizer.step()' before 'lr_scheduler.step()'
        epoch_loss.append(loss.item()) #

        save_image(images, args.train_img_dir + '/' + str("%03d" % iteration) + '_raw.png', normalize=False)
        save_image(output, args.train_img_dir + '/' + str("%03d" % iteration) + '_rec.png', normalize=False)
        save_image(noises, args.train_img_dir + '/' + str("%03d" % iteration) + '_noi.png', normalize=False)



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

            output = model(images)
            if args.loss == 'SSIM':
                loss = 1 - criterion(output, images)
            else:
                loss = criterion(output, images)
            epoch_loss.append(loss.item())  #



    average_epoch_loss_valid = sum(epoch_loss) / len(epoch_loss)

    return average_epoch_loss_valid

def test(args, test_loader, model):
    """
    args:
      val_loader: loaded for validation dataset
      model: model
    return: mean IoU and IoU class
    """
    # evaluation mode
    model.eval()

    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []


    for iteration, batch in enumerate(test_loader, 0):
        with torch.no_grad():
            images, noises, gt = batch
            images = images.cuda()
            output = model(images)


            if args.loss == 'SSIM':
                anomaly_map = 1 - pytorch_ssim.ssim_map(images, output)
            else:
                anomaly_map = 1 - pytorch_ssim.ssim_map(images, output)

            anomaly_map = anomaly_map/2

            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.cpu().numpy().ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map.cpu().numpy()))

    auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 7)
    auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 7)

    return auroc_px, auroc_sp

def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

#     df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    d = {'pro':[], 'fpr':[],'threshold': []}
    binary_amaps = np.zeros_like(amaps, dtype=np.bool_)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

#         df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
        d['pro'].append(mean(pros))
        d['fpr'].append(fpr)
        d['threshold'].append(th)
    df = pd.DataFrame(d)
    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc



if __name__ == '__main__':
    start = timeit.default_timer()
    args = parse_args()

    args.dataset = 'syringe'

    if args.dataset == 'syringe':
        args.input_size = (120,350)
        args.batch_size =1
        args.input_channels = 1
        args.output_channels = 1
        args.aucpro = True
    else:
        raise NotImplementedError(
            "This repository now supports two datasets: cityscapes and camvid, %s is not included" % args.dataset)

    #set model dir
    #args.pretrain = './log/syringe/Finish_P_UNet_HFP_SSIM_0.5Rate_0.15Var_1Flag_200Epoch/best_model.pth'
    #model_path = './test_model/ssim_model.pth'
    model = torch.load(args.pretrain)
    model = model.cuda()

    #create visualization results dir
    result_dir = args.result_dir + 'visual_result_' + str(args.batch_size)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)


    # load the test set
    testLoader = build_dataset_test(args.dataset, args.input_size, args.batch_size, args.num_workers, args.var)

    #test model
    model.eval()

    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []


    start_time = time.time()
    for iteration, batch in enumerate(testLoader, 0):
        with torch.no_grad():
            images, noises, gt = batch
            images = images.cuda()
            output = model(images)


            if args.loss == 'SSIM':
                anomaly_map =  1- pytorch_ssim.ssim_map(images, output)
            else:
                anomaly_map = torch.abs(images-output)

            save_image(images, result_dir + '/' + str("%03d" % iteration) + '_raw.png', normalize=False)
            save_image(output, result_dir + '/' + str("%03d" % iteration) + '_rec.png', normalize=False)
            save_image(noises, result_dir + '/' + str("%03d" % iteration) + '_noi.png', normalize=False)
            save_image(gt, result_dir + '/' + str("%03d" % iteration) + '_gt.png', normalize=False)
            save_image(anomaly_map, result_dir + '/' + str("%03d" % iteration) + '_sub.png', normalize=False)

            #anomaly_map = anomaly_map*2.5


            enhance = images + anomaly_map
            enhance = enhance/torch.max(enhance)
            save_image(enhance, result_dir + '/' + str("%03d" % iteration) + '_enh.png', normalize=False)


            enhance = cv2.imread(result_dir + '/' + str("%03d" % iteration) + '_enh.png')
            heatmap = cv2.applyColorMap(enhance, cv2.COLORMAP_JET)
            cv2.imwrite(result_dir + '/' + str("%03d" % iteration) + '_heat.png', heatmap)

            enhance = cv2.imread(result_dir + '/' + str("%03d" % iteration) + '_sub.png')
            heatmap = cv2.applyColorMap(enhance, cv2.COLORMAP_JET)
            cv2.imwrite(result_dir + '/' + str("%03d" % iteration) + '_sub_heat.png', heatmap)



            anomaly_map = gaussian_filter(anomaly_map[0, 0, ::].cpu().numpy(), sigma=4)
            #anomaly_map = anomaly_map[0, 0, ::].cpu().numpy()



            if args.batch_size == 1:
                #
                #anomaly_map = F.softmax(anomaly_map, dim=-1)
                gt[gt > 0.5] = 1
                gt[gt <= 0.5] = 0

                if np.max(gt.cpu().numpy()) > 0:
                    aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                                                  anomaly_map[np.newaxis, :, :]))


                gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
                pr_list_px.extend(anomaly_map.ravel())
                gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
                pr_list_sp.append(np.max(anomaly_map))

    end_time = time.time()
    totoal_time = end_time-start_time
    per_time = totoal_time/(iteration+1)



    if args.batch_size == 1:
        try:
            auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 4)
        except Exception:
            auroc_px = 0
        try:
            auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)
        except Exception:
            auroc_sp = 0
        try:
            aupro = round(np.mean(aupro_list), 4)
        except Exception:
            aupro = 0
        try:
            ap = round(average_precision_score(gt_list_sp, pr_list_sp),4)
        except Exception:
            ap = 0
        input = torch.randn(1, 1, 120, 350)
        input = input.cuda()
        Flops, params = profile(model, inputs=(input,))
        print('%s AUROC_PX=%.4f AUROC_SP=%.4f PRO=%.4f AP=%.4f FLOPs=%.4f Params=%.4fM Infertime=%.4f'%
              (args.pretrain, auroc_px, auroc_sp,aupro, ap, Flops/ 1000000000, params/ 1000000, per_time))