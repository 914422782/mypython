# import importlib,sys
# importlib.reload(sys)
import numpy as np


import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams["font.sans-serif"] = ['STHeiti']
plt.rcParams["axes.unicode_minus"] = False
# from matplotlib.font_manager import FontProperties
# font = FontProperties(fname='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf', size=14)

plt.rcParams["font.sans-serif"] = ['Songti SC']
plt.rcParams["axes.unicode_minus"] = False

def load_data(path):
    file = open(path, "r", encoding="utf8")
    data = {}
    epoch = []
    loss_tr = []
    loss_va = []

    lines = []
    for line in file.readlines():
        content = line.strip("\n")
        lines.append(content)
    lines = lines[2:]



    for item in lines:
        arr = item.split("\t");
        epoch.append(int(arr[0]))
        loss_tr.append(1-float(arr[1]))
        loss_va.append(1-float(arr[2]))


    data['epoch'] = epoch
    data['loss_tr'] = loss_tr
    data['loss_va'] = loss_va

    return data

def show_unet_and_segnet_loss():
    unet_data = load_data('./log/syringe/Finish_UNet_Train_0Seed_SSIM_0.001lr_64Batch_0.05Var_2Flag_200Epoch/log.txt')
    segnet_data = load_data('./log/syringe/Finish_SegNet_Train_0Seed_SSIM_0.001lr_64Batch_0.05Var_1Flag_200Epoch/log.txt')
    plt.figure(figsize=(12, 6))

    plt.plot(unet_data['epoch'], unet_data['loss_va'], label='UNet')
    plt.plot(segnet_data['epoch'], segnet_data['loss_va'], label='SegNet')

    plt.ylim(0.5, 1)
    plt.xlim(0, len(unet_data['epoch']))
    plt.grid()
    # 常规的绘图轴、标题设置
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Reconstruction Precision', fontsize=18)
    #plt.title("Reconstruction Precision on Valid Set", fontsize=15)
    plt.legend(fontsize=18)  # 调整图例的大小

    plt.savefig("./temp.pdf")

def show_unet_pruning_loss():
    a = 1
    # hfp_data = load_data('./log/syringe/Finish_P_UNet_HFP_SSIM_0.1Rate_0.05Var_2Flag_100Epoch/log.txt')
    # sfp_data = load_data(
    #     './log/syringe/Finish_P_UNet_SFP_SSIM_0.1Rate_0.05Var_2Flag_100Epoch/log.txt')
    # fpgm_data = load_data(
    #     './log/syringe/Finish_P_UNet_FPGM_SSIM_0.1Rate_0.05Var_2Flag_100Epoch/log.txt')
    # our_data = load_data(
    #     './log/syringe/Finish_P_UNet_Taylor_SSIM_0.1Rate_0.5Decay_0.05Var_2Flag_100Epoch/log.txt')

    hfp_data = load_data('./log/syringe/Finish_P_SegNet_HFP_SSIM_0.1Rate_0.15Var_1Flag_100Epoch/log.txt')
    sfp_data = load_data(
        './log/syringe/Finish_P_SegNet_SFP_SSIM_0.1Rate_0.15Var_1Flag_100Epoch/log.txt')
    fpgm_data = load_data(
        './log/syringe/Finish_P_SegNet_FPGM_SSIM_0.1Rate_0.15Var_1Flag_100Epoch/log.txt')
    our_data = load_data(
        './log/syringe/Finish_P_SegNet_Taylor_SSIM_0.1Rate_0.5Decay_0.15Var_1Flag_100Epoch/log_modify.txt')


    plt.figure(figsize=(12, 6))



    plt.plot(hfp_data['epoch'], hfp_data['loss_va'], label='HFP')
    plt.plot(sfp_data['epoch'], sfp_data['loss_va'], label='SFP')
    plt.plot(fpgm_data['epoch'], fpgm_data['loss_va'], label='FPGM')
    plt.plot(our_data['epoch'], our_data['loss_va'], label='本文方法', marker='^')

    plt.ylim(0.1, 0.96)
    plt.xlim(0, len(hfp_data['epoch']))
    plt.grid()
    # 常规的绘图轴、标题设置
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Reconstruction Precision', fontsize=18)
    # plt.title("Reconstruction Precision on Valid Set", fontsize=15)
    plt.legend(fontsize=18)  # 调整图例的大小

    plt.savefig("./temp.pdf")

def relu(x):
    return np.where(x < 0, 0, x)


def show_relu():
    x = np.arange(-10, 10, 0.1)
    y = relu(x)

    plt.figure(figsize=(10, 6))

    plt.plot(x, y, label='ReLu', color="red", lw=2)


    plt.xlim(-10.5, 10.5)
    plt.ylim(-0.5, 10.5)
    plt.grid()
    # 常规的绘图轴、标题设置
    # plt.title("Reconstruction Precision on Valid Set", fontsize=15)
    plt.legend(fontsize=18)  # 调整图例的大小

    plt.savefig("./temp.pdf")



if __name__ == '__main__':
    #show_unet_and_segnet_loss()
    #show_unet_pruning_loss()
    show_relu()