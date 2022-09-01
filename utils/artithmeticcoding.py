import os
import math


# 以二进制的方式读取文件，结果为字节
def fileload(filename):
    file_pth = os.path.dirname(__file__) + '/' + filename
    file_in = os.open(file_pth, os.O_BINARY | os.O_RDONLY)
    file_size = os.stat(file_in)[6]
    data = os.read(file_in, file_size)
    os.close(file_in)
    return data


# 计算文件中不同字节的频数和累积频数
def cal_pr(data):
    pro_dic = {}
    data_set = set(data)
    for i in data_set:
        pro_dic[i] = data.count(i)  # 统计频数
    sym_pro = []  # 频数列表
    accum_pro = []  # 累积频数列表
    keys = []  # 字节名列表
    accum_p = 0
    data_size = len(data)
    for k in sorted(pro_dic, key=pro_dic.__getitem__, reverse=True):
        sym_pro.append(pro_dic[k])
        keys.append(k)
    for i in sym_pro:
        accum_pro.append(accum_p)
        accum_p += i
    accum_pro.append(data_size)
    tmp = 0
    for k in sorted(pro_dic, key=pro_dic.__getitem__, reverse=True):
        pro_dic[k] = [pro_dic[k], accum_pro[tmp]]
        tmp += 1
    return pro_dic, keys, accum_pro


# 小数十进制转二进制
def dec2bin(x_up, x_down, L):
    bins = ""
    while ((x_up != x_down) & (len(bins) < L)):
        x_up *= 2
        if x_up > x_down:
            bins += "1"
            x_up -= x_down
        elif x_up < x_down:
            bins += "0"
        else:
            bins += "1"
    return bins


def encode(data, pro_dic, data_size):
    C_up = 0
    A_up = A_down = C_down = 1
    for i in range(len(data)):
        C_up = C_up * data_size + A_up * pro_dic[data[i]][1]
        C_down = C_down * data_size
        A_up *= pro_dic[data[i]][0]
        A_down *= data_size
    L = math.ceil(len(data) * math.log2(data_size) - math.log2(A_up))  # 计算编码长度
    bin_C = dec2bin(C_up, C_down, L)
    amcode = bin_C[0:L]  # 生成编码
    return L
