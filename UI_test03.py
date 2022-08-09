
##################################################
# 放置少女　育成シミュレータ Streamlit版
#  Ver 0.10 2022/8/9
###################################################

import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import copy

########################################
# 定数
SW_VERSION = '0.10'

CLASS_C_J_MAX = 26  # 上昇値のパターン数
CLASS_B_J_MAX = 24
CLASS_A_J_MAX = 22

FAST_MODE_RESO = 1000   # 高速計算時の分解能

STEP_MAX = 4    # 計算段階上限
STATUS_NUM = 4   # ステータス数

I_KIN = 0   # インデックス用
I_BIN = 1
I_CHI = 2
I_TAI = 3

########################################
# グローバル変数
v_c = []    # インデックス→上昇値変換
v_b = []
v_a = []
B_c = [i for i in range(0, 16)] # 符号組み合わせケース数
B_b = [i for i in range(0, 16)]
B_a = [i for i in range(0, 16)]

# グラフ用プロファイル格納
num_pro = []                                # 育成回数(X軸)
att_pro = [[] * 4 for i in [1] * 4]         # 属性値
att_rate_pro = [[] * 4 for i in [1] * 4]    # 属性値(割合)
ev_pro = [[] * 4 for i in [1] * 4]          # 期待値

########################################
# ダミー処理
def dummy():
    return 0

########################################
# C級プラス出現確率    x: 育成割合(=属性値/属性値上限)
def cal_p_c(x):
    a = -17
    b = 0.055
    c = 1
    d = 0.2436
    e = 0.265936288
    
    if x < b:
        return 1
    elif x < e:
        return a * (x - b) ** 2 + c
    else:
        return d

########################################
# B級プラス出現確率    x: 育成割合(=属性値/属性値上限)
def cal_p_b(x):
    a = -17
    b = 0.29
    c = 1
    d = 0.38166
    e = 0.480716914
    
    if x < b:
        return 1
    elif x < e:
        return a * (x - b) ** 2 + c
    else:
        return d

########################################
# A級プラス出現確率    x: 育成割合(=属性値/属性値上限)
def cal_p_a(x):
    a = -17
    b = 0.52
    c = 1
    d = 0.56463
    e = 0.680031247
    
    if x < b:
        return 1
    elif x < e:
        return a * (x - b) ** 2 + c
    else:
        return d

########################################
# 符号組み合わせケース別確率
def cal_pp(p):

    pp = [0] * 16
    
    pp[0] = (1 - p[I_KIN]) * (1 - p[I_BIN]) * (1 - p[I_CHI]) * (1 - p[I_TAI])   #----
    pp[1] = (1 - p[I_KIN]) * (1 - p[I_BIN]) * (1 - p[I_CHI]) * p[I_TAI]         #---+
    pp[2] = (1 - p[I_KIN]) * (1 - p[I_BIN]) * p[I_CHI] * (1 - p[I_TAI])         #--+-
    pp[3] = (1 - p[I_KIN]) * (1 - p[I_BIN]) * p[I_CHI] * p[I_TAI]               #--++
    pp[4] = (1 - p[I_KIN]) * p[I_BIN] * (1 - p[I_CHI]) * (1 - p[I_TAI])         #-+--
    pp[5] = (1 - p[I_KIN]) * p[I_BIN] * (1 - p[I_CHI]) * p[I_TAI]               #-+-+
    pp[6] = (1 - p[I_KIN]) * p[I_BIN] * p[I_CHI] * (1 - p[I_TAI])               #-++-
    pp[7] = (1 - p[I_KIN]) * p[I_BIN] * p[I_CHI] * p[I_TAI]                     #-+++
    pp[8] = p[I_KIN] * (1 - p[I_BIN]) * (1 - p[I_CHI]) * (1 - p[I_TAI])         #+---
    pp[9] = p[I_KIN] * (1 - p[I_BIN]) * (1 - p[I_CHI]) * p[I_TAI]               #+--+
    pp[10] = p[I_KIN] * (1 - p[I_BIN]) * p[I_CHI] * (1 - p[I_TAI])              #+-+-
    pp[11] = p[I_KIN] * (1 - p[I_BIN]) * p[I_CHI] * p[I_TAI]                    #+-++
    pp[12] = p[I_KIN] * p[I_BIN] * (1 - p[I_CHI]) * (1 - p[I_TAI])              #++--
    pp[13] = p[I_KIN] * p[I_BIN] * (1 - p[I_CHI]) * p[I_TAI]                    #++-+
    pp[14] = p[I_KIN] * p[I_BIN] * p[I_CHI] * (1 - p[I_TAI])                    #+++-
    pp[15] = p[I_KIN] * p[I_BIN] * p[I_CHI] * p[I_TAI]                          #++++

    return pp

########################################
# 定数配列初期化
def init01():

    #C級の上昇値は-12～-1, +1～+14
    v_c.extend(i for i in range(-12, 0))
    v_c.extend(i for i in range(1, 15))

    #B級の上昇値は-12～-1, +6～+17
    v_b.extend(i for i in range(-12, 0))
    v_b.extend(i for i in range(6, 18))
    
    #A級の上昇値は-12～-1, +16～+25
    v_a.extend(i for i in range(-12, 0))
    v_a.extend(i for i in range(16, 26))

    #C級　マイナス：-12～-1, プラス：+1～+14, 筋力/敏捷/知力/体力の符号組み合わせケース数
    B_c[0] = 20736      #---- 12*12*12*12
    B_c[1] = 24192      #---+ 12*12*12*14
    B_c[2] = 24192      #--+- 12*12*14*12
    B_c[3] = 28224      #--++ 12*12*14*14
    B_c[4] = 24192      #-+-- 12*14*12*12
    B_c[5] = 28224      #-+-+ 12*14*12*14
    B_c[6] = 28224      #-++- 12*14*14*12
    B_c[7] = 32928      #-+++ 12*14*14*14
    B_c[8] = 24192      #+--- 14*12*12*12
    B_c[9] = 28224      #+--+ 14*12*12*14
    B_c[10] = 28224     #+-+- 14*12*14*12
    B_c[11] = 32928     #+-++ 14*12*14*14
    B_c[12] = 28224     #++-- 14*14*12*12
    B_c[13] = 32928     #++-+ 14*14*12*14
    B_c[14] = 32928     #+++- 14*14*14*12
    B_c[15] = 38416     #++++ 14*14*14*14
    
    #B級　マイナス：-12～-1, プラス：+6～+17, 筋力/敏捷/知力/体力の符号組み合わせケース数
    for i in range(0, 16):
        B_b[i] = 20736  #B級は全部 12^4

    #A級　マイナス：-12～-1, プラス：+16～+25, 筋力/敏捷/知力/体力の符号組み合わせケース数
    B_a[0] = 20736      #---- 12*12*12*12
    B_a[1] = 17280      #---+ 12*12*12*10
    B_a[2] = 17280      #--+- 12*12*10*12
    B_a[3] = 14400      #--++ 12*12*10*10
    B_a[4] = 17280      #-+-- 12*10*12*12
    B_a[5] = 14400      #-+-+ 12*10*12*10
    B_a[6] = 14400      #-++- 12*10*10*12
    B_a[7] = 12000      #-+++ 12*10*10*10
    B_a[8] = 17280      #+--- 10*12*12*12
    B_a[9] = 14400      #+--+ 10*12*12*10
    B_a[10] = 14400     #+-+- 10*12*10*12
    B_a[11] = 12000     #+-++ 10*12*10*10
    B_a[12] = 14400     #++-- 10*10*12*12
    B_a[13] = 12000     #++-+ 10*10*12*10
    B_a[14] = 12000     #+++- 10*10*10*12
    B_a[15] = 10000     #++++ 10*10*10*10

########################################
# C級育成処理
def tr_class_C(tr_num, tr_num_ini, tr_w, tr_bias, att_ini, att_max, fast_mode_rasio):
    
    a = [0] * 4     # 育成出現値
    s = [0] * 4     # 育成出現値符号
    p = [0] * 4     # 育成プラス出現確率
    pp = [0] * 16   # ケース別確率
    pd = [[[0] * 16 for i in range(CLASS_C_J_MAX)] for j in range(4)]  # 確率分布
    ev = [0] * 4    # 期待値
    n = [[[0] * 16 for i in range(CLASS_C_J_MAX)] for j in range(4)]   # 育成結果保存判定ケース数
    r = [[[0] * 16 for i in range(CLASS_C_J_MAX)] for j in range(4)]   # 育成結果保存判定ケース割合
    att = copy.copy(att_ini)   # 属性値

    progress_bar = st.progress(0)   # progress_bar設置    

    # 育成結果保存判定ケース数のカウント処理
    for j_k in range(0, CLASS_C_J_MAX): # 筋力 -12～-1, +1～+14のループ
        a[I_KIN] = v_c[j_k]
        if a[I_KIN] > 0:
            s[I_KIN] = 8
        else:
            s[I_KIN] = 0
        
        for j_b in range(0, CLASS_C_J_MAX): # 敏捷 -12～-1, +1～+14のループ
            a[I_BIN] = v_c[j_b]
            if a[I_BIN] > 0:
                s[I_BIN] = 4
            else:
                s[I_BIN] = 0
            
            for j_c in range(0, CLASS_C_J_MAX): # 知力 -12～-1, +1～+14のループ
                a[I_CHI] = v_c[j_c]
                if a[I_CHI] > 0:
                    s[I_CHI] = 2
                else:
                    s[I_CHI] = 0
                
                for j_t in range(0, CLASS_C_J_MAX): # 体力 -12～-1, +1～+14のループ
                    a[I_TAI] = v_c[j_t]
                    if a[I_TAI] > 0:
                        s[I_TAI] = 1
                    else:
                        s[I_TAI] = 0
                    
                    # 育成結果の保存判定用　評価値
                    e = a[I_KIN] * tr_w[I_KIN] + a[I_BIN] * tr_w[I_BIN] + a[I_CHI] * tr_w[I_CHI] + a[I_TAI] * tr_w[I_TAI] + tr_bias
                    
                    # ケースNo.
                    ss = s[I_KIN] + s[I_BIN] + s[I_CHI] + s[I_TAI]
                    
                    if e > 0:   # 保存判定ケース数
                        n[I_KIN][j_k][ss] = n[I_KIN][j_k][ss] + 1
                        n[I_BIN][j_b][ss] = n[I_BIN][j_b][ss] + 1
                        n[I_CHI][j_c][ss] = n[I_CHI][j_c][ss] + 1
                        n[I_TAI][j_t][ss] = n[I_TAI][j_t][ss] + 1

    # 育成結果保存判定ケース割合
    for i in range(I_KIN, I_TAI + 1):
        for j in range(0, CLASS_C_J_MAX):
            for ss in range(0, 16):
                r[i][j][ss] = n[i][j][ss] / B_c[ss]

    # 期待値計算
    tr_num = round(tr_num * fast_mode_rasio)
    for tr_cnt in range(1, tr_num + 1):
        num_pro.append(tr_cnt / fast_mode_rasio + tr_num_ini)
        
        # progress_bar
        pro = (tr_cnt * 100) // tr_num
        if pro % 5 == 0:
            progress_bar.progress(pro)

        # 育成プラス出現率
        p[I_KIN] = cal_p_c(att[I_KIN] / att_max[I_KIN])
        p[I_BIN] = cal_p_c(att[I_BIN] / att_max[I_BIN])
        p[I_CHI] = cal_p_c(att[I_CHI] / att_max[I_CHI])
        p[I_TAI] = cal_p_c(att[I_TAI] / att_max[I_TAI])
        
        # ケース別確率
        pp = cal_pp(p)
        
        for i in range(I_KIN, I_TAI + 1):
            ev[i] = 0
            for j in range(0, CLASS_C_J_MAX):
                for ss in range(0, 16):
                    pd[i][j][ss] = r[i][j][ss] * pp[ss]     # 確率分布
                    ev[i] = ev[i] + pd[i][j][ss] * v_c[j]   # 期待値

        # 属性値を期待値分増加
        for i in range(I_KIN, I_TAI + 1):
            att[i] = att[i] + ev[i] / fast_mode_rasio
            if att[i] > att_max[i]:
                att[i] = att_max[i]
            
            # 期待値・属性値のプロファイルを保存
            ev_pro[i].append(ev[i])
            att_pro[i].append(att[i])
            att_rate_pro[i].append(att[i] / att_max[i])

########################################
# B級育成処理
def tr_class_B(tr_num, tr_num_ini, tr_w, tr_bias, att_ini, att_max, fast_mode_rasio):
    
    a = [0] * 4     # 育成出現値
    s = [0] * 4     # 育成出現値符号
    p = [0] * 4     # 育成プラス出現確率
    pp = [0] * 16   # ケース別確率
    pd = [[[0] * 16 for i in range(CLASS_B_J_MAX)] for j in range(4)]  # 確率分布
    ev = [0] * 4    # 期待値
    n = [[[0] * 16 for i in range(CLASS_B_J_MAX)] for j in range(4)]   # 育成結果保存判定ケース数
    r = [[[0] * 16 for i in range(CLASS_B_J_MAX)] for j in range(4)]   # 育成結果保存判定ケース割合
    att = copy.copy(att_ini)   # 属性値

    progress_bar = st.progress(0)   # progress_bar設置    
    
    # 育成結果保存判定ケース数のカウント処理
    for j_k in range(0, CLASS_B_J_MAX): # 筋力 -12～-1, +6～+17のループ
        a[I_KIN] = v_b[j_k]
        if a[I_KIN] > 0:
            s[I_KIN] = 8
        else:
            s[I_KIN] = 0
        
        for j_b in range(0, CLASS_B_J_MAX): # 敏捷 -12～-1, +6～+17のループ
            a[I_BIN] = v_b[j_b]
            if a[I_BIN] > 0:
                s[I_BIN] = 4
            else:
                s[I_BIN] = 0
            
            for j_c in range(0, CLASS_B_J_MAX): # 知力 -12～-1, +6～+17のループ
                a[I_CHI] = v_b[j_c]
                if a[I_CHI] > 0:
                    s[I_CHI] = 2
                else:
                    s[I_CHI] = 0
                
                for j_t in range(0, CLASS_B_J_MAX): # 体力 -12～-1, +6～+17のループ
                    a[I_TAI] = v_b[j_t]
                    if a[I_TAI] > 0:
                        s[I_TAI] = 1
                    else:
                        s[I_TAI] = 0
                    
                    # 育成結果の保存判定用　評価値
                    e = a[I_KIN] * tr_w[I_KIN] + a[I_BIN] * tr_w[I_BIN] + a[I_CHI] * tr_w[I_CHI] + a[I_TAI] * tr_w[I_TAI] + tr_bias
                    
                    # ケースNo.
                    ss = s[I_KIN] + s[I_BIN] + s[I_CHI] + s[I_TAI]
                    
                    if e > 0:   # 保存判定ケース数
                        n[I_KIN][j_k][ss] = n[I_KIN][j_k][ss] + 1
                        n[I_BIN][j_b][ss] = n[I_BIN][j_b][ss] + 1
                        n[I_CHI][j_c][ss] = n[I_CHI][j_c][ss] + 1
                        n[I_TAI][j_t][ss] = n[I_TAI][j_t][ss] + 1

    # 育成結果保存判定ケース割合
    for i in range(I_KIN, I_TAI + 1):
        for j in range(0, CLASS_B_J_MAX):
            for ss in range(0, 16):
                r[i][j][ss] = n[i][j][ss] / B_b[ss]

    # 期待値計算
    tr_num = round(tr_num * fast_mode_rasio)
    for tr_cnt in range(1, tr_num + 1):
        num_pro.append(tr_cnt / fast_mode_rasio + tr_num_ini)

        # progress_bar
        pro = (tr_cnt * 100) // tr_num
        if pro % 5 == 0:
            progress_bar.progress(pro)

        # 育成プラス出現率
        p[I_KIN] = cal_p_b(att[I_KIN] / att_max[I_KIN])
        p[I_BIN] = cal_p_b(att[I_BIN] / att_max[I_BIN])
        p[I_CHI] = cal_p_b(att[I_CHI] / att_max[I_CHI])
        p[I_TAI] = cal_p_b(att[I_TAI] / att_max[I_TAI])
        
        # ケース別確率
        pp = cal_pp(p)
        
        for i in range(I_KIN, I_TAI + 1):
            ev[i] = 0
            for j in range(0, CLASS_B_J_MAX):
                for ss in range(0, 16):
                    pd[i][j][ss] = r[i][j][ss] * pp[ss]     # 確率分布
                    ev[i] = ev[i] + pd[i][j][ss] * v_b[j]   # 期待値

        # 属性値を期待値分増加
        for i in range(I_KIN, I_TAI + 1):
            att[i] = att[i] + ev[i] / fast_mode_rasio
            if att[i] > att_max[i]:
                att[i] = att_max[i]
            
            # 期待値・属性値のプロファイルを保存
            ev_pro[i].append(ev[i])
            att_pro[i].append(att[i])
            att_rate_pro[i].append(att[i] / att_max[i])

########################################
# A級育成処理
def tr_class_A(tr_num, tr_num_ini, tr_w, tr_bias, att_ini, att_max, fast_mode_rasio):
    
    a = [0] * 4     # 育成出現値
    s = [0] * 4     # 育成出現値符号
    p = [0] * 4     # 育成プラス出現確率
    pp = [0] * 16   # ケース別確率
    pd = [[[0] * 16 for i in range(CLASS_A_J_MAX)] for j in range(4)]  # 確率分布
    ev = [0] * 4    # 期待値
    n = [[[0] * 16 for i in range(CLASS_A_J_MAX)] for j in range(4)]   # 育成結果保存判定ケース数
    r = [[[0] * 16 for i in range(CLASS_A_J_MAX)] for j in range(4)]   # 育成結果保存判定ケース割合
    att = copy.copy(att_ini)   # 属性値

    progress_bar = st.progress(0)   # progress_bar設置    
        
    # 育成結果保存判定ケース数のカウント処理
    for j_k in range(0, CLASS_A_J_MAX): # 筋力 -12～-1, +16～+25のループ
        a[I_KIN] = v_a[j_k]
        if a[I_KIN] > 0:
            s[I_KIN] = 8
        else:
            s[I_KIN] = 0
        
        for j_b in range(0, CLASS_A_J_MAX): # 敏捷 -12～-1, +16～+25のループ
            a[I_BIN] = v_a[j_b]
            if a[I_BIN] > 0:
                s[I_BIN] = 4
            else:
                s[I_BIN] = 0
            
            for j_c in range(0, CLASS_A_J_MAX): # 知力 -12～-1, +16～+25のループ
                a[I_CHI] = v_a[j_c]
                if a[I_CHI] > 0:
                    s[I_CHI] = 2
                else:
                    s[I_CHI] = 0
                
                for j_t in range(0, CLASS_A_J_MAX): # 体力 -12～-1, +16～+25のループ
                    a[I_TAI] = v_a[j_t]
                    if a[I_TAI] > 0:
                        s[I_TAI] = 1
                    else:
                        s[I_TAI] = 0
                    
                    # 育成結果の保存判定用　評価値
                    e = a[I_KIN] * tr_w[I_KIN] + a[I_BIN] * tr_w[I_BIN] + a[I_CHI] * tr_w[I_CHI] + a[I_TAI] * tr_w[I_TAI] + tr_bias
                    
                    # ケースNo.
                    ss = s[I_KIN] + s[I_BIN] + s[I_CHI] + s[I_TAI]
                    
                    if e > 0:   # 保存判定ケース数
                        n[I_KIN][j_k][ss] = n[I_KIN][j_k][ss] + 1
                        n[I_BIN][j_b][ss] = n[I_BIN][j_b][ss] + 1
                        n[I_CHI][j_c][ss] = n[I_CHI][j_c][ss] + 1
                        n[I_TAI][j_t][ss] = n[I_TAI][j_t][ss] + 1

    # 育成結果保存判定ケース割合
    for i in range(I_KIN, I_TAI + 1):
        for j in range(0, CLASS_A_J_MAX):
            for ss in range(0, 16):
                r[i][j][ss] = n[i][j][ss] / B_a[ss]

    # 期待値計算
    tr_num = round(tr_num * fast_mode_rasio)
    for tr_cnt in range(1, tr_num + 1):
        num_pro.append(tr_cnt / fast_mode_rasio + tr_num_ini)

        # progress_bar
        pro = (tr_cnt * 100) // tr_num
        if pro % 5 == 0:
            progress_bar.progress(pro)

        # 育成プラス出現率
        p[I_KIN] = cal_p_a(att[I_KIN] / att_max[I_KIN])
        p[I_BIN] = cal_p_a(att[I_BIN] / att_max[I_BIN])
        p[I_CHI] = cal_p_a(att[I_CHI] / att_max[I_CHI])
        p[I_TAI] = cal_p_a(att[I_TAI] / att_max[I_TAI])
        
        # ケース別確率
        pp = cal_pp(p)
        
        for i in range(I_KIN, I_TAI + 1):
            ev[i] = 0
            for j in range(0, CLASS_A_J_MAX):
                for ss in range(0, 16):
                    pd[i][j][ss] = r[i][j][ss] * pp[ss]     # 確率分布
                    ev[i] = ev[i] + pd[i][j][ss] * v_a[j]   # 期待値

        # 属性値を期待値分増加
        for i in range(I_KIN, I_TAI + 1):
            att[i] = att[i] + ev[i] / fast_mode_rasio
            if att[i] > att_max[i]:
                att[i] = att_max[i]
            
            # 期待値・属性値のプロファイルを保存
            ev_pro[i].append(ev[i])
            att_pro[i].append(att[i])
            att_rate_pro[i].append(att[i] / att_max[i])

########################################
# S級育成処理
def tr_class_S(tr_num, tr_num_ini, att_ini, att_max, fast_mode_rasio):

    ev = [0] * 4    # 期待値
    att = copy.copy(att_ini)   # 属性値

    progress_bar = st.progress(0)   # progress_bar設置    
    
    # 期待値計算
    tr_num = round(tr_num * fast_mode_rasio)
    for tr_cnt in range(1, tr_num + 1):
        num_pro.append(tr_cnt / fast_mode_rasio + tr_num_ini)

        # progress_bar
        pro = (tr_cnt * 100) // tr_num
        if pro % 5 == 0:
            progress_bar.progress(pro)

        # 属性値を期待値分増加
        for i in range(I_KIN, I_TAI + 1):
            ev[i] = 35  # S級期待値は35固定
            att[i] = att[i] + ev[i] / fast_mode_rasio
            if att[i] > att_max[i]:
                att[i] = att_max[i]
            
            # 期待値・属性値のプロファイルを保存
            ev_pro[i].append(ev[i])
            att_pro[i].append(att[i])
            att_rate_pro[i].append(att[i] / att_max[i])

########################################
# メイン処理
def main():
    tr_class = [0] * STEP_MAX                                                   # 育成級
    tr_num = [0] * STEP_MAX                                                     # 育成回数
    att_ini = [[0 for i in range(STATUS_NUM)] for j in range(STEP_MAX)]         # ステップ毎の属性値初期値
    att_max = [0] * STATUS_NUM                                                  # 属性値上限値
    att_rate_ini = [0] * STATUS_NUM                                             # 育成割合初期値
    tr_w = [[0 for i in range(STATUS_NUM)] for j in range(STEP_MAX)]            # ステップ毎の重み
    tr_bias = [0] * STEP_MAX                                                    # ステップ毎のバイアス
    
    # 入力画面
    st.header('放置少女　育成シミュレータ')
    st.caption('Ver. ' + SW_VERSION)
    st.write("使い方： [放置少女研究所](https://idleheroine-research.com/tools/cultivatecalc)")
    st.markdown('***')
    weight_check = st.checkbox('全STEP重み = STEP 1の重み')
    bias_check = st.checkbox('バイアス使用')
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        att_ini[0][I_KIN] = st.number_input(label='筋力 初期値', value=5015, min_value=0)
    with col2:
        att_ini[0][I_BIN] = st.number_input(label='敏捷 初期値', value=4515, min_value=0)
    with col3:
        att_ini[0][I_CHI] = st.number_input(label='知力 初期値', value=4515, min_value=0)
    with col4:
        att_ini[0][I_TAI] = st.number_input(label='体力 初期値', value=5115, min_value=0)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        att_max[I_KIN] = st.number_input(label='筋力 上限値', value=267360, min_value=0)
    with col2:
        att_max[I_BIN] = st.number_input(label='敏捷 上限値', value=251424, min_value=0)
    with col3:
        att_max[I_CHI] = st.number_input(label='知力 上限値', value=248328, min_value=0)
    with col4:
        att_max[I_TAI] = st.number_input(label='体力 上限値', value=259152, min_value=0)

    st.subheader('STEP 1')
    tr_step_num = 1
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        tr_class[0] = st.selectbox('育成級', ['C級', 'B級', 'A級', 'S級'], index=0)
    with col2:
        tr_num[0] = st.number_input(label='育成回数', value=15000, min_value=1, max_value=100000)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        tr_w[0][I_KIN] = st.number_input(label='筋力 重み', value=1.0, )
    with col2:
        tr_w[0][I_BIN] = st.number_input(label='敏捷 重み', value=1.0, )
    with col3:
        tr_w[0][I_CHI] = st.number_input(label='知力 重み', value=1.0, )
    with col4:
        tr_w[0][I_TAI] = st.number_input(label='体力 重み', value=1.0, )
    
    if bias_check:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            tr_bias[0] = st.number_input(label='バイアス', value=0.0, )
    
    st.markdown('***')
    
    if st.checkbox('STEP 2入力'):
        st.subheader('STEP 2')
        tr_step_num = 2

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            tr_class[1] = st.selectbox('育成級2', ['C級', 'B級', 'A級', 'S級'], index=1)
        with col2:
            tr_num[1] = st.number_input(label='育成回数2', value=8000, min_value=1, max_value=100000)
        
        if not weight_check:                
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                tr_w[1][I_KIN] = st.number_input(label='筋力 重み2', value=1.0, )
            with col2:
                tr_w[1][I_BIN] = st.number_input(label='敏捷 重み2', value=1.0, )
            with col3:
                tr_w[1][I_CHI] = st.number_input(label='知力 重み2', value=1.0, )
            with col4:
                tr_w[1][I_TAI] = st.number_input(label='体力 重み2', value=1.0, )
        
        if bias_check:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                tr_bias[1] = st.number_input(label='バイアス2', value=0.0, )
        
        st.markdown('***')

        if st.checkbox('STEP 3入力'):        
            st.subheader('STEP 3')
            tr_step_num = 3

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                tr_class[2] = st.selectbox('育成級3', ['C級', 'B級', 'A級', 'S級'], index=2)
            with col2:
                tr_num[2] = st.number_input(label='育成回数3', value=3000, min_value=1, max_value=100000)
            
            if not weight_check:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    tr_w[2][I_KIN] = st.number_input(label='筋力 重み3', value=1.0, )
                with col2:
                    tr_w[2][I_BIN] = st.number_input(label='敏捷 重み3', value=1.0, )
                with col3:
                    tr_w[2][I_CHI] = st.number_input(label='知力 重み3', value=1.0, )
                with col4:
                    tr_w[2][I_TAI] = st.number_input(label='体力 重み3', value=1.0, )
            
            if bias_check:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    tr_bias[2] = st.number_input(label='バイアス3', value=0.0, )
                    
            st.markdown('***')

            if st.checkbox('STEP 4入力'):        
                st.subheader('STEP 4')
                tr_step_num = 4

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    tr_class[3] = st.selectbox('育成級4', ['C級', 'B級', 'A級', 'S級'], index=3)
                with col2:
                    tr_num[3] = st.number_input(label='育成回数4', value=1000, min_value=1, max_value=100000)
                
                if not weight_check:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        tr_w[3][I_KIN] = st.number_input(label='筋力 重み4', value=1.0, )
                    with col2:
                        tr_w[3][I_BIN] = st.number_input(label='敏捷 重み4', value=1.0, )
                    with col3:
                        tr_w[3][I_CHI] = st.number_input(label='知力 重み4', value=1.0, )
                    with col4:
                        tr_w[3][I_TAI] = st.number_input(label='体力 重み4', value=1.0, )
                
                if bias_check:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        tr_bias[3] = st.number_input(label='バイアス4', value=0.0, )
                        
                st.markdown('***')                
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        GRAPH_PLOT = st.selectbox('グラフ描画', ['ON', 'OFF'])
    with col2:
        FAST_MODE = st.selectbox('高速モード', ['ON', 'OFF'])
 
    # 処理実行    
    if st.button('実行'):
        
        # 定数配列初期化
        init01()

        for i  in range(STATUS_NUM):
            att_rate_ini[i] = att_ini[0][i]  / att_max[i]
        
        if weight_check:    # 全STEP重み同じ
            for i in range(1, tr_step_num):
                for j in range(STATUS_NUM):
                    tr_w[i][j] = tr_w[0][j]

        if not bias_check:
            for i in range(tr_step_num):
                tr_bias[i] = 0
        
        # グラフ用DataFrame
        df1 = pd.DataFrame([], columns=['筋力','敏捷','知力','体力','筋力増分','敏捷増分','知力増分','体力増分'], index=['初期値'])
        df2 = pd.DataFrame([], columns=['筋力','敏捷','知力','体力','筋力増分','敏捷増分','知力増分','体力増分'], index=['初期値'])

        df1.loc['初期値'] = ['{:>6.0f}'.format(att_ini[0][I_KIN]), '{:>6.0f}'.format(att_ini[0][I_BIN]), '{:>6.0f}'.format(att_ini[0][I_CHI]), '{:>6.0f}'.format(att_ini[0][I_TAI]), '-', '-', '-', '-']
        df2.loc['初期値'] = ['{:>6.1%}'.format(att_rate_ini[I_KIN]), '{:>6.1%}'.format(att_rate_ini[I_BIN]), '{:>6.1%}'.format(att_rate_ini[I_CHI]), '{:>6.1%}'.format(att_rate_ini[I_TAI]), '-', '-', '-', '-']
            
        for tr_step in range(tr_step_num):
            
            # 高速計算モード設定
            if (FAST_MODE == "ON") and (tr_num[tr_step] > FAST_MODE_RESO):
                fast_mode_rasio = FAST_MODE_RESO / tr_num[tr_step]
            else:
                fast_mode_rasio = 1.0
            
            # 前ステップ引継ぎ
            if tr_step >= 1:
                tr_num_ini = tr_num_ini + tr_num[tr_step - 1]
                for i in range(STATUS_NUM):
                    att_ini[tr_step][i] = att_pro[i][-1]
            else:
                tr_num_ini = 0
            
            # 育成級別処理
            if tr_class[tr_step] == "C級":
                tr_class_C(tr_num[tr_step], tr_num_ini, tr_w[tr_step], tr_bias[tr_step], att_ini[tr_step], att_max, fast_mode_rasio)
            elif tr_class[tr_step] == "B級":
                tr_class_B(tr_num[tr_step], tr_num_ini, tr_w[tr_step], tr_bias[tr_step], att_ini[tr_step], att_max, fast_mode_rasio)
            elif tr_class[tr_step] == "A級":
                tr_class_A(tr_num[tr_step], tr_num_ini, tr_w[tr_step], tr_bias[tr_step], att_ini[tr_step], att_max, fast_mode_rasio)
            elif tr_class[tr_step] == "S級":
                tr_class_S(tr_num[tr_step], tr_num_ini, att_ini[tr_step], att_max, fast_mode_rasio)
            else:
                dummy() #何もしない
            
            df1.loc['STEP '+str(tr_step + 1)] = ['{:>6.0f}'.format(att_pro[I_KIN][-1]), 
                                                '{:>6.0f}'.format(att_pro[I_BIN][-1]), 
                                                '{:>6.0f}'.format(att_pro[I_CHI][-1]),
                                                '{:>6.0f}'.format(att_pro[I_TAI][-1]), 
                                                '+{:>6.0f}'.format(att_pro[I_KIN][-1] - att_ini[tr_step][I_KIN]),
                                                '+{:>6.0f}'.format(att_pro[I_BIN][-1] - att_ini[tr_step][I_BIN]),
                                                '+{:>6.0f}'.format(att_pro[I_CHI][-1] - att_ini[tr_step][I_CHI]),
                                                '+{:>6.0f}'.format(att_pro[I_TAI][-1] - att_ini[tr_step][I_TAI])]
            
            df2.loc['STEP '+str(tr_step + 1)] = ['{:>6.1%}'.format(att_rate_pro[I_KIN][-1]), 
                                                '{:>6.1%}'.format(att_rate_pro[I_BIN][-1]), 
                                                '{:>6.1%}'.format(att_rate_pro[I_CHI][-1]), 
                                                '{:>6.1%}'.format(att_rate_pro[I_TAI][-1]), 
                                                '+{:>6.1%}'.format(att_rate_pro[I_KIN][-1] - att_rate_ini[I_KIN]), 
                                                '+{:>6.1%}'.format(att_rate_pro[I_BIN][-1] - att_rate_ini[I_BIN]), 
                                                '+{:>6.1%}'.format(att_rate_pro[I_CHI][-1] - att_rate_ini[I_CHI]), 
                                                '+{:>6.1%}'.format(att_rate_pro[I_TAI][-1] - att_rate_ini[I_TAI])]
            # 次ステップ引継ぎ
            for i  in range(STATUS_NUM):
                att_rate_ini[i] = att_rate_pro[i][-1]

        if tr_step_num >= 2:
            df1.loc['最終値'] = ['{:>6.0f}'.format(att_pro[I_KIN][-1]), 
                                '{:>6.0f}'.format(att_pro[I_BIN][-1]),
                                '{:>6.0f}'.format(att_pro[I_CHI][-1]), 
                                '{:>6.0f}'.format(att_pro[I_TAI][-1]), 
                                '+{:>6.0f}'.format(att_pro[I_KIN][-1] - att_ini[0][I_KIN]),
                                '+{:>6.0f}'.format(att_pro[I_BIN][-1] - att_ini[0][I_BIN]),
                                '+{:>6.0f}'.format(att_pro[I_CHI][-1] - att_ini[0][I_CHI]),
                                '+{:>6.0f}'.format(att_pro[I_TAI][-1] - att_ini[0][I_TAI])]
            df2.loc['最終値'] = ['{:>6.1%}'.format(att_rate_pro[I_KIN][-1]),
                                 '{:>6.1%}'.format(att_rate_pro[I_BIN][-1]),
                                 '{:>6.1%}'.format(att_rate_pro[I_CHI][-1]),
                                 '{:>6.1%}'.format(att_rate_pro[I_TAI][-1]),
                                 '+{:>6.1%}'.format(att_rate_pro[I_KIN][-1] - att_ini[0][I_KIN] / att_max[I_KIN]), 
                                 '+{:>6.1%}'.format(att_rate_pro[I_BIN][-1] - att_ini[0][I_BIN] / att_max[I_BIN]), 
                                 '+{:>6.1%}'.format(att_rate_pro[I_CHI][-1] - att_ini[0][I_CHI] / att_max[I_CHI]), 
                                 '+{:>6.1%}'.format(att_rate_pro[I_TAI][-1] - att_ini[0][I_TAI] / att_max[I_TAI])]
        
        complete_check = True
        
        # 結果表示(表)
        st.info("育成結果(絶対値)")
        st.dataframe(df1)
        
        st.info("育成結果(育成割合[%])")
        st.dataframe(df2)
        
        # グラフ表示
        if GRAPH_PLOT == "ON":
            plt.style.use('default')
            sns.set()
            sns.set_style('whitegrid')
            sns.set_palette('Set1')
                
            x = np.array(num_pro)

            # 属性値(絶対値)のグラフ
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

            ax.plot(x, np.array(att_pro[I_KIN]), label="KIN")
            ax.plot(x, np.array(att_pro[I_BIN]), label="BIN")
            ax.plot(x, np.array(att_pro[I_CHI]), label="CHI")
            ax.plot(x, np.array(att_pro[I_TAI]), label="TAI")

            ax.legend()
            ax.set_xlabel('NUM')
            ax.set_ylabel("Status (Abs.)")

            st.info('グラフ：育成回数－ステータス値')
            st.pyplot(fig)

            # 属性値(育成割合[%])のグラフ
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

            ax.plot(x, np.array(att_rate_pro[I_KIN]), label="KIN")
            ax.plot(x, np.array(att_rate_pro[I_BIN]), label="BIN")
            ax.plot(x, np.array(att_rate_pro[I_CHI]), label="CHI")
            ax.plot(x, np.array(att_rate_pro[I_TAI]), label="TAI")

            ax.legend()
            ax.set_xlabel("NUM")
            ax.set_ylabel("Status (%)")
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

            st.info('グラフ：育成回数－育成割合[%]')
            st.pyplot(fig)

            # 属性値(育成割合[%])のグラフ
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

            ax.plot(x, np.array(ev_pro[I_KIN]), label="KIN")
            ax.plot(x, np.array(ev_pro[I_BIN]), label="BIN")
            ax.plot(x, np.array(ev_pro[I_CHI]), label="CHI")
            ax.plot(x, np.array(ev_pro[I_TAI]), label="TAI")

            ax.legend()
            ax.set_xlabel("NUM")
            ax.set_ylabel("Expected value")

            st.info('グラフ：育成回数－1回当たりの期待値')
            st.pyplot(fig)

########################################
# メイン実行
if __name__ == '__main__':
    main()