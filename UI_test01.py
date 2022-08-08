
##################################################
# 放置少女　育成シミュレータ(仮) オンライン版
#  Ver 1.00 2021/7/24
###################################################

import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#import matplotlib
#import matplotlib.pyplot as plt
#import seaborn as sns
#import numpy as np

########################################
# 定数
CLASS_C_J_MAX = 26  # 上昇値のパターン数
CLASS_B_J_MAX = 24
CLASS_A_J_MAX = 22

FAST_MODE_RESO = 1000   # 高速計算時の分解能

i_kin = 0   # インデックス用
i_bin = 1
i_chi = 2
i_tai = 3

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
    
    pp[0] = (1 - p[i_kin]) * (1 - p[i_bin]) * (1 - p[i_chi]) * (1 - p[i_tai])   #----
    pp[1] = (1 - p[i_kin]) * (1 - p[i_bin]) * (1 - p[i_chi]) * p[i_tai]         #---+
    pp[2] = (1 - p[i_kin]) * (1 - p[i_bin]) * p[i_chi] * (1 - p[i_tai])         #--+-
    pp[3] = (1 - p[i_kin]) * (1 - p[i_bin]) * p[i_chi] * p[i_tai]               #--++
    pp[4] = (1 - p[i_kin]) * p[i_bin] * (1 - p[i_chi]) * (1 - p[i_tai])         #-+--
    pp[5] = (1 - p[i_kin]) * p[i_bin] * (1 - p[i_chi]) * p[i_tai]               #-+-+
    pp[6] = (1 - p[i_kin]) * p[i_bin] * p[i_chi] * (1 - p[i_tai])               #-++-
    pp[7] = (1 - p[i_kin]) * p[i_bin] * p[i_chi] * p[i_tai]                     #-+++
    pp[8] = p[i_kin] * (1 - p[i_bin]) * (1 - p[i_chi]) * (1 - p[i_tai])         #+---
    pp[9] = p[i_kin] * (1 - p[i_bin]) * (1 - p[i_chi]) * p[i_tai]               #+--+
    pp[10] = p[i_kin] * (1 - p[i_bin]) * p[i_chi] * (1 - p[i_tai])              #+-+-
    pp[11] = p[i_kin] * (1 - p[i_bin]) * p[i_chi] * p[i_tai]                    #+-++
    pp[12] = p[i_kin] * p[i_bin] * (1 - p[i_chi]) * (1 - p[i_tai])              #++--
    pp[13] = p[i_kin] * p[i_bin] * (1 - p[i_chi]) * p[i_tai]                    #++-+
    pp[14] = p[i_kin] * p[i_bin] * p[i_chi] * (1 - p[i_tai])                    #+++-
    pp[15] = p[i_kin] * p[i_bin] * p[i_chi] * p[i_tai]                          #++++

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
def tr_class_C(tr_num, tr_w, tr_bias, att_ini, att_max, fast_mode_rasio):
    
    att = att_ini   # 属性値
    
    a = [0] * 4     # 育成出現値
    s = [0] * 4     # 育成出現値符号
    p = [0] * 4     # 育成プラス出現確率
    pp = [0] * 16   # ケース別確率
    pd = [[[0] * 16 for i in range(CLASS_C_J_MAX)] for j in range(4)]  # 確率分布

    ev = [0] * 4    # 期待値

    n = [[[0] * 16 for i in range(CLASS_C_J_MAX)] for j in range(4)]   # 育成結果保存判定ケース数
    r = [[[0] * 16 for i in range(CLASS_C_J_MAX)] for j in range(4)]   # 育成結果保存判定ケース割合
    
    progress_bar = st.progress(0)
    
    # 育成結果保存判定ケース数のカウント処理
    for j_k in range(0, CLASS_C_J_MAX): # 筋力 -12～-1, +1～+14のループ
        a[i_kin] = v_c[j_k]
        if a[i_kin] > 0:
            s[i_kin] = 8
        else:
            s[i_kin] = 0
        
        for j_b in range(0, CLASS_C_J_MAX): # 敏捷 -12～-1, +1～+14のループ
            a[i_bin] = v_c[j_b]
            if a[i_bin] > 0:
                s[i_bin] = 4
            else:
                s[i_bin] = 0
            
            for j_c in range(0, CLASS_C_J_MAX): # 知力 -12～-1, +1～+14のループ
                a[i_chi] = v_c[j_c]
                if a[i_chi] > 0:
                    s[i_chi] = 2
                else:
                    s[i_chi] = 0
                
                for j_t in range(0, CLASS_C_J_MAX): # 体力 -12～-1, +1～+14のループ
                    a[i_tai] = v_c[j_t]
                    if a[i_tai] > 0:
                        s[i_tai] = 1
                    else:
                        s[i_tai] = 0
                    
                    # 育成結果の保存判定用　評価値
                    e = a[i_kin] * tr_w[i_kin] + a[i_bin] * tr_w[i_bin] + a[i_chi] * tr_w[i_chi] + a[i_tai] * tr_w[i_tai] + tr_bias
                    
                    # ケースNo.
                    ss = s[i_kin] + s[i_bin] + s[i_chi] + s[i_tai]
                    
                    if e > 0:   # 保存判定ケース数
                        n[i_kin][j_k][ss] = n[i_kin][j_k][ss] + 1
                        n[i_bin][j_b][ss] = n[i_bin][j_b][ss] + 1
                        n[i_chi][j_c][ss] = n[i_chi][j_c][ss] + 1
                        n[i_tai][j_t][ss] = n[i_tai][j_t][ss] + 1

    # 育成結果保存判定ケース割合
    for i in range(i_kin, i_tai + 1):
        for j in range(0, CLASS_C_J_MAX):
            for ss in range(0, 16):
                r[i][j][ss] = n[i][j][ss] / B_c[ss]

    # 期待値計算
    tr_num = round(tr_num * fast_mode_rasio)
    for tr_cnt in range(1, tr_num + 1):
        num_pro.append(tr_cnt / fast_mode_rasio)
        
        # progress_bar
        pro = (tr_cnt * 100) // tr_num
        if pro % 5 == 0:
            progress_bar.progress(pro)

        # 育成プラス出現率
        p[i_kin] = cal_p_c(att[i_kin] / att_max[i_kin])
        p[i_bin] = cal_p_c(att[i_bin] / att_max[i_bin])
        p[i_chi] = cal_p_c(att[i_chi] / att_max[i_chi])
        p[i_tai] = cal_p_c(att[i_tai] / att_max[i_tai])
        
        # ケース別確率
        pp = cal_pp(p)
        
        for i in range(i_kin, i_tai + 1):
            ev[i] = 0
            for j in range(0, CLASS_C_J_MAX):
                for ss in range(0, 16):
                    pd[i][j][ss] = r[i][j][ss] * pp[ss]     # 確率分布
                    ev[i] = ev[i] + pd[i][j][ss] * v_c[j]   # 期待値

        # 属性値を期待値分増加
        for i in range(i_kin, i_tai + 1):
            att[i] = att[i] + ev[i] / fast_mode_rasio
            if att[i] > att_max[i]:
                att[i] = att_max[i]
            
            # 期待値・属性値のプロファイルを保存
            ev_pro[i].append(ev[i])
            att_pro[i].append(att[i])
            att_rate_pro[i].append(att[i] / att_max[i])

########################################
# B級育成処理
def tr_class_B(tr_num, tr_w, tr_bias, att_ini, att_max, fast_mode_rasio):
    
    att = att_ini   # 属性値
    
    a = [0] * 4     # 育成出現値
    s = [0] * 4     # 育成出現値符号
    p = [0] * 4     # 育成プラス出現確率
    pp = [0] * 16   # ケース別確率
    pd = [[[0] * 16 for i in range(CLASS_B_J_MAX)] for j in range(4)]  # 確率分布

    ev = [0] * 4    # 期待値

    n = [[[0] * 16 for i in range(CLASS_B_J_MAX)] for j in range(4)]   # 育成結果保存判定ケース数
    r = [[[0] * 16 for i in range(CLASS_B_J_MAX)] for j in range(4)]   # 育成結果保存判定ケース割合

    progress_bar = st.progress(0)
    
    # 育成結果保存判定ケース数のカウント処理
    for j_k in range(0, CLASS_B_J_MAX): # 筋力 -12～-1, +6～+17のループ
        a[i_kin] = v_b[j_k]
        if a[i_kin] > 0:
            s[i_kin] = 8
        else:
            s[i_kin] = 0
        
        for j_b in range(0, CLASS_B_J_MAX): # 敏捷 -12～-1, +6～+17のループ
            a[i_bin] = v_b[j_b]
            if a[i_bin] > 0:
                s[i_bin] = 4
            else:
                s[i_bin] = 0
            
            for j_c in range(0, CLASS_B_J_MAX): # 知力 -12～-1, +6～+17のループ
                a[i_chi] = v_b[j_c]
                if a[i_chi] > 0:
                    s[i_chi] = 2
                else:
                    s[i_chi] = 0
                
                for j_t in range(0, CLASS_B_J_MAX): # 体力 -12～-1, +6～+17のループ
                    a[i_tai] = v_b[j_t]
                    if a[i_tai] > 0:
                        s[i_tai] = 1
                    else:
                        s[i_tai] = 0
                    
                    # 育成結果の保存判定用　評価値
                    e = a[i_kin] * tr_w[i_kin] + a[i_bin] * tr_w[i_bin] + a[i_chi] * tr_w[i_chi] + a[i_tai] * tr_w[i_tai] + tr_bias
                    
                    # ケースNo.
                    ss = s[i_kin] + s[i_bin] + s[i_chi] + s[i_tai]
                    
                    if e > 0:   # 保存判定ケース数
                        n[i_kin][j_k][ss] = n[i_kin][j_k][ss] + 1
                        n[i_bin][j_b][ss] = n[i_bin][j_b][ss] + 1
                        n[i_chi][j_c][ss] = n[i_chi][j_c][ss] + 1
                        n[i_tai][j_t][ss] = n[i_tai][j_t][ss] + 1

    # 育成結果保存判定ケース割合
    for i in range(i_kin, i_tai + 1):
        for j in range(0, CLASS_B_J_MAX):
            for ss in range(0, 16):
                r[i][j][ss] = n[i][j][ss] / B_b[ss]

    # 期待値計算
    tr_num = round(tr_num * fast_mode_rasio)
    for tr_cnt in range(1, tr_num + 1):
        num_pro.append(tr_cnt / fast_mode_rasio)

        # progress_bar
        pro = (tr_cnt * 100) // tr_num
        if pro % 5 == 0:
            progress_bar.progress(pro)

        # 育成プラス出現率
        p[i_kin] = cal_p_b(att[i_kin] / att_max[i_kin])
        p[i_bin] = cal_p_b(att[i_bin] / att_max[i_bin])
        p[i_chi] = cal_p_b(att[i_chi] / att_max[i_chi])
        p[i_tai] = cal_p_b(att[i_tai] / att_max[i_tai])
        
        # ケース別確率
        pp = cal_pp(p)
        
        for i in range(i_kin, i_tai + 1):
            ev[i] = 0
            for j in range(0, CLASS_B_J_MAX):
                for ss in range(0, 16):
                    pd[i][j][ss] = r[i][j][ss] * pp[ss]     # 確率分布
                    ev[i] = ev[i] + pd[i][j][ss] * v_b[j]   # 期待値

        # 属性値を期待値分増加
        for i in range(i_kin, i_tai + 1):
            att[i] = att[i] + ev[i] / fast_mode_rasio
            if att[i] > att_max[i]:
                att[i] = att_max[i]
            
            # 期待値・属性値のプロファイルを保存
            ev_pro[i].append(ev[i])
            att_pro[i].append(att[i])
            att_rate_pro[i].append(att[i] / att_max[i])

########################################
# A級育成処理
def tr_class_A(tr_num, tr_w, tr_bias, att_ini, att_max, fast_mode_rasio):
    
    att = att_ini   # 属性値
    
    a = [0] * 4     # 育成出現値
    s = [0] * 4     # 育成出現値符号
    p = [0] * 4     # 育成プラス出現確率
    pp = [0] * 16   # ケース別確率
    pd = [[[0] * 16 for i in range(CLASS_A_J_MAX)] for j in range(4)]  # 確率分布

    ev = [0] * 4    # 期待値

    n = [[[0] * 16 for i in range(CLASS_A_J_MAX)] for j in range(4)]   # 育成結果保存判定ケース数
    r = [[[0] * 16 for i in range(CLASS_A_J_MAX)] for j in range(4)]   # 育成結果保存判定ケース割合

    progress_bar = st.progress(0)
        
    # 育成結果保存判定ケース数のカウント処理
    for j_k in range(0, CLASS_A_J_MAX): # 筋力 -12～-1, +16～+25のループ
        a[i_kin] = v_a[j_k]
        if a[i_kin] > 0:
            s[i_kin] = 8
        else:
            s[i_kin] = 0
        
        for j_b in range(0, CLASS_A_J_MAX): # 敏捷 -12～-1, +16～+25のループ
            a[i_bin] = v_a[j_b]
            if a[i_bin] > 0:
                s[i_bin] = 4
            else:
                s[i_bin] = 0
            
            for j_c in range(0, CLASS_A_J_MAX): # 知力 -12～-1, +16～+25のループ
                a[i_chi] = v_a[j_c]
                if a[i_chi] > 0:
                    s[i_chi] = 2
                else:
                    s[i_chi] = 0
                
                for j_t in range(0, CLASS_A_J_MAX): # 体力 -12～-1, +16～+25のループ
                    a[i_tai] = v_a[j_t]
                    if a[i_tai] > 0:
                        s[i_tai] = 1
                    else:
                        s[i_tai] = 0
                    
                    # 育成結果の保存判定用　評価値
                    e = a[i_kin] * tr_w[i_kin] + a[i_bin] * tr_w[i_bin] + a[i_chi] * tr_w[i_chi] + a[i_tai] * tr_w[i_tai] + tr_bias
                    
                    # ケースNo.
                    ss = s[i_kin] + s[i_bin] + s[i_chi] + s[i_tai]
                    
                    if e > 0:   # 保存判定ケース数
                        n[i_kin][j_k][ss] = n[i_kin][j_k][ss] + 1
                        n[i_bin][j_b][ss] = n[i_bin][j_b][ss] + 1
                        n[i_chi][j_c][ss] = n[i_chi][j_c][ss] + 1
                        n[i_tai][j_t][ss] = n[i_tai][j_t][ss] + 1

    # 育成結果保存判定ケース割合
    for i in range(i_kin, i_tai + 1):
        for j in range(0, CLASS_A_J_MAX):
            for ss in range(0, 16):
                r[i][j][ss] = n[i][j][ss] / B_a[ss]

    # 期待値計算
    tr_num = round(tr_num * fast_mode_rasio)
    for tr_cnt in range(1, tr_num + 1):
        num_pro.append(tr_cnt / fast_mode_rasio)

        # progress_bar
        pro = (tr_cnt * 100) // tr_num
        if pro % 5 == 0:
            progress_bar.progress(pro)

        # 育成プラス出現率
        p[i_kin] = cal_p_a(att[i_kin] / att_max[i_kin])
        p[i_bin] = cal_p_a(att[i_bin] / att_max[i_bin])
        p[i_chi] = cal_p_a(att[i_chi] / att_max[i_chi])
        p[i_tai] = cal_p_a(att[i_tai] / att_max[i_tai])
        
        # ケース別確率
        pp = cal_pp(p)
        
        for i in range(i_kin, i_tai + 1):
            ev[i] = 0
            for j in range(0, CLASS_A_J_MAX):
                for ss in range(0, 16):
                    pd[i][j][ss] = r[i][j][ss] * pp[ss]     # 確率分布
                    ev[i] = ev[i] + pd[i][j][ss] * v_a[j]   # 期待値

        # 属性値を期待値分増加
        for i in range(i_kin, i_tai + 1):
            att[i] = att[i] + ev[i] / fast_mode_rasio
            if att[i] > att_max[i]:
                att[i] = att_max[i]
            
            # 期待値・属性値のプロファイルを保存
            ev_pro[i].append(ev[i])
            att_pro[i].append(att[i])
            att_rate_pro[i].append(att[i] / att_max[i])

########################################
# S級育成処理
def tr_class_S(tr_num, att_ini, att_max, fast_mode_rasio):

    att = att_ini   # 属性値
    ev = [0] * 4    # 期待値

    progress_bar = st.progress(0)
    
    # 期待値計算
    tr_num = round(tr_num * fast_mode_rasio)
    for tr_cnt in range(1, tr_num + 1):
        num_pro.append(tr_cnt / fast_mode_rasio)

        # progress_bar
        pro = (tr_cnt * 100) // tr_num
        if pro % 5 == 0:
            progress_bar.progress(pro)

        # 属性値を期待値分増加
        for i in range(i_kin, i_tai + 1):
            ev[i] = 35  # S級期待値は35固定
            att[i] = att[i] + ev[i] / fast_mode_rasio
            if att[i] > att_max[i]:
                att[i] = att_max[i]
            
            # 期待値・属性値のプロファイルを保存
            ev_pro[i].append(ev[i])
            att_pro[i].append(att[i])
            att_rate_pro[i].append(att[i] / att_max[i])

###########################
# メイン処理
def main():
    st.title('放置少女　育成シミュレータ')
    st.header('Ver 1.00')

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        CLASS = st.selectbox('育成級', ['C級', 'B級', 'A級', 'S級'])
    with col2:
        NUM = st.number_input(label='育成回数', value=7000, min_value=1, max_value=100000)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        KIN_ini = st.number_input(label='筋力 初期値', value=4291, )
    with col2:
        BIN_ini = st.number_input(label='敏捷 初期値', value=3837, )
    with col3:
        CHI_ini = st.number_input(label='知力 初期値', value=3751, )
    with col4:
        TAI_ini = st.number_input(label='体力 初期値', value=4282, )
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        KIN_max = st.number_input(label='筋力 上限値', value=192200, )
    with col2:
        BIN_max = st.number_input(label='敏捷 上限値', value=178920, )
    with col3:
        CHI_max = st.number_input(label='知力 上限値', value=176340, )
    with col4:
        TAI_max = st.number_input(label='体力 上限値', value=185360, )
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        W_kin = st.number_input(label='筋力 重み', value=1.0, )
    with col2:
        W_bin = st.number_input(label='敏捷 重み', value=1.0, )
    with col3:
        W_chi = st.number_input(label='知力 重み', value=1.0, )
    with col4:
        W_tai = st.number_input(label='体力 重み', value=1.0, )
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        BIAS = st.number_input(label='バイアス', value=0.0, )
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        GRAPH_PLOT = st.selectbox('グラフ描画', ['ON', 'OFF'])
    with col2:
        FAST_MODE = st.selectbox('高速モード', ['ON', 'OFF'])
 
    if st.button('実行'):
        att_ini = [KIN_ini, BIN_ini, CHI_ini, TAI_ini]
        att_max = [KIN_max, BIN_max, CHI_max, TAI_max]
        tr_w = [W_kin, W_bin, W_chi, W_tai]

        # 高速計算モード設定
        if (FAST_MODE == "ON") and (NUM > FAST_MODE_RESO):
            fast_mode_rasio = FAST_MODE_RESO / NUM
        else:
            fast_mode_rasio = 1.0

        # 定数配列初期化
        init01()
        
        # 育成級別処理
        if CLASS == "C級":
            tr_class_C(NUM, tr_w, BIAS, att_ini, att_max, fast_mode_rasio)
        elif CLASS == "B級":
            tr_class_B(NUM, tr_w, BIAS, att_ini, att_max, fast_mode_rasio)
        elif CLASS == "A級":
            tr_class_A(NUM, tr_w, BIAS, att_ini, att_max, fast_mode_rasio)
        elif CLASS == "S級":
            tr_class_S(NUM, att_ini, att_max, fast_mode_rasio)
        else:
            dummy() #何もしない
    
        # 結果表示
        st.info("育成結果(絶対値)")
        st.write("筋力:{:>6.0f}".format(KIN_ini), "  ->  {:>6.0f}".format(att_pro[0][-1]), "({:>+6.0f})".format(att_pro[0][-1] - KIN_ini))
        st.write("敏捷:{:>6.0f}".format(BIN_ini), "  ->  {:>6.0f}".format(att_pro[1][-1]), "({:>+6.0f})".format(att_pro[1][-1] - BIN_ini))
        st.write("知力:{:>6.0f}".format(CHI_ini), "  ->  {:>6.0f}".format(att_pro[2][-1]), "({:>+6.0f})".format(att_pro[2][-1] - CHI_ini))
        st.write("体力:{:>6.0f}".format(TAI_ini), "  ->  {:>6.0f}".format(att_pro[3][-1]), "({:>+6.0f})".format(att_pro[3][-1] - TAI_ini))
        
        st.info("育成結果(育成割合[%])")
        st.write("筋力:{:>6.1%}".format(KIN_ini / att_max[0]), "  ->  {:>6.1%}".format(att_rate_pro[0][-1]), "({:>+6.1%})".format((att_pro[0][-1] - KIN_ini) / att_max[0]))
        st.write("敏捷:{:>6.1%}".format(BIN_ini / att_max[1]), "  ->  {:>6.1%}".format(att_rate_pro[1][-1]), "({:>+6.1%})".format((att_pro[1][-1] - BIN_ini) / att_max[1]))
        st.write("知力:{:>6.1%}".format(CHI_ini / att_max[2]), "  ->  {:>6.1%}".format(att_rate_pro[2][-1]), "({:>+6.1%})".format((att_pro[2][-1] - CHI_ini) / att_max[2]))
        st.write("体力:{:>6.1%}".format(TAI_ini / att_max[3]), "  ->  {:>6.1%}".format(att_rate_pro[3][-1]), "({:>+6.1%})".format((att_pro[3][-1] - TAI_ini) / att_max[3]))    

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

            ax.plot(x, np.array(att_pro[0]), label="KIN")
            ax.plot(x, np.array(att_pro[1]), label="BIN")
            ax.plot(x, np.array(att_pro[2]), label="CHI")
            ax.plot(x, np.array(att_pro[3]), label="TAI")

            ax.legend()
            ax.set_xlabel('NUM')
            ax.set_ylabel("Status (Abs.)")

            st.info('グラフ：育成回数－ステータス値')
            st.pyplot(fig)

            # 属性値(育成割合[%])のグラフ
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

            ax.plot(x, np.array(att_rate_pro[0]), label="KIN")
            ax.plot(x, np.array(att_rate_pro[1]), label="BIN")
            ax.plot(x, np.array(att_rate_pro[2]), label="CHI")
            ax.plot(x, np.array(att_rate_pro[3]), label="TAI")

            ax.legend()
            ax.set_xlabel("NUM")
            ax.set_ylabel("Status (%)")
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

            st.info('グラフ：育成回数－育成割合[%]')
            st.pyplot(fig)

            # 属性値(育成割合[%])のグラフ
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

            ax.plot(x, np.array(ev_pro[0]), label="KIN")
            ax.plot(x, np.array(ev_pro[1]), label="BIN")
            ax.plot(x, np.array(ev_pro[2]), label="CHI")
            ax.plot(x, np.array(ev_pro[3]), label="TAI")

            ax.legend()
            ax.set_xlabel("NUM")
            ax.set_ylabel("Expected value")

            st.info('グラフ：育成回数－1回当たりの期待値')
            st.pyplot(fig)

if __name__ == '__main__':
    main()