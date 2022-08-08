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

def hs_trainning_sim_main():
    tr_num = 10000
    x = 0.055
    y = 0
    for tr_cnt in range(1, tr_num + 1):
        y = y + cal_p_c(x)
        x = x + 0.00001
    
    print(str(y))

###########################
# メイン呼び出し
hs_trainning_sim_main()