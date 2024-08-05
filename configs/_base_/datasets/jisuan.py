def sigmoid_bianhua(x, p1):
    p = (np.log((1 / 0.99) - 1) + 6) / (p1)
    print("p",p)
    x = p * (x)
    x = x - 6
    return 1 / (1 + np.exp(x))  # sigmoid函数

sigmoid_bianhua(32,1.5p)