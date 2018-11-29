def printStatics(data, train, test, dic, total_n, total_p):
    print '\n########### Statics ###########'
    print '>Tweets:', str(len(data))
    print '  Positive:', total_p, ' Negative:', total_n, '   Ratio Positive/Negative :', total_p / float(total_n)

    print '>TRAIN:', str(len(train))
    total_p_t, total_n_t = 0, 0
    for t in train:
        if t[3] == 0:
            total_n_t += 1
        else:
            total_p_t += 1
    print '  Positive:', total_p_t, ' Negative:', total_n_t, '   Ratio Positive/Negative :', total_p_t / float(
        total_n_t)
    print '>TEST:', str(len(test))
    total_p_t, total_n_t = 0, 0
    for t in test:
        if t[3] == 0:
            total_n_t += 1
        else:
            total_p_t += 1
    print '  Positive:', total_p_t, ' Negative:', total_n_t, '   Ratio Positive/Negative :', total_p_t / float(
        total_n_t)
    print '\n\n'
