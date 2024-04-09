from data_clean import*
# data/1_0.csv




path='data/5_1.csv'
for k in range(6):
    num='no'+str(k+1)
    print(num)
    dianya,maichong=solve_data(path,num)
    n = 3
    index, data, datarec, bizhi, mintime, maxtime = xiaobo(maichong, dianya)
    outlier_x, overier_x, outlier_n, lsttype = lianghua(datarec, n)
    # draw(index, data, datarec, bizhi, mintime, maxtime)
    my_dict=solve_dict(lsttype,outlier_n)
    alst1,weizhi=get_weizhi(outlier_x,lsttype,overier_x,index)
    path_s="clean_data/s_"+str(k+1)+".csv"
    write_data(weizhi,lsttype,path_s)
