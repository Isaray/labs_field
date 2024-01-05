import numpy as np
#顺序无所谓，不能组合完全一样

def myfun(mylist,tgt):
    if len(mylist)==0 :
        return []
    if len(mylist)==1:
        if mylist[0]==tgt:
            return [[mylist[0]]]
    total_res=[]
    for i in range(len(mylist)):
        t=tgt-mylist[i]       
        if t==0:
            total_res.append([mylist[i]])
        else:
            add_list=myfun(mylist[i+1:],t)
            #print(mylist[i],t,add_list)
            for j in range(len(add_list)):
                total_res.append([mylist[i]]+add_list[j])
    return total_res
s=[2,3,5,7,8,10,11,13]
# s=[3,7,13]
tgt=16
res=myfun(s,tgt) 
print(res)
