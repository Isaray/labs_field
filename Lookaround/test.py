import numpy as np  
  
# 读取npy文件  
res_0 = np.load('out/resnet50_cifar10/TRAIN_Lookaround/five-10-200/test_acc.npy')  
res_1 = np.load('out/resnet50_cifar10/TRAIN_Lookaround/five-10-200/test_acc.npy')   
res_r_0 = np.load('out/resnet50_cifar10/TRAIN_Lookaround/five-rand-1/test_acc.npy')  
res_r_1 = np.load('out/resnet50_cifar10/TRAIN_Lookaround/five-rand-2/test_acc.npy')  
res_old = np.load('out/resnet50_cifar10/TRAIN_Lookaround-0/test_acc.npy')  
res_100=np.load('out/resnet50_cifar10_1/TRAIN_Lookaround/five-rand-3/test_acc.npy')  
res=(res_0[:,0]+res_1[:,0])/2
res_r=(res_r_0[:,0]+res_r_1[:,0])/2
res_old=res_old[:,0]
# 打印读取的数据  
import matplotlib.pyplot as plt    
x=[i for i in range(1,201)]
plt.plot(x, res_old)  
plt.plot(x, res)  
plt.plot(x, res_r)  
last_index = len(x) - 1  
last_y = res[last_index]  
# plt.annotate(f'y_old_3 = {last_y}', xy=(last_index, last_y), xytext=(0, 5), textcoords='offset points') 
# plt.annotate(f'y_old_5 = {last_y}', xy=(last_index, last_y), xytext=(0, 10), textcoords='offset points') 
# plt.annotate(f'y_random = {last_y}', xy=(last_index, last_y), xytext=(0, 10), textcoords='offset points') 
plt.legend(["Lookaround_old_3","Lookaround_old_5","Lookaround_random_3in5"])
plt.title('Top1 acc with epoch')  
plt.xlabel('epoch')  
plt.ylabel('Top 1 acc')   
plt.savefig("test.png")

print("max old:",max(res_old),",max old 5:",max(res),",max random:",max(res_r))
