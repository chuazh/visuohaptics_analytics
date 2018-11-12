import numpy as np
import bca
import matplotlib.pyplot as plt
import pylab


def moving_average(data,periods):
    weights = np.ones(periods)/periods
    return np.convolve(data,weights,mode='same')

def moving_median(data,periods):
    y = np.zeros((len(data),))
    last_entry = data[-1]
    data_aug = np.hstack((data,last_entry*np.ones((periods,))))
    for ctr in range(len(data)):
        temp_array = data_aug[ctr:ctr+periods]
        y[ctr] = np.median(temp_array)

    return y

def plot_progression(data,col,type=0, color= [] , linetype = '-o' ):
    num_trials =  np.arange(1,len(data)+1,1)
    plot_y = []
    label_name = str(data[0][0,1])
    for i in range(0,len(data)):
            plot_y.append(data[i][-1,col])
    if type==1:
        plot_y = moving_average(np.asarray(plot_y),5)
        label_name = label_name + " movAvg"
    elif type==2:
        plot_y = moving_median(np.asarray(plot_y),5)
        label_name = label_name + " movMed"
    else:
        label_name = label_name + " actual"

    if not color:
        #plt.plot(num_trials,plot_y,linetype)
        plt.plot(num_trials,plot_y,linetype,label=label_name)
    else:
        plt.plot(num_trials,plot_y,linetype,c=color,label=label_name)

        plt.xlabel('Number of Trials')
        plt.ylabel('Force (N)')

def plot_progression(data,col,type=0, color= [] , linetype = '-o' ):
    num_trials =  np.arange(1,len(data)+1,1)
    plot_y = []
    label_name = str(data[0][0,1])
    for i in range(0,len(data)):
            plot_y.append(data[i][-1,col])
    if type==1:
        plot_y = moving_average(np.asarray(plot_y),5)
        label_name = label_name + " movAvg"
    elif type==2:
        plot_y = moving_median(np.asarray(plot_y),5)
        label_name = label_name + " movMed"
    else:
        label_name = label_name + " actual"

    if not color:
        #plt.plot(num_trials,plot_y,linetype)
        plt.plot(num_trials,plot_y,linetype,label=label_name)
    else:
        plt.plot(num_trials,plot_y,linetype,c=color,label=label_name)

        plt.xlabel('Number of Trials')
        plt.ylabel('Force (N)')

def plot_force_error(data,col,type=0, color= [] , linetype = '-o' ):
    target_force = data[0][0,1]
    label_name = str(target_force)
    num_trials =  np.arange(1,len(data)+1,1)
    plot_y = []
    for i in range(0,len(data)):
            plot_y.append(np.abs(data[i][-1,col]-target_force))
    if type==1:
        label_name = label_name + "movAvg"
        plot_y = moving_average(np.asarray(plot_y),5)
    elif type==2:
        label_name = label_name + "movMed"
        plot_y = moving_median(np.asarray(plot_y),5)
    else:
        label_name = label_name + " actual"

    if not color:
        #plt.plot(num_trials,plot_y,linetype)
        plt.plot(num_trials,plot_y,linetype,label=label_name)
    else:
        plt.plot(num_trials,plot_y,linetype,c=color,label=label_name)

    plt.xlabel('Number of Trials')
    plt.ylabel('|Force Error| (N)')

def plot_displacement_progression(data,col,p,type=0, color= [] , linetype = '-o' ):
    num_trials =  np.arange(1,len(data)+1,1)
    plot_y = []
    label_name = str(data[0][0,1])
    for i in range(0,len(data)):
            plot_y.append(p(data[i][-1,col]))
    if type==1:
        plot_y = moving_average(np.asarray(plot_y),5)
        label_name = label_name + " movAvg"
    elif type==2:
        plot_y = moving_median(np.asarray(plot_y),5)
        label_name = label_name + " movMed"
    else:
        label_name = label_name + " actual"

    if not color:
        #plt.plot(num_trials,plot_y,linetype)
        plt.plot(num_trials,plot_y,linetype,label=label_name)
    else:
        plt.plot(num_trials,plot_y,linetype,c=color,label=label_name)

        plt.xlabel('Number of Trials')
        plt.ylabel('Force (N)')

def plot_displacement_error(data,col,p,type=0, color= [] , linetype = '-o' ):
    target_force = data[0][0,1]
    target_displacement = p(data[0][0,1])
    label_name = str(target_force)
    num_trials =  np.arange(1,len(data)+1,1)
    plot_y = []
    for i in range(0,len(data)):
            displacement = p(data[i][-1,col])
            plot_y.append(np.abs(displacement-target_displacement))
    if type==1:
        label_name = label_name + "movAvg"
        plot_y = moving_average(np.asarray(plot_y),5)
    elif type==2:
        label_name = label_name + "movMed"
        plot_y = moving_median(np.asarray(plot_y),5)
    else:
        label_name = label_name + " actual"

    if not color:
        #plt.plot(num_trials,plot_y,linetype)
        plt.plot(num_trials,plot_y,linetype,label=label_name)
    else:
        plt.plot(num_trials,plot_y,linetype,c=color,label=label_name)

    plt.xlabel('Number of Trials')
    plt.ylabel('|Displacement Error| (N)')


filename = "Subj1003_nohaptic_train_ef50.csv"
data = np.loadtxt(filename,delimiter=",")

target_forces = np.unique(data[:, 1])
segmented_data = []

for i in target_forces:
    temp_list = []
    temp_array = bca.split_data(data,1,i)
    trial_num = np.unique(temp_array[:,0])
    for j in trial_num:
        temp_list.append(bca.split_data(temp_array,0,j))

    segmented_data.append(temp_list)

cmap = pylab.get_cmap('brg')

for i in range(1,len(segmented_data)):
    color = cmap(1.*i/len(segmented_data))
    plot_progression(segmented_data[i],15,2,color,'-o')
    plot_progression(segmented_data[i],15,0,color,'--')

    plt.plot(np.arange(1,len(segmented_data[i])+1,1),target_forces[i]*np.ones((len(segmented_data[i]),)),'-',c=color)

plt.legend(fontsize='small')
plt.show()

plt.figure()
for i in range(1,len(segmented_data)):
    color = cmap(1.*i/len(segmented_data))
    plot_force_error(segmented_data[i],15,2,color,'-o')
    plot_force_error(segmented_data[i],15,0,color,'--')

plt.legend(fontsize='small')
plt.show()

'''crazy fitting shit'''
x = data[:,3]
y = data[:,15]
'''idx = np.where(x>0.08)
x = x[idx]
y = y[idx]'''
idx = np.where(x<0.15)
x=x[idx]
y=y[idx]
idx = np.where(y>0.2)
x=x[idx]
y=y[idx]

coeffs = np.polyfit(y,x,3)
print(coeffs)
p = np.poly1d(coeffs)
#p = np.poly1d([-0.00073916, 0.00698321,-0.01911569, -0.00194705,  0.14501212])
plt.figure()
plt.plot(x,y,'b.')
plt.plot(p(y),y,'r.')
plt.show()

'''
coeffs = np.polyfit(x,y,4)
p = np.poly1d(coeffs)
plt.figure()
plt.plot(x,y,'b.')
plt.plot(x,p(x),'r.')
plt.show()'''

plt.figure()
for i in range(1,len(segmented_data)):
    color = cmap(1.*i/len(segmented_data))
    plot_displacement_error(segmented_data[i],15,p,2,color,'-o')
    plot_displacement_error(segmented_data[i],15,p,0,color,'--')
plt.legend(fontsize='small')
plt.show()

for i in range(1,len(segmented_data)):
    color = cmap(1.*i/len(segmented_data))
    plot_displacement_progression(segmented_data[i],15,p,2,color,'-o')
    plot_displacement_progression(segmented_data[i],15,p,0,color,'--')

    plt.plot(np.arange(1,len(segmented_data[i])+1,1),p(target_forces[i])*np.ones((len(segmented_data[i]),)),'-',c=color)

plt.legend(fontsize='small')
plt.show()
