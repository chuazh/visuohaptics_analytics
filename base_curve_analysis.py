#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import pandas as pd
import seaborn as sns
    

'''This script will load data from a csv file and fit a curve to the base material being tested'''


def loadfile(filename):
    data = np.loadtxt(filename, delimiter=',')
    data = data[1:-1, :]
    return data

def split_data(data, displacement_level):
    # returns an array for that target displacement level
    # input:  data array Nx16
    #         target displacement to extract from data array
    # output: list of data arrays for that discplament level. The length of the list is the number of trials at that target displacement.  
    
    idx = np.where(data[:, 0] == displacement_level) # find the entries in the data that equal the target displacement
    split_disp = data[idx] # grab only those entries
    max_trials = np.amax(split_disp[:,1]) # look at the trial number to get the max trial number
    idxes =[]
    output=[]
    for i in range(0,int(max_trials)): # for each trial do
        idxes.append(np.where(split_disp[:, 1] == i+1)) # append to list an array of idxes corresponding to that trial
        output.append(split_disp[idxes[i]]) # append to list an array of the outputs of that trial.
        output[i][:,6] = output[i][:,6]-output[i][0,6] # zero out column 6 using the initial displacement

    return output

def interpAndAvg(data):
    # input:    data as a list of data arrays corresponding to the same forces.
    # output:   an vector of the displacement and a vector of force
    # This function first splits each array (in the list) into an elongation and retraction portion and interpolates the curves using the first curves displacement values
    # Once they are interpolated into the same x values, it averages them to get a single curve.

    numData = len(data)
    new_data = []

    # split the array into elongation and relaxation portions
    for i in range(0, numData):
        idx_max = np.argmax(data[i][:, 13])
        new_data.append(data[i][0:idx_max, :])
        new_data.append(data[i][idx_max:-1, :])

    numData = len(new_data)
    scrubbed_data = []
    interped_forces = []
    interped_displacements = []

    for i in range(0, numData):
        # first determine that the reference array has unique values of force and time
        scrubbed_data.append(check_unique(new_data[i]))

        if i > 0:
            interped_forces.append(np.interp(scrubbed_data[0][:, 6], scrubbed_data[i][:, 6], scrubbed_data[i][:, 13]))
            interped_displacements.append(scrubbed_data[i][:, 6])
            mean_force_arr = np.vstack((mean_force_arr, np.array(interped_forces[i])))
            # mean_disp_arr = np.vstack((mean_disp_arr, np.array(interped_displacements[i])))
        else:
            interped_forces.append(scrubbed_data[i][:, 13]) # initialize the force data points
            interped_displacements.append(scrubbed_data[i][:, 6]) # initialize the displacement data points
            mean_force_arr = np.array(interped_forces[i])
            # mean_disp_arr = np.array(interped_displacements[i])

    mean_force = np.mean(mean_force_arr, 0)
    mean_disp = scrubbed_data[0][:, 6]
    # mean_disp = np.mean(mean_disp_arr,0)

    return mean_disp, mean_force


def check_unique(data):
    # takes in an array
    val, idxF = np.unique(data[:, 13], return_index=True)
    scrubbed_data = data[idxF]
    val, idxD = np.unique(scrubbed_data[:, 6], return_index=True)
    unique_data = scrubbed_data[idxD]

    return unique_data


def plot_raw(segmented_data,color):
    list_size = len(segmented_data)

    #plt.figure()

    for i in range(0, list_size):
        numTrials = len(segmented_data[i])

        for j in range(0, numTrials):
            force = segmented_data[i][j][:, 13]
            displacement = -segmented_data[i][j][:, 6]
            plt.plot(displacement, force,color=color)
            #plt.plot(displacement, force)

    #plt.show()


def plot_avgs(displacement_list, force_list):
    
    numForces = len(force_list)
    
    color = sns.color_palette("GnBu_d",numForces)
    
    #plt.figure()

    for j in range(0, numForces):
        force = force_list[j]
        displacement = displacement_list[j]
        plt.plot(-displacement, force, color = color[j])

    #plt.show()
    
def plot_avg_seaborn(displacement_list,force_list):
    # for this functiont the displacements in discplacement list should be all the same.

    sns.set_palette("Reds_r")

    df = pd.DataFrame()
    disp = []
    force = []
    name = []
    
    for i in range(len(displacement_list)):
        disp = disp+ list(-1*displacement_list[i])
        force = force + list(force_list[i])
        name = name + [str(i) for k in range(len(displacement_list[i]))]
    
    df['disp'] = disp
    df['force'] = force
    df['name'] = name
    
    #plt.figure()
    g = sns.lineplot(x="disp", y="force",ci = 'sd' ,data = df )
            
    return df,g
    


def fit_poly3(displacement_list, force_list):
    # takes in a list of forces and their corresponding displacements
    # returns a 1-D polynomial class

    numForces = len(force_list)

    # iterate through the force and disp list and populate a 2 arrays of data point pairs
    for i in range(0, numForces):
        if i == 0:
            aggregate_forces = force_list[i]
            aggregate_disp = displacement_list[i]
        else:
            aggregate_forces = np.hstack((aggregate_forces, force_list[i]))
            aggregate_disp = np.hstack((aggregate_disp, displacement_list[i]))

    coeffs = np.polyfit(-aggregate_disp, aggregate_forces, 3)
    p = np.poly1d(coeffs)

    return p


def plot_poly3(polyfunc, displacement_list, force_list,color='r'):
    # taks in 1-D polynomial object and the list of average displacments and forces

    numForces = len(force_list)

    # iterate through the force and disp list and populate a 2 arrays of data point pairs
    for i in range(0, numForces):
        if i == 0:
            aggregate_forces = force_list[i]
            aggregate_disp = displacement_list[i]
        else:
            aggregate_forces = np.hstack((aggregate_forces, force_list[i]))
            aggregate_disp = np.hstack((aggregate_disp, displacement_list[i]))

    x_poly = -displacement_list[-1]

    #plt.figure()
    #plt.plot(-aggregate_disp, aggregate_forces, '.b')
    plt.plot(x_poly, polyfunc(x_poly),color = color)
    #plt.show()


def get_force_error_bounds(polyfunc, target_forces, eps):
    # takes in 1-D polynomial object and a list of target forces from which to calculate an error bounds based on
    # eps value which is given in terms of displacement.

    force_bound = np.zeros((3,len(target_forces)))

    for i in range(0,len(target_forces)):
        roots = (polyfunc - target_forces[i]).r
        rv = roots.real[abs(roots.imag) < 1e-5]
        force_bound[0,i] = target_forces[i]
        force_bound[1,i] = np.absolute(polyfunc(rv + eps))
        force_bound[2,i] = np.absolute(polyfunc(rv-eps))
        print('target force = ' + str(target_forces[i]))
        print('upper bound = ' + str(np.absolute(force_bound[1,i])))
        print('lower bound = ' + str(np.absolute(force_bound[2,i])))
    
    return force_bound

  
#%%
        
if __name__ == "__main__":
    # START MAIN SCRIPT
    
    filename = "base_curve"

    p = []
    #color = pl.cm.jet(np.linspace(0,1,15))
    color = sns.color_palette("Blues",15)
    fig, axs = plt.subplots(2)
    k= 0 

    for j in range(0, 15):
    #for j in [2,3]:
        filename = "basecurves_062419/original names/062419_" + str(j + 1) + "_2.csv"
        #filename = "basecurves_062419/original names/062419" + str(j + 1) + "_2.csv"
        print(filename)
        data = loadfile(filename)
        target_displacements = np.unique(data[:, 0])
        segmented_data = []
        average_disp_curves = []
        average_force_curves = []
        
        # split the data and average the load and unload portions over all trials at that discplacement level to get a single curve
        for i in range(0, np.shape(target_displacements)[0]):
            segmented_data.append(split_data(data, target_displacements[i]))
            average_disp, average_force = interpAndAvg(segmented_data[i])
            average_disp_curves.append(average_disp)
            average_force_curves.append(average_force)
        
        # some visualization code right here

        #plot_raw(segmented_data,color[j])
        plt.sca(axs[0])
        plot_avgs(average_disp_curves, average_force_curves)
        p.append(fit_poly3(average_disp_curves, average_force_curves))
        plt.sca(axs[1])
        #plot_poly3(p[j], average_disp_curves, average_force_curves)
        plot_poly3(p[k], average_disp_curves, average_force_curves)
        k+=1
        plt.draw()
        #plt.waitforbuttonpress()
        #plt.close("all")
    
    for j in range(0, 14):
    #for j in [2,3]:
        filename = "basecurves_062619/original names/062619_" + str(j + 1) + "_2.csv"
        #filename = "basecurves_062419/original names/062419" + str(j + 1) + "_2.csv"
        print(filename)
        data = loadfile(filename)
        target_displacements = np.unique(data[:, 0])
        segmented_data = []
        average_disp_curves = []
        average_force_curves = []
        
        # split the data and average the load and unload portions over all trials at that discplacement level to get a single curve
        for i in range(0, np.shape(target_displacements)[0]):
            segmented_data.append(split_data(data, target_displacements[i]))
            average_disp, average_force = interpAndAvg(segmented_data[i])
            average_disp_curves.append(average_disp)
            average_force_curves.append(average_force)
        
        # some visualization code right here

        #plot_raw(segmented_data,color[j])
        plt.sca(axs[0])
        plot_avgs(average_disp_curves, average_force_curves)
        p.append(fit_poly3(average_disp_curves, average_force_curves))
        plt.sca(axs[1])
        #plot_poly3(p[j], average_disp_curves, average_force_curves)
        plot_poly3(p[k], average_disp_curves, average_force_curves,'b')
        k+=1
        plt.draw()
        #plt.waitforbuttonpress()
        #plt.close("all")
    
        
    plt.show()
#%% I think this code does an x-offset 
    
    plt.figure()
    
    # curves we like is what will get 
    curves_we_like = [0] # always -1 of the sample number
    #curves_we_like = np.arange(0,15)
    f_curves_we_like = []
    d_curves_we_like = []

    ref_curve_idx = 0
    
    # the x offset makes an assumption. that it is not the force sensor that has a offset error, but the length of the sample at zero that has some offset error?
    # need to think more about this...
    
    for i in curves_we_like:
        # the x offset is the difference of the x intercepts of the reference curve and the curve i
        # we get this by taking the largest root that is not complex for both curves.
        x_offset = np.amax(np.roots(p[ref_curve_idx])[np.iscomplex(np.roots(p[ref_curve_idx]))==False])-np.amax(np.roots(p[i])[np.iscomplex(np.roots(p[i]))==False])
        # using the offset and the displacement values of the last curve (why though?) we can recompute the forces but now shifted.
        f_curves_we_like.append(p[i](-(average_disp_curves[-1]+x_offset)))
        d_curves_we_like.append(average_disp_curves[-1])

    p_avg = fit_poly3(d_curves_we_like,f_curves_we_like)
    plot_poly3(p_avg,d_curves_we_like,f_curves_we_like)
    #plot_avgs(d_curves_we_like,f_curves_we_like)
    plot_avg_seaborn(d_curves_we_like,f_curves_we_like)
    plt.show()
    
    
#%% I think this code does the y offset
    
    color = pl.cm.jet(np.linspace(0,1,len(p)))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for i in curves_we_like: #for i in range(0, len(p)):
        #p[i].coeffs[5] += -p[ref_curve_idx](-average_disp_curves[-1])[-1] - p[i](-average_disp_curves[-1])[-1]
        #x_offset = np.amax(np.roots(p[ref_curve_idx])[np.iscomplex(np.roots(p[ref_curve_idx]))==False])-np.amax(np.roots(p[i])[np.iscomplex(np.roots(p[i]))==False])
        y_offset = np.polyval(p[ref_curve_idx],0)-np.polyval(p[i],0)
        #x_offset = np.roots(p[ref_curve_idx])[np.iscomplex(np.roots(p[ref_curve_idx]))==False][1]-np.roots(p[i])[np.iscomplex(np.roots(p[i]))==False][1]
        #plt.plot(-average_disp_curves[-1], p[i](-(average_disp_curves[-1]+x_offset)),color=color[i])

        x = np.linspace(-60,0,100)
        plt.plot(-x, p[i](-x)+y_offset,color=color[i])
        #plt.plot(-average_disp_curves[-1], p[i](-(average_disp_curves[-1]))+y_offset,color=color[i])
        #plt.xlim((0,0.045))
        major_ticks = np.arange(0, 60, 1)#major_ticks = np.arange(0, 0.06, 0.001)
        minor_ticks = np.arange(0, 60, 0.1)#minor_ticks = np.arange(0, 0.06, 0.0001)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        # And a corresponding grid
        ax.grid(which='both')
        # Or if you want different settings for the grids:
        ax.grid(which='minor', alpha=0.75)
        ax.grid(which='major', alpha=0.75)

    #plt.legend(["1", '2', '3', '4', '5', '6','7','8','9','10','11','12','13','14','15','16','17','18','19'], loc=0)

    #plt.plot(-average_disp_curves[-1], p_avg(-(average_disp_curves[-1]+x_offset)),lw=5)
    plt.show()
    
#%%
    
    force_bounds = get_force_error_bounds(p_avg,[1.5,2.5,3.5,4.5,6],2)
    np.savetxt('force_bounds.csv',force_bounds,'%.3f',delimiter=',')
    coeffs = p_avg.coeffs
    coeffs = coeffs.real
    np.savetxt('avg_curves_coeffs.csv',coeffs,'%.3f',delimiter=',')
    print('end')
'''
#%% This can be used as a check for the maximum discplacement we will get
    
    # calculate the displacements at 10N
    print("calc roots at 10N")
    array10n = []
    for i in range(0,19):
        y_offset = np.polyval(p[ref_curve_idx],0)-np.polyval(p[i],0)
        yy = (p[i]-10-y_offset).r
        array10n.append(np.real(yy[~np.iscomplex(yy)])[0])

    print(array10n)
    plt.figure()
    plt.plot(np.linspace(1,19,19),array10n,'x')
    plt.grid(which="both")
    plt.show

#%%
    import pickle
    
    f = open('test_d.dat','rb')
    test_displacement = pickle.load(f)
    f.close()
    
    f = open('test_f.dat','rb')
    test_force = pickle.load(f)
    f.close()
    
    f=open('train_d.dat','rb')
    train_displacement = pickle.load(f)
    f.close()
    
    f = open('train_f.dat','rb')
    train_force = pickle.load(f)
    f.close()
    
    plt.gca()
    n = 5-1
    plt.plot(train_displacement[n],train_force[n],'.m')
    plt.plot(test_displacement[n],test_force[n],'.g')
'''