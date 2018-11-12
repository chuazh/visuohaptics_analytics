#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt

'''This script will load data from a csv file and fit a curve to the base material being tested'''

def loadfile(filename):
    data = np.loadtxt(filename,delimiter=',')
    data = data[1:-1,:]
    return data

def split_data(data, col, split_value):
    # returns an array for that target displacement level
    idx = np.where(data[:, col] == split_value)
    split_disp = data[idx]

    return split_disp

def interpAndAvg(data):
    # take in data as a list of data arrays corresponding to the same forces.

    numData = len(data)
    new_data = []

    # split the array into elongation and relaxation portions
    for i in range(0, numData):
        idx_max = np.argmax(data[i][:,6])
        new_data.append(data[i][0:idx_max,:])
        new_data.append(data[i][idx_max:-1,:])

    numData = len(new_data)
    scrubbed_data = []
    interped_forces = []
    interped_displacements = []

    for i in range(0,numData):
        # first determine that the reference array has unique values of force and time
        scrubbed_data.append(check_unique(new_data[i]))

        if i > 0:
            interped_forces.append(np.interp(scrubbed_data[0][:,3],scrubbed_data[i][:,3],scrubbed_data[i][:,6]))
            interped_displacements.append(scrubbed_data[i][:,3])
            mean_force_arr= np.vstack((mean_force_arr,np.array(interped_forces[i])))
            #mean_disp_arr = np.vstack((mean_disp_arr, np.array(interped_displacements[i])))
        else:
            interped_forces.append(scrubbed_data[i][:,6])
            interped_displacements.append(scrubbed_data[i][:, 3])
            mean_force_arr = np.array(interped_forces[i])
            #mean_disp_arr = np.array(interped_displacements[i])


    mean_force = np.mean(mean_force_arr,0)
    mean_disp = scrubbed_data[0][:,3]
    #mean_disp = np.mean(mean_disp_arr,0)

    return mean_disp,mean_force

def check_unique(data):
        # takes in an array
        val, idxF = np.unique(data[:,6],return_index=True)
        scrubbed_data = data[idxF]
        val, idxD = np.unique(scrubbed_data[:,3],return_index=True)
        unique_data = scrubbed_data[idxD]

        return unique_data

def plot_raw(segmented_data):

    list_size = len(segmented_data)

    plt.figure()

    for i in range(0, list_size):
        numTrials = len(segmented_data[i])

        for j in range(0, numTrials):
            force = segmented_data[i][j][:, 6]
            displacement = -segmented_data[i][j][:, 3]
            plt.plot(displacement, force)

    plt.show()

def plot_avgs(displacement_list,force_list):

    numForces = len(force_list)

    plt.figure()

    for j in range(0, numForces):
        force = force_list[j]
        displacement = displacement_list[j]
        plt.plot(-displacement, force)

    plt.show()

def fit_poly3(displacement_list,force_list):
    # takes in a list of forces and their corresponding displacements
    # returns a 1-D polynomial class


    numForces = len(force_list)

    # iterate through the force and disp list and populate a 2 arrays of data point pairs
    for i in range(0,numForces):
        if i == 0:
            aggregate_forces = force_list[i]
            aggregate_disp = displacement_list[i]
        else:
            aggregate_forces = np.hstack((aggregate_forces,force_list[i]))
            aggregate_disp = np.hstack((aggregate_disp,displacement_list[i]))

    coeffs = np.polyfit(-aggregate_disp,aggregate_forces,5)
    p = np.poly1d(coeffs)

    return p

def plot_poly3(polyfunc,displacement_list,force_list):
    # taks in 1-D polynomial object and the list of average displacments and forces

    numForces = len(force_list)

    # iterate through the force and disp list and populate a 2 arrays of data point pairs
    for i in range(0,numForces):
        if i == 0:
            aggregate_forces = force_list[i]
            aggregate_disp = displacement_list[i]
        else:
            aggregate_forces = np.hstack((aggregate_forces,force_list[i]))
            aggregate_disp = np.hstack((aggregate_disp,displacement_list[i]))


    x_poly = -displacement_list[0]

    plt.figure()
    plt.plot(-aggregate_disp,aggregate_forces,'.b')
    plt.plot(x_poly,polyfunc(x_poly),'r')
    plt.show()

def get_force_error_bounds(polyfunc,target_forces,eps):
    # takes in 1-D polynomial object and a list of target forces from which to calculate an error bounds based on
    # eps value which is given in terms of displacement.

    for i in target_forces:
        roots = (polyfunc-i).r
        rv = roots.real[abs(roots.imag) < 1e-5]
        print('target force = ' + str(i))
        print('upper bound = ' + str(p(rv+eps)))
        print('lower bound = ' + str(p(rv-eps)))

