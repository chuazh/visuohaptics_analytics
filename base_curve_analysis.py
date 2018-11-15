#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl

'''This script will load data from a csv file and fit a curve to the base material being tested'''


def loadfile(filename):
    data = np.loadtxt(filename, delimiter=',')
    data = data[1:-1, :]
    return data


def split_data(data, displacement_level):
    # returns an array for that target displacement level
    idx = np.where(data[:, 0] == displacement_level)
    split_disp = data[idx]

    idx1 = np.where(split_disp[:, 1] == 1)
    idx2 = np.where(split_disp[:, 1] == 2)

    output1 = split_disp[idx1]
    output2 = split_disp[idx2]

    output1[:,3] = output1[:,3]-output1[0,3]
    output2[:,3] = output2[:,3]-output2[0,3]

    output = [output1, output2]

    return output


def interpAndAvg(data):
    # take in data as a list of data arrays corresponding to the same forces.

    numData = len(data)
    new_data = []

    # split the array into elongation and relaxation portions
    for i in range(0, numData):
        idx_max = np.argmax(data[i][:, 6])
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
            interped_forces.append(np.interp(scrubbed_data[0][:, 3], scrubbed_data[i][:, 3], scrubbed_data[i][:, 6]))
            interped_displacements.append(scrubbed_data[i][:, 3])
            mean_force_arr = np.vstack((mean_force_arr, np.array(interped_forces[i])))
            # mean_disp_arr = np.vstack((mean_disp_arr, np.array(interped_displacements[i])))
        else:
            interped_forces.append(scrubbed_data[i][:, 6])
            interped_displacements.append(scrubbed_data[i][:, 3])
            mean_force_arr = np.array(interped_forces[i])
            # mean_disp_arr = np.array(interped_displacements[i])

    mean_force = np.mean(mean_force_arr, 0)
    mean_disp = scrubbed_data[0][:, 3]
    # mean_disp = np.mean(mean_disp_arr,0)

    return mean_disp, mean_force


def check_unique(data):
    # takes in an array
    val, idxF = np.unique(data[:, 6], return_index=True)
    scrubbed_data = data[idxF]
    val, idxD = np.unique(scrubbed_data[:, 3], return_index=True)
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

    #plt.show()


def plot_avgs(displacement_list, force_list):
    numForces = len(force_list)

    plt.figure()

    for j in range(0, numForces):
        force = force_list[j]
        displacement = displacement_list[j]
        plt.plot(-displacement, force)

    #plt.show()


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

    coeffs = np.polyfit(-aggregate_disp, aggregate_forces, 5)
    p = np.poly1d(coeffs)

    return p


def plot_poly3(polyfunc, displacement_list, force_list):
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

    plt.figure()
    plt.plot(-aggregate_disp, aggregate_forces, '.b')
    plt.plot(x_poly, polyfunc(x_poly), 'r')
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

if __name__ == "__main__":
    # START MAIN SCRIPT

    # filename = "basecurve_ef50_101618.csv"
    # filename = "basecurve_1_pull.csv"
    filename = "base_curve"

    p = []

    for j in range(0, 6):
        filename = "basecurves_111318/base_curve" + str(j + 1) + "_ds.csv"
        #filename = "basecurve2_" + str(j+1) + ".csv"
        data = loadfile(filename)
        target_displacements = np.unique(data[:, 0])
        segmented_data = []
        average_disp_curves = []
        average_force_curves = []

        for i in range(0, np.shape(target_displacements)[0]):
            segmented_data.append(split_data(data, target_displacements[i]))
            average_disp, average_force = interpAndAvg(segmented_data[i])
            average_disp_curves.append(average_disp)
            average_force_curves.append(average_force)

        plot_raw(segmented_data)
        plot_avgs(average_disp_curves, average_force_curves)
        p.append(fit_poly3(average_disp_curves, average_force_curves))
        plot_poly3(p[j], average_disp_curves, average_force_curves)
        plt.close("all")

    plt.figure()
    curves_we_like = [0,3,5]
    #curves_we_like = [0,1,2]
    f_curves_we_like = []
    d_curves_we_like = []

    ref_curve_idx = 4

    for i in curves_we_like:
        x_offset = np.amax(np.roots(p[ref_curve_idx])[np.iscomplex(np.roots(p[ref_curve_idx]))==False])-np.amax(np.roots(p[i])[np.iscomplex(np.roots(p[i]))==False])
        f_curves_we_like.append(p[i](-(average_disp_curves[-1]+x_offset)))
        d_curves_we_like.append(average_disp_curves[-1])
    p_avg = fit_poly3(d_curves_we_like,f_curves_we_like)
    plot_poly3(p_avg,d_curves_we_like,f_curves_we_like)
    plt.show()

    plt.figure()

    color = pl.cm.jet(np.linspace(0,1,len(p)))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for i in range(0, len(p)):
        #p[i].coeffs[5] += -p[ref_curve_idx](-average_disp_curves[-1])[-1] - p[i](-average_disp_curves[-1])[-1]
        #x_offset = np.amax(np.roots(p[ref_curve_idx])[np.iscomplex(np.roots(p[ref_curve_idx]))==False])-np.amax(np.roots(p[i])[np.iscomplex(np.roots(p[i]))==False])
        y_offset = np.polyval(p[ref_curve_idx],0)-np.polyval(p[i],0)
        #x_offset = np.roots(p[ref_curve_idx])[np.iscomplex(np.roots(p[ref_curve_idx]))==False][1]-np.roots(p[i])[np.iscomplex(np.roots(p[i]))==False][1]
        #plt.plot(-average_disp_curves[-1], p[i](-(average_disp_curves[-1]+x_offset)),color=color[i])

        plt.plot(-average_disp_curves[-1], p[i](-(average_disp_curves[-1]))+y_offset,color=color[i])
        #plt.xlim((0,0.045))
        major_ticks = np.arange(0, 0.06, 0.001)
        minor_ticks = np.arange(0, 0.06, 0.0001)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        # And a corresponding grid
        ax.grid(which='both')
        # Or if you want different settings for the grids:
        ax.grid(which='minor', alpha=0.75)
        ax.grid(which='major', alpha=0.75)

    plt.legend(["1", '2', '3', '4', '5', '6','2 rpt','5 rpt'], loc=0)

    #plt.plot(-average_disp_curves[-1], p_avg(-(average_disp_curves[-1]+x_offset)),lw=5)
    plt.show()
    force_bounds = get_force_error_bounds(p_avg,[1,1.5,2.5,4,5.5],0.002)
    np.savetxt('force_bounds.csv',force_bounds,'%.3f',delimiter=',')
    coeffs = p_avg.coeffs
    coeffs = coeffs.real
    np.savetxt('avg_curves_coeffs.csv',coeffs,'%.3f',delimiter=',')
    print('end')

