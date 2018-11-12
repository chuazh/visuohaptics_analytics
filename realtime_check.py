import load_test_4
import numpy as np
import rospy
from geometry_msgs.msg import Wrench
import dvrk
from matplotlib import pyplot as plt
from matplotlib import animation

# initialization function: plot the background of each frame
class realtime_fig:

    def __init__(self,x_offset,poly):
        self.fig, self.ax = plt.subplots()
        self.x_offset = x_offset
        x_array = np.arange(-0.15,-0.11,0.001)
        plt.plot(x_array, poly(x_array))

        self.ani = animation.FuncAnimation(self.fig, self.animate, init_func=self.setup_plot, interval=100, blit=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        self.x = []
        self.y = []
        self.x.append(x_offset)
        self.y.append(0)

        self.ax.axis([-0.17, -0.11, -0.1, 8])
        self.scat = self.ax.scatter(self.x, self.y, animated=True)


        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat

    # animation function.  This is called sequentially
    def animate(self):
        PSM_pose = p2.get_current_position()
        pos2 = load_test_4.get_cartesian(PSM_pose)
        x_pos = pos2[0]
        self.x.append(x_pos-self.x_offset)
        self.y.append(force_feedback[0])
        data = np.vstack((self.x,self.y))
        self.scat.set_offsets(data.T)

        return self.scat

    def show(self):
        plt.show()

force_sub = rospy.Subscriber('/force_sensor', Wrench, load_test_4.haptic_feedback)
p2 = dvrk.psm('PSM2')

if __name__ == "__main__":

    sampling_period = 1 / 100
    force_feedback = [0, 0, 0]

    load_test_4.zero_forces(p2,0.05)
    PSM_pose = p2.get_current_position()
    pos2 = load_test_4.get_cartesian(PSM_pose)
    x_neutral = pos2[0]

    coeffs = np.loadtxt('avg_curves_coeffs.csv',delimiter=',')
    p = np.poly1d(coeffs)

    x_offset = np.roots(p)[np.iscomplex(np.roots(p))== False]-x_neutral

    #rt_scatter = realtime_fig(x_offset,p)
    #rt_scatter.show()

    fig, ax = plt.subplots()
    x_array = np.arange(-0.15,-0.07,0.001)
    #plt.plot(x_array, p(x_array))
    x = []
    y = []
    x.append(-0.12)
    y.append(5)

    ax.axis([-0.15, -0.11, -0.1, 8])
    scat = ax.scatter(x, y, marker='o', animated=True)
    plt.show()

    '''
    PSM_pose = p2.get_current_position()
    pos2 = load_test_4.get_cartesian(PSM_pose)
    x_pos = pos2[0]
    x.append(x_pos-x_offset)
    y.append(force_feedback[0])
    data = np.vstack((x,y))
    scat.set_offsets(data.T)
    '''

