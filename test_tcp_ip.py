#!/usr/bin/env python

import socket
import rospy
import os
import struct
import numpy as np
import sys
import copy 

HOST = '192.168.1.10'
PORT = 27015

s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

rospy.init_node('position_tracker_node')

rate = rospy.Rate(30)

s.connect((HOST,PORT))

print 'connected!'

filename = "tcp_data.csv"
skip_flag = False
pose_msg = 0
exit = False
pose_data = np.empty([0,3],dtype=float)

old_time = 0

while exit == False:
    time = rospy.get_time()
    #print('Fs = %d'%(1/(time-old_time)))
    pose_msg = s.recv(12,socket.MSG_WAITALL) # we receive 7 floats each of size 4 bytes  
    pose = struct.unpack('<fff',pose_msg)   # < denotes little-endian byte order for x86 systems and we have 7 fs for 7 floats
    
    print(pose)
    pose_data = np.vstack((pose_data,np.array(pose)))
    
    if len(pose_data)%10 == 0:
        print('saving...')
        f = open(filename,'ab')
        np.savetxt(f,pose_data)
        pose_data = np.empty([0,3],dtype=float)
        f.close()
    
    pose_msg = 0
    old_time = copy.copy(time)
    #rate.sleep()
   
s.close()

