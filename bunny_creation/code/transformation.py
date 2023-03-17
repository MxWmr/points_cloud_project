#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      First script of the practical session. Transformation of a point cloud
#
# ------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 13/12/2017
#


# ------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np
import math as mt

# Import functions to read and write ply files
from ply import write_ply, read_ply


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#


# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # Load point cloud
    # ****************
    #

    # Path of the file
    file_path = 'data/decimated_bunny.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate x, y, and z in a (N*3) point matrix
    points = np.vstack((data['x'], data['y'], data['z'])).T

    # Concatenate R, G, and B channels in a (N*3) color matrix
    colors = np.vstack((data['red'], data['green'], data['blue'])).T

    # Transform point cloud
    # *********************
    #
    #   Follow the instructions step by step
    #

    
    centroid = [points[:,0].mean(),points[:,1].mean(),points[:,2].mean()]
    transformed_points = np.copy(points)

    for i in range(len(points)):

        for j in range(3):
            # step b: divide scale by 2
            transformed_points[i,j]/=2
 
        # rotation !
        r = np.array([[mt.cos(30),-mt.sin(30),0],[mt.sin(30),mt.cos(30),0],[0,0,1]])
        transformed_points[i,:] = r.dot(transformed_points[i,:])
        #step d: -10cm translation on y axis
        #transformed_points[i,1]-=0.5
    
    # add outliers
    n_outliers = int(len(transformed_points)*0.2)
    min_x = np.amin(transformed_points[:,0])
    max_x = np.amax(transformed_points[:,0])
    min_y = np.amin(transformed_points[:,1])
    max_y = np.amax(transformed_points[:,1])
    min_z = np.amin(transformed_points[:,2])
    max_z = np.amax(transformed_points[:,2])
    for i in range(n_outliers):
        o_x = np.random.uniform(low=min_x,high=max_x,size=1)[0]
        o_y = np.random.uniform(low=min_y,high=max_y,size=1)[0]
        o_z = np.random.uniform(low=min_z,high=max_z,size=1)[0]
        col = np.array([[o_x],[o_y],[o_z]]).T
        transformed_points = np.concatenate([transformed_points,col],axis=0)


    print(transformed_points.shape)

    # Save point cloud
    # *********************
    #
    #   Save your result file
    #   (See write_ply function)
    #

    # Save point cloud
    write_ply('data/bunny_full_perturbed.ply', [transformed_points], ['x', 'y', 'z'])


    ## Two bunnies

    two_b = np.concatenate((points,transformed_points),axis=0)
    write_ply('data/two_bunny.ply', [two_b], ['x', 'y', 'z'])

    print('Done')
