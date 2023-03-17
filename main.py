

## import module

import numpy as np
from TEASER.source import TEASER_solver
from matplotlib import pyplot as plt
from TEASER.ply import write_ply, read_ply


## import data

if False:    #  bunny

    data_1_path = "data/decimated_bunny.ply"
    data_2_path = "data/bunny_full_perturbed.ply"

    data1_ply = read_ply(data_1_path)
    data2_ply = read_ply(data_2_path)

    data1 =  np.vstack((data1_ply['x'], data1_ply['y'], data1_ply['z']))
    data2 =  np.vstack((data2_ply['x'], data2_ply['y'], data2_ply['z']))

    two_b = np.concatenate((data1.T,data2.T),axis=0)
    write_ply('data/two_bunny_before.ply', [two_b], ['x', 'y', 'z'])

    new_data = TEASER_solver(data1.T,data2.T,n_edges=50,scale=True,translation=True,rotation=True,outliers=True)

    write_ply('data/new_bunny', [new_data], ['x', 'y', 'z'])

    # Compute RMS
    distances2_before = np.sum(np.power(data2[:,:len(data1.T)] - data1, 2), axis=0)
    RMS_before = np.sqrt(np.mean(distances2_before))
    distances2_after = np.sum(np.power(new_data.T[:,:len(data1.T)] - data1, 2), axis=0)
    RMS_after = np.sqrt(np.mean(distances2_after))

    print('Average RMS between points :')
    print('Before = {:.3f}'.format(RMS_before))
    print(' After = {:.3f}'.format(RMS_after))

    ## Two bunnies

    two_b = np.concatenate((data1.T,new_data),axis=0)
    write_ply('data/two_bunny_after.ply', [two_b], ['x', 'y', 'z'])

if True:    # Notre Dame des Champs

    data_1_path = "data/Notre_Dame_Des_Champs_2.ply"
    data_2_path = "data/Notre_Dame_Des_Champs_1.ply"

    data1_ply = read_ply(data_1_path)
    data2_ply = read_ply(data_2_path)

    data1 =  np.vstack((data1_ply['x'], data1_ply['y'], data1_ply['z']))
    data2 =  np.vstack((data2_ply['x'], data2_ply['y'], data2_ply['z']))

    two_b = np.concatenate((data1.T,data2.T),axis=0)
    write_ply('data/ndc_before.ply', [two_b], ['x', 'y', 'z'])

    new_data = TEASER_solver(data1.T,data2.T,n_edges=10000,lim_point=100000,scale=True,translation=True,rotation=True,outliers=True)

    write_ply('data/new_ndc', [new_data], ['x', 'y', 'z'])

    # Compute RMS
    distances2_before = np.sum(np.power(data2[:,:len(data1.T)] - data1, 2), axis=0)
    RMS_before = np.sqrt(np.mean(distances2_before))
    distances2_after = np.sum(np.power(new_data.T[:,:len(data1.T)] - data1, 2), axis=0)
    RMS_after = np.sqrt(np.mean(distances2_after))

    print('Average RMS between points :')
    print('Before = {:.3f}'.format(RMS_before))
    print(' After = {:.3f}'.format(RMS_after))

    ## comparison

    two_b = np.concatenate((data1.T,new_data),axis=0)
    write_ply('data/ndc__after.ply', [two_b], ['x', 'y', 'z'])
