## import modules

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import time
from tqdm import tqdm
import math as mt


## Step 1 Compute TIM and TRIM


def gen_graphe(points,n_edges):
    # random graph with 50 edges
    edges=[]

    buffer = np.random.choice(len(points),2*n_edges,replace=False)

    for i in range(0,len(buffer),2):
        edges.append((buffer[i],buffer[i+1]))

    return edges

def get_tim_trim(points1,points2,edges1,beta):

    K = len(edges1)
    a = np.zeros((K,3))
    b = np.zeros((K,3))
    delta = np.zeros(K)
    s_ij = np.zeros(K)
    alpha = np.zeros(K)

    for k,(i,j) in enumerate(tqdm(edges1)):
        a[k,:] = points1[j,:]-points1[i,:]
        b[k,:] = points2[j,:]-points2[i,:]
        delta[k] = beta[i]+beta[j]
        b_norm = np.linalg.norm(points2[j,:]-points2[i,:])
        s_ij[k]=np.linalg.norm(points1[j,:]-points1[i,:])/b_norm
        alpha[k] = delta[k]/b_norm

    return a,b,delta,s_ij,alpha


## Step 2 Scale estimation


def scale_adaptative_voting(sk,alpha,c):

    K = len(sk)
    v = []
    for i in range(K):
        v.append(sk[i]+alpha[i]*c)
        v.append(sk[i]-alpha[i]*c)

    v = np.sort(np.array(v))
    print(len(v))
    m = np.array([(v[i]+v[i+1])/2 for i in range(2*K-1)])

    consensus = []
    for i in tqdm(range(2*K-1)):
        consensus_i = []
        for k in range(K):
            if m[i]>=sk[k]-alpha[k]*c and m[i]<=sk[k]+alpha[k]*c:
                consensus_i.append(k)
        if len(consensus_i)>len(consensus):
            consensus = consensus_i
    
    sum1 = 0
    sum2 = 0
    for k in consensus:
        sum1+=1/alpha[k]**2
        sum2+=sk[k]/alpha[k]**2
    s_opt = sum2/sum1
    return s_opt


## Step 3 Prune outliers


def max_clique(edges,s_ij,s_opt,alpha,c):
    K = len(edges)
    id_edges = []
    for k in tqdm(range(K)):
        if abs(s_ij[k]-s_opt)>=c*alpha[k]:
            id_edges.append(k)

    new_edges=[]
    for i in id_edges:
        new_edges.append(edges[i])
    return new_edges


## Step 4 Rotation estimation

def omega_1(q):
    [[q1,q2,q3],q4] = q

    return np.array([[q4,-q3,q2,q1],[q3,q4,-q1,q2],[-q2,q1,q4,q3],[-q1,-q2,-q3,q4]])

def omega_2(q):
    [[q1,q2,q3],q4] = q

    return np.array([[q4,q3,-q2,q1],[-q3,q4,q1,q2],[q2,-q1,q4,q3],[-q1,-q2,-q3,q4]])


def get_Q(a,b,delta,c,s,K):

    Q = np.zeros((4*(K),4*(K)))
            

    for k in range(K):
        Q_k = np.zeros((4*(K),4*(K)))

        a_p = np.array([s*a[k].T,0]).T
        b_p = np.array([s*b[k].T,0]).T

        Q_kk = ((np.linalg.norm(b[k])**2+np.linalg.norm(a[k])**2)*np.eye(4)+2*omega_1(b_p)+omega_2(a_p))/(2*delta[k]**2)+c**2/2*np.eye(4)

        Q_0k = ((np.linalg.norm(b[k])**2+np.linalg.norm(a[k])**2)*np.eye(4)+2*omega_1(b_p)+omega_2(a_p))/(4*delta[k]**2)-c**2/4*np.eye(4)

        Q_k[4*k:4*k+4,4*k:4*k+4] = Q_kk

        Q_k[0:4,4*k:4*k+4] =  Q_0k

        Q_k[4*k:4*k+4,0:4] = Q_0k

        Q += Q_k

    return Q


def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1*x0 - y1*y0 - z1*z0 + w1*w0,
                        x1*w0 + y1*z0 - z1*y0 + w1*x0,
                        -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                        x1*y0 - y1*x0 + z1*w0 + w1*z0], dtype=np.float64)


def SDRRC(a,b,delta,c,s):

    K = len(delta)

    # define Q

    Q = get_Q(a,b,delta,c,s,K)


    # define function to optimize

    def f_to_opt(x):
        Z = np.reshape(x,(4*K,4*K))
        return np.trace(Q.dot(Z))

    # define constraints

    def f_cons1(x):
        Z = np.reshape(x,(4*K,4*K))
        out = np.trace(Z[:4,:4]) - 1
        return out

    l_cons = []
    l_cons.append({'type':'eq', 'fun': f_cons1})


    for k in range(K):

        def f_cons2(x):
            Z = np.reshape(x,(4*K,4*K))
            out =Z[4*k:4*k+4,4*k:4*k+4]-Z[:4,:4]
            return out

        l_cons.append({'type':'eq', 'fun': f_cons2})


    for i in range(K):
        for j in range(i+1,K):
            def f_cons3(x):
                Z = np.reshape(x,(4*K,4*K))
                out =Z[4*i:4*i+4,4*j:4*j+4]-Z.T[4*i:4*i+4,4*j:4*j+4]
                return out

            l_cons.append({'type':'eq', 'fun': f_cons3})


    # Resolve the optimization problem
    x_0 = np.random.rand(4*K*4*K)

    res = minimize(f_to_opt,x_0, method='SLSQP',constraints=l_cons, options={'ftol': 1e-9, 'disp': True})

    Z_opt = res.x

    # find q from Z

    q = Z_opt[:4,:4].dot(np.linalg.inv(Z_opt[:4,:4].T))

    # find R from q

    [qw,qx,qy,qz] = q

    R = np.array([[1 - 2*qy**2 - 2*qz**2,2*qx*qy - 2*qz*qw   ,2*qx*qz + 2*qy*qw],
                  [2*qx*qy + 2*qz*qw 	,1 - 2*qx**2 - 2*qz**2,2*qy*qz - 2*qx*qw],
                  [2*qx*qz - 2*qy*qw 	,2*qy*qz + 2*qx*qw 	 ,1 - 2*qx**2 - 2*qy**2]])

    return R

def classic_rotation_estimation(a,b,s,delta,c):

    K = len(a)

    # define function to optimize
    def f_to_opt(x):
        R = np.reshape(x,(3,3))
        sum = 0
        for k in range(K):
            sum += min(np.linalg.norm(a[k]-s*R.dot(b[k]))**2/delta[k]**2,c**2)
        return sum
    
    # Resolve the optimization problem
    x_0 = np.random.rand(3*3)

    res = minimize(f_to_opt,x_0, method='Newton-CG',options={'xatol': 1e-8, 'disp': True})

    x_opt = res.x
    R_opt = np.reshape(x_opt,(3,3))

    return R_opt

def classic_rotation_fast(a,b,s,delta,c):
    set2 = s*b
    R,rssd= Rotation.align_vectors(a,set2)
    return R.as_matrix()


## Step 5 Translation estimation



def translation_adaptative_voting(points1,points2,s,R,beta,c,lim_point):
    K = len(points1)
    if K > lim_point:
        K = lim_point
        new_id = np.random.randint(0,len(points1),K)
        points1 = points1[new_id,:]
        points2 = points2[new_id,:]
        beta = beta[new_id]
    t_k = []

    

    for i in range(K):
        t_k.append(points1[i]-s*R.dot(points2[i]))
  
       
    t_k = np.array(t_k)
    t_opt=np.zeros(3)

    for j in range(3):

    
        v = []
        for i in range(K):
            v.append(t_k[i,j]+beta[i]*c)
            v.append(t_k[i,j]-beta[i]*c)

        v = np.sort(np.array(v))

        m = np.array([(v[i]+v[i+1])/2 for i in range(2*K-1)])

        consensus = []
        for i in tqdm(range(2*K-1)):
            consensus_i = []
            for k in range(K):
                if m[i]>=t_k[k,j]-beta[k]*c and m[i]<=t_k[k,j]+beta[k]*c:
                    consensus_i.append(k)
            if len(consensus_i)>len(consensus):
                consensus = consensus_i
        
        sum1 = 0
        sum2 = 0
        for k in consensus:
            sum1+=1/beta[k]**2
            sum2+=t_k[k,j]/beta[k]**2

        t_opt[j] = sum2/sum1

    return t_opt



## adjust points2

def adjust_points(points2,s,R,t):

    new_points = np.zeros_like(points2)

    for i,p in enumerate(tqdm(points2)):
        p2 = s*R.dot(p)
        new_points[i,:] = p2+t

    return new_points



def TEASER_solver(points1,points2,n_edges=50,lim_point=100000,scale=True,translation=True,rotation=True,rot_method='classic',outliers=True):

    c = 1
    beta = [0.01]*len(points1)
    beta = np.array(beta)

    t1 = time.time()

    ## generate graph 
    print('graph generation')
    edges = gen_graphe(points1,n_edges)
    print('done')

    ## get tim and trim
    print('Getting TIM and TRIM')
    a,b,delta,s_ij,alpha = get_tim_trim(points1,points2,edges,beta)
    print('done')

    ## scale estimation

    if scale:
        print('Scale estimation')
        s = scale_adaptative_voting(s_ij,alpha,c)
        print(s)
        print('done')
    else:
        s =1
    

    ## prune outliers 

    if outliers:
        print('Pruning outliers')
        edges = max_clique(edges,s_ij,s,alpha,c)
        print('done')
    
    ## rotation estimation 

    if rotation:
        print('Rotation estimation')
        if rot_method == 'classic':
            R = classic_rotation_fast(a,b,s,delta,c)
        else:
            R = SDRRC(a,b,delta,c,s)
        print('Done')
    else:
        R = np.eye(3)


    ## translation estimation 

    if translation:
        print('Translation estimation')
        t = translation_adaptative_voting(points1,points2,s,R,beta,c,lim_point)
        print(t)
        print('done')
    else:
        t=np.array([0,0,0])


    ## adjust the points 2
    print('Adjusting the cloud')
    new_points = adjust_points(points2,s,R,t)
    t2 = time.time()
    print('all done in {}'.format(t2-t1))

    return new_points

