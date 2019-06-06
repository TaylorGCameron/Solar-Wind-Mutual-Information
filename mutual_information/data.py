# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 12:51:22 2019

When directly run, pulls ACE swe and mfi data, and GOES data from 2000 to 2009 
from CDAWeb and stores it in files in a location specified in config.par.

@author: Taylor
"""

import numpy as np
import matplotlib.dates as mdate
import datetime
from ai import cdas
import mutual_information.useful_functions as uf
import os
import numpy.linalg as LA

def process_SuperMAG():
    start_year = int(uf.get_parameter('start_year'))
    end_year = int(uf.get_parameter('end_year'))

    #check for superMAG data
    txt_filepath = uf.get_parameter('SuperMAG_filepath')
    if not os.path.isfile(txt_filepath+'SuperMAG_2000.txt'):
        print('Hey, check your SuperMAG txt files. They arent there.')

    for year in range(start_year,end_year): 
    
        print('Processing data for '+str(year))
        filepath = uf.get_parameter('filepath')
    
        #check if there's a folder there, if not, make it
        if not os.path.exists(filepath+'Data/'):
            os.makedirs(filepath+'Data/')
        
        filename = filepath+'Data/SuperMAG_'+str(year)+'.npy'
    
    
        #Check if file already exists
        if os.path.exists(filename):
             print('File '+'SuperMAG_'+str(year)+'.npy'+' already exists! Skipping...')
             continue

        
        data = np.loadtxt(txt_filepath+'SuperMAG_'+str(year)+'.txt', skiprows = 88)
        
        years = np.array(data[...,0], dtype = int)
        month = np.array(data[...,1], dtype = int)
        day = np.array(data[...,2], dtype = int)
        hour = np.array(data[...,3], dtype = int)
        minute = np.array(data[...,4], dtype = int)
        second = np.array(data[...,5], dtype = int)
        time = np.full(len(day), datetime.datetime(year,1,1), dtype = object)
        for j in range(len(day)):
            time[j] = datetime.datetime(years[j], month[j], day[j], hour[j], minute[j], second[j])
        
        tnum = mdate.date2num(time)
    
        sme = data[...,6]
        sme[sme > 99999] = np.nan
        sml = data[...,7]
        sml[sml > 99999] = np.nan
        smu = data[...,8]
        smu[smu > 99999] = np.nan
    
        data = None
        
        dtype = np.dtype([('t','f8'), ('sme','f8') ,('sml','f8' ), ('smu','f8' )])
        sm = np.ndarray(len(tnum), dtype = dtype)
        
        sm['t'] = tnum
        sm['sme'] = sme
        sm['sml'] = sml
        sm['smu'] = smu   
        np.save(filename, sm)
        print('Finished ', year)

def pull_data():
    '''
    Pull a range (specified in config.par) of years of ACE SWE, ACE MFI data from CDAWeb, clean it, and store it in a location specified in config.par.
    
    Arguments:
        
    Returns:
        int: Function finished indicator
    '''
    start_year = int(uf.get_parameter('start_year'))
    end_year = int(uf.get_parameter('end_year'))
    for i in range(start_year, end_year):
        pull_ACE_year(i)
        pull_ACE_B_year(i)
    print('')
    return 1

def pull_ACE_year(year):
    '''
    Pull a year of ACE SWE data from CDAWeb, clean it, and store it in a location specified in config.par
    
    Arguments:
        year(int) -- The year for which data will be pulled
        
    Returns:
        int: Function finished indicator
    '''
    print('Pulling data for '+str(year))
    filepath = uf.get_parameter('filepath')

    #check if there's a folder there, if not, make it
    if not os.path.exists(filepath+'Data/'):
        os.makedirs(filepath+'Data/')
    
    filename = filepath+'Data/ACE_'+str(year)+'.npy'


    #Check if file already exists
    if os.path.exists(filename):
         print('File '+'ACE_'+str(year)+'.npy'+' already exists! Skipping...')
         return 1
    
    #First create empty structures to hold the data
    
    ACE_dtype = np.dtype([('t','f8'), ('pos','3f8'), ('v', '3f8' ), ('n','f8' ), ('p','f8' ), ('spd','f8')])
    ACE = np.ndarray(0, dtype = ACE_dtype)
    
    print('Pulling ACE swe data from '+str(year) )
    uf.status(0)
    
    #Pull the data from CDAWeb in month chunks
    for i in range(1,13):
        t1 = datetime.datetime(year, i,1)
    
        if i+1<13:
            t2 = datetime.datetime(year, i+1,1)
        else:
            t2 = datetime.datetime(year+1, 1,1)
        #print('Pulling '+str(t1)[0:10] + ' - ' + str(t2)[0:10])

        
        swe_data = cdas.get_data('sp_phys', 'AC_H0_SWE',t1, t2, ['Np', 'Vp', 'V_GSE', 'SC_pos_GSE'])
    
        #make temp structure
        ACE_month = np.ndarray(len(swe_data['EPOCH']), dtype = ACE_dtype)
        
        #throw data into structure and clean it up
        ACE_month['t'] = mdate.date2num(swe_data['EPOCH'])
        ACE_month['pos'] = np.transpose([swe_data['ACE_X-GSE'],swe_data['ACE_Y-GSE'],swe_data['ACE_Z-GSE']])
        ACE_month['n'] = swe_data['H_DENSITY']
        ACE_month['v'] = np.transpose([swe_data['VX_(GSE)'],swe_data['VY_(GSE)'],swe_data['VZ_(GSE)']])
        
        #clean up ACE data
        ACE_month['n'][ACE_month['n'] < -10**30 ] = np.nan
        ACE_month['v'][ACE_month['v'] < -10**30] = np.nan
    
        ACE_month['spd'] = np.sqrt(np.sum(ACE_month['v']**2,axis = 1))    
        ACE_month['p'] = 1.6726*10**(-6) * ACE_month['n'] * ACE_month['spd']**2 # Units are nPa
    
        ACE = np.append(ACE,ACE_month)
        uf.status(int((i/12)*100))

    np.save(filename, ACE)
    print(str(year)+' finished!')
    print('File saved to ' + filename)
    return 1

#Take a 1D array, and return an array where every n entries were averaged together.
def collapse_down(arr,n):
    '''Average every n elements of an array (arr), returning a smaller array of length/n'''
    return np.mean(arr[:(len(arr)//n)*n].reshape(-1,n), axis=1)

#Pulls a year of ACE magnetic field data, collapses it down to 64 second cadence, and saves it to a file
def pull_ACE_B_year(year, filepath = ''):
    '''
    Pull a year of ACE MFI data from CDAWeb, clean it, and store it in a location specified in config.par
    
    Arguments:
        year(int) -- The year for which data will be pulled
        
    Returns:
        int: Function finished indicator
    '''
    
    filepath = uf.get_parameter('filepath')

    #check if there's a folder there, if not, make it
    if not os.path.exists(filepath+'Data/'):
        os.makedirs(filepath+'Data/')
    
    filename = filepath+'Data/ACE_B_'+str(year)+'.npy'


    #Check if file already exists
    if os.path.exists(filename):
         print('File '+'ACE_B_'+str(year)+'.npy'+' already exists! Skipping...')
         return 1
     
    print('Pulling ACE mfi data from '+str(year) )
    uf.status(0)

    ACE_B_dtype = np.dtype([('t','f8'), ('B','3f8' )])
    ACE_B = np.ndarray(0, dtype = ACE_B_dtype)
    
    for i in range(1,13):
                
        t1 = datetime.datetime(year, i,1)
    
        if i+1<13:
            t2 = datetime.datetime(year, i+1,1)
        else:
            t2 = datetime.datetime(year+1, 1,1)
        #print('Pulling '+str(t1)[0:10] + ' - ' + str(t2)[0:10])

        mfi_data = cdas.get_data('sp_phys', 'AC_H0_MFI', t1, t2, ['BGSEc'])

        ACE_B_month = np.ndarray(len(mfi_data['EPOCH'])//4, dtype = ACE_B_dtype)
       
        np.transpose([collapse_down(mfi_data['BX_GSE'], 4),collapse_down(mfi_data['BY_GSE'], 4),collapse_down(mfi_data['BZ_GSE'], 4)])
        
        ACE_B_month['B'] = np.transpose([collapse_down(mfi_data['BX_GSE'], 4),collapse_down(mfi_data['BY_GSE'], 4),collapse_down(mfi_data['BZ_GSE'], 4)])
        ACE_B_month['t'] = collapse_down(mdate.date2num(mfi_data['EPOCH']), 4)
        
        #Clean bad data
        ACE_B_month['B'][ACE_B_month['B'] < -10**30] = np.nan
        
        #append to the full array
        ACE_B = np.append(ACE_B,ACE_B_month)
        uf.status(int((i/12)*100))

    
    np.save(filename, ACE_B)
    print(str(year)+' finished!')
    print('File saved to ' + filename)


    
def calc_time_indices():
    '''
    Create and save to file two lists of indices for each year, for ACE swe, ACE mfi.
    The indices define time intervals separated by a time dt, of length interval_length, both contained 
    in config.par. The range of years computed for are also contained in config.par.
    
    Arguments:
        year(int) -- The year for which indices will be calculated
        
    Returns:
        int: Function finished indicator
    '''
    start_year = int(uf.get_parameter('start_year'))
    end_year = int(uf.get_parameter('end_year'))
    for i in range(start_year, end_year):
        calc_time_indices_year(i)
    print('')
    return 1
    
def calc_time_indices_year(year):
    '''
    Create and save to file two lists of indices for one year, for ACE swe, ACE mfi.
    The indices define time intervals separated by a time dt, of length interval_length. 
    (These are defined in config.par)
    
    Arguments:
        year(int) -- The year for which indices will be calculated
        
    Returns:
        int: Function finished indicator
    '''
    
    filepath = uf.get_parameter('filepath')    

    interval_length = eval(uf.get_parameter('interval_length'))
    dt = eval(uf.get_parameter('dt'))

    print('Calculating indices for '+str(year))

    if not os.path.exists(filepath+'Indices/'):
        os.makedirs(filepath+'Indices/')

    filename = filepath+'Indices/ACE_indices_'+str(year)+'.npy'

    #Check if file already exists
    if os.path.exists(filename):
         print('File '+'ACE_indices_'+str(year)+'.npy'+' already exists! Skipping...')
         return 1    
    
    ACE = np.load(filepath+'Data/ACE_'+str(year)+'.npy') 
    ACE_B = np.load(filepath+'Data/ACE_B_'+str(year)+'.npy')
            
    ACE_t = ACE['t'].copy()
    ACE_B_t = ACE_B['t'].copy()            
    #Create an array of start times based on year, with each time separated by half an hour
            
    tstart = mdate.date2num(datetime.datetime(year,1,1,1,0,0))        
    tend = mdate.date2num(datetime.datetime(year,12,31,23,0,0))

            
    start_times = np.arange(tstart+3./24.,tend-3./24.,dt)        
    end_times = start_times + interval_length
    
    ACE_B_time_indices = np.empty([len(start_times),2], dtype = int)            
    ACE_time_indices = np.empty([len(start_times),2], dtype = int)
    
    for i in range(0,len(start_times)):  
            [Abt1, Abt2] = uf.interval(start_times[i], end_times[i], ACE_B_t) 
            [At1, At2] = uf.interval(start_times[i], end_times[i], ACE_t)   
            if np.isnan(At1):
               ACE_B_time_indices[i] = [-1,-1] 
               ACE_time_indices[i] = [-1,-1] 
               continue
            if len(ACE['p'][At1:At2][np.isfinite(ACE['p'][At1:At2])]) < 20:
                ACE_B_time_indices[i] = [-1,-1] 
                ACE_time_indices[i] = [-1,-1]                
                continue
            ACE_time_indices[i] = [At1,At2]
            ACE_B_time_indices[i] = [Abt1,Abt2] 
            if np.mod(i,200) == 0 and i != 0:
                uf.status(int(float(i)/float(len(start_times))*100))

    np.save(filepath+'Indices/ACE_indices_'+str(year)+'.npy',  ACE_time_indices)
    np.save(filepath+'Indices/ACE_B_indices_'+str(year)+'.npy',  ACE_B_time_indices)    
    print('')
    return 1


def average_arr(arr, indices):
    '''
    Averages a structured array based on a list of indices indicating
    start and end chunks of data.
    
    Arguments:
        arr(array): A structured array
        indices(array) An array of size nx2, with elements referring to indices of arr
        
    Returns:
        array: A structured array of length n, containing averages of arr.
    '''
    #First create empty structures to hold the data
    avg = np.full(len(indices),np.nan ,dtype = arr.dtype)
    
    uf.status(0)
    
    for i in range(len(avg)):
        if indices[i,0] == -1:
            continue
        for var in arr.dtype.names:
            if np.isnan(arr[var][indices[i,0]:indices[i,1]]).all():
                continue
            avg[var][i]= np.nanmean(arr[var][indices[i,0]:indices[i,1]], axis = 0)
        if np.mod(i,100) == 0:
            uf.status(int(i/len(indices)*100))
    uf.status(100)
    print('')
    
    return avg

def average_ACE_year(year):
    '''
    For each interval in a year, compute the average of a bunch of different 
    ACE quantities, and save them to file for use later. Requires data files to have been downloaded.
    
    Arguments:
        year(int): The year for which the averages will be computed
    
    Returns:
        int: Function completed indicator
    '''

    filepath = uf.get_parameter('filepath')

    print('Starting '+str(year))
    
    #ACE
    #check for file
    if os.path.exists(filepath+'Averages/ACE_avg_'+str(year)+'.npy') & os.path.exists(filepath+'Averages/ACE_B_avg_'+str(year)+'.npy'):
        print('File '+'ACE_avg_'+str(year)+'.npy'+' already exists! Skipping...')
        return 1
    
    ACE = np.load(filepath+'Data/ACE_'+str(year)+'.npy')
    ACE_indices = np.load(filepath+'Indices/ACE_indices_'+str(year)+'.npy')
    
    ACE_avg = average_arr(ACE, ACE_indices)
    np.save(filepath+'Averages/ACE_avg_'+str(year)+'.npy' ,ACE_avg)

    #ACE_B
    ACE_B = np.load(filepath+'Data/ACE_B_'+str(year)+'.npy')
    ACE_B_indices = np.load(filepath+'Indices/ACE_B_indices_'+str(year)+'.npy')
    
    ACE_B_avg = average_arr(ACE_B, ACE_B_indices)
    np.save(filepath+'Averages/ACE_B_avg_'+str(year)+'.npy' ,ACE_B_avg)
    
    return ACE_avg

def average_ACE():
    '''
    Computes average_data_year() for each year from start_year to end_year as
    specified in config.par. 
    
    Arguments:
    
    Returns:
        int: Function completed indicator
    '''
    start_year = int(uf.get_parameter('start_year'))
    end_year = int(uf.get_parameter('end_year'))
    for i in range(start_year, end_year):
        average_ACE_year(i)
    return 1





def average_SuperMAG():
    filepath = uf.get_parameter('filepath')
    start_year = int(uf.get_parameter('start_year'))
    end_year = int(uf.get_parameter('end_year'))
    
    interval_length = eval(uf.get_parameter('interval_length'))
    dt = eval(uf.get_parameter('dt'))
    
    
    for year in range(start_year,end_year):
        #Load in times  
        tstart = mdate.date2num(datetime.datetime(year,1,1,1,0,0))        
        tend = mdate.date2num(datetime.datetime(year,12,31,23,0,0))        
        start_times = np.arange(tstart+3./24.,tend-3./24.,dt)        
        end_times = start_times + interval_length

        if os.path.exists(filepath+'Averages/SuperMAG_avg_'+str(year)+'.npy'):
            print('File '+'SuperMAG_avg_'+str(year)+'.npy'+' already exists! Skipping...')
            continue

        #Load in  data
        smag = np.load(filepath+'Data/SuperMAG_'+str(year)+'.npy')    
        ACE = np.load(filepath+'Averages/ACE_avg_'+str(year)+'.npy')    
        print('Processing '+str(year))
        smu = np.zeros(len(start_times))+np.nan   
        sme = np.zeros(len(start_times))+np.nan   
        sml = np.zeros(len(start_times))+np.nan   

        p0 = 0
        for i in range(len(start_times)):

            #Calculate timeshift for interval
            v= ACE['v'][i,0]
            ACE_x = ACE['pos'][i,0] 
            ts = (ACE_x/v)/86400. #get timeshift, convert to days
            
            if np.isnan(ts):
                smu[i] = np.nan
                sme[i] = np.nan
                sml[i] = np.nan
                continue
            
            t1 = start_times[i]+ts
            t2 = end_times[i]+ts
            
            t = smag['t'][np.logical_and(smag['t'] > t1, smag['t'] < t2)]
            
            if t != []:
                smu[i] = np.nanmax(smag['smu'][np.logical_and(smag['t'] > t1, smag['t'] < t2)])
                sme[i] = np.nanmax(smag['sme'][np.logical_and(smag['t'] > t1, smag['t'] < t2)])
                sml[i] = np.nanmax(smag['sml'][np.logical_and(smag['t'] > t1, smag['t'] < t2)])

            else:
                smu[i] = np.nan
                sme[i] = np.nan
                sml[i] = np.nan
                
                
            p = int(float(i)/len(start_times)*100.)
            if p != p0:
                uf.status(p)
                p0 = p
            
            
            
        print (' ')
        
        dtype = np.dtype([('sme','f8') ,('sml','f8' ), ('smu','f8' )])
        sm = np.ndarray(len(sme), dtype = dtype)
        sm['sme'] = sme
        sm['smu'] = smu
        sm['sml'] = sml
        np.save(filepath+'Averages/SuperMAG_avg_'+str(year), sm)

    
    
def calc_MVAB0_normals():
    '''
    Computes calc_MVAB0_normals_year() for each year from start_year to end_year as
    specified in config.par. 
    
    Arguments:
    
    Returns:
        int: Function completed indicator
    '''
    start_year = int(uf.get_parameter('start_year'))
    end_year = int(uf.get_parameter('end_year'))
    for i in range(start_year, end_year):
        calc_MVAB0_normals_year(i)
    return 1
    #For each interval, compute MVAB0 normals
    
def calc_MVAB0_normals_year(year):
    print('Calculating '+ str(year))
    filepath = uf.get_parameter('filepath')
        #check for file
    if os.path.exists(filepath+'Averages/MVAB0_normals_'+str(year)+'.npy'):
        print('File '+'MVAB0_normals_'+str(year)+'.npy'+' already exists! Skipping...')
        return 1
    
    #For each interval, compute MVAB0 normals

    ACE_B = np.load(filepath+'Data/ACE_B_'+str(year)+'.npy')
    ACE_B_indices = np.load(filepath+'Indices/ACE_B_indices_'+str(year)+'.npy')
    
    normals = np.full([len(ACE_B_indices),3], np.nan)
    
    p0 = 0
    for i in range(len(ACE_B_indices)):
        if ACE_B_indices[i,0] == -1:
            continue
        else:
            normals[i] = MVAB0(ACE_B['B'][ACE_B_indices[i,0]:ACE_B_indices[i,1]], 2)

        p = int(float(i)/len(ACE_B_indices)*100.)
        if p != p0:
            uf.status(p)
            p0 = p
    print('')
    dtype = np.dtype([('n','3f8')])
    data = np.ndarray(len(normals), dtype = dtype)
    data['n'] = normals
    
    np.save(filepath+'Averages/MVAB0_normals_'+str(year)+'.npy',data)
    
    
def MVAB0(B, ratio = 2):
    if len(B) == 0:
        return np.nan
    M = np.zeros([3,3])
    P = np.zeros([3,3])

    B_av = np.nanmean(B,0)
    B_norm = B_av / np.sqrt(B_av[0]**2 + B_av[1]**2 + B_av[2]**2)
    #Create a covariance matrix
    
    for i in range(3):
        for j in range(3):
            M[i,j] = np.nanmean(B[:,i]*B[:,j]) -np.nanmean(B[:,i])*np.nanmean(B[:,j])
            if i == j:
                P[i,j] = 1 - (B_norm[i]*B_norm[j])
            else:
                P[i,j] = -1* (B_norm[i]*B_norm[j])
            if np.isnan(M[i, j]):
                return np.nan
    #Get eigenvalues and eigenvectors
    M = np.dot(np.dot(P,M),P)
    eigenvalues, eigenvectors = LA.eig(M)
    args = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[args]
    eigenvectors = eigenvectors[:,args]
                
    #The vector corresponding to the middle (absolute value) eigenvalue is the minimum variance direction 

    front_normal = eigenvectors[:,1]
    
    #The x component of the vector should point towards the sun (positive)
    if front_normal[0] < 0:
        front_normal = -1*front_normal
    
    # Do a test. For the result to be valid, the second smallest eigenvalue should be x times larger than the smallest
    r = eigenvalues[2] / eigenvalues[1]   
    if r < ratio:
        return np.nan
        
    else:
        return front_normal

def load_data():
    filepath = uf.get_parameter('filepath')
    start_year = int(uf.get_parameter('start_year'))
    end_year = int(uf.get_parameter('end_year'))
    
    dtype = np.dtype([('t','f8'), ('pos','3f8'), ('v', '3f8' ), ('n','f8' ), ('p','f8' ), ('spd','f8'), ('B','3f8' ), ('normal','3f8'), ('sme','f8'), ('az','f8'), ('inc','f8'),('angle','f8'), ('smu','f8'), ('sml','f8') ])
    data = np.ndarray(0, dtype = dtype)
    
    for year in range(start_year, end_year):
        #Get ACE data
        ACE = np.load(filepath+'Averages/ACE_avg_'+str(year)+'.npy')
        ACE_B = np.load(filepath+'Averages/ACE_B_avg_'+str(year)+'.npy')
        #get normals
        normals = np.load(filepath+'Averages/MVAB0_normals_'+str(year)+'.npy')
        #get SuperMAG
        SuperMAG = np.load(filepath+'Averages/SuperMAG_avg_'+str(year)+'.npy')

        data_year = np.zeros(len(ACE), dtype = dtype)

        data_year['t'] = ACE['t']
        data_year['pos'] = ACE['pos']
        data_year['v'] = ACE['v']
        data_year['n'] = ACE['n']
        data_year['p'] = ACE['p']
        data_year['spd'] = ACE['spd']
        
        data_year['B'] = ACE_B['B']
        
        data_year['normal'] = normals['n']
        
        data_year['sme'] = SuperMAG['sme']
        data_year['smu'] = SuperMAG['smu']
        data_year['sml'] = SuperMAG['sml']
        
        data = np.append(data, data_year)
        
    data['angle'], data['inc'], data['az'] = process_normals(data['normal'])
        
    return data

def process_normals(n):

    angles = np.rad2deg(np.arccos(n[...,0]))
    ##inclination
    base = np.sqrt(n[...,0]**2+n[...,1]**2)
    inc = uf.rad2deg(np.arctan2(n[...,2],base))
    #Azimuth
    az = uf.rad2deg(np.arctan2(n[...,1],n[...,0]))
    return angles, inc, az


#Shock routines that require Denny Oliviera's shock list
#def load_shock_data():
#    LIST_PATH = 'C:/Users/Taylor/Google Drive/Science/Python/GSFC/data/'
#    
#    shock_data = np.load(LIST_PATH + 'shock_data.npy')
#    shock_t = np.load(LIST_PATH + 'shock_data_t.npy')
#    
#    shock_data = shock_data[10:]
#    shock_t = shock_t[10:]
#    
#    return shock_t, shock_data
#
#def remove_shocks(interval, q, reverse = False):
#
#    shock_t, shock_data = mi.load_shock_data()
#
#    t_full = mi.full_timelist(0.5/24.)
#    
#    truth_array = np.full(len(t_full), False, dtype = bool)
#    p0 = 0
#    for i, t in enumerate(shock_t):
#        truth_array = np.logical_or(truth_array, np.logical_and(t_full > t-3./24.,t_full < t+3./24.))
#        
#        p = int(float(i)/len(shock_t)*100.)
#        if p != p0:
#            uf.status(p)
#            p0 = p    
#
#
#
#
#    if reverse == False:
#        q = q[np.logical_not(truth_array)]
#    else:
#        q = q[truth_array]
