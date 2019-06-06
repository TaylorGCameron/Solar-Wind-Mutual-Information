# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 11:07:26 2017

@author: Taylor
"""
import matplotlib.pyplot as plt
import numpy as np
import mutual_information.useful_functions as uf 

import matplotlib
plot_color = matplotlib.cm.plasma
plot_color.set_bad('white',1.)

def histogram3d(X,Y,Z, nbins = 10, prange = [[0,1],[0,1],[0,1]]):
    
    if len(X) != len(Y) or len(Y) != len(Z):
        print ('Make sure each list has the same length')
        return 0
    

    bZ = np.linspace(prange[2][0], prange[2][1], nbins+1)

    hist = np.zeros([nbins, nbins, nbins])

    for i in range(nbins):

        #make local list
        Xz = X[np.logical_and(Z > bZ[i], Z < bZ[i+1])]
        Yz = Y[np.logical_and(Z > bZ[i], Z < bZ[i+1])]

        heatmap, xedges, yedges = np.histogram2d(Xz,Yz, bins=nbins, range = [[prange[0][0], prange[0][1]],[prange[1][0], prange[1][1]]])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
        #heatmap = heatmap.T
        
        hist[:,:,i] = heatmap
        
    bX = xedges
    bY = yedges
                
    
    return np.array([bX,bY, bZ]) , hist

def keep_finite3(q1, q2, q3):
    
    q11 = q1[np.logical_and(np.logical_and(np.isfinite(q1), np.isfinite(q2)), np.isfinite(q3))]
    q22 = q2[np.logical_and(np.logical_and(np.isfinite(q1), np.isfinite(q2)), np.isfinite(q3))]
    q33 = q3[np.logical_and(np.logical_and(np.isfinite(q1), np.isfinite(q2)), np.isfinite(q3))]

    return q11, q22, q33
def information(X, nbins = 10, prange = [0,1]):
    
    h, edges = np.histogram(X, range = [prange[0],prange[1]], bins = nbins)
    
    px = h/float(np.sum(h))
    
    inf = np.sum(-1*px * np.log2(px))
    
    return inf
def mutual_information(X,Y, nbins = 10, prange = [[0,1],[0,1]]):
    X,Y = uf.keep_finite(X,Y)
    
    heatmap, xedges, yedges = np.histogram2d(X,Y, bins=nbins, range = [[prange[0][0], prange[0][1]],[prange[1][0], prange[1][1]]])
    
    pxy = heatmap/np.sum(heatmap) 

    px = np.sum(heatmap,axis = 1)/np.sum(heatmap)
    
    py = np.sum(heatmap,axis =0)/np.sum(heatmap)
    
    py = np.sum(heatmap,axis =0)/np.sum(heatmap)

    qxy = np.dot(np.reshape(px,[nbins,1]),np.reshape(py,[1,nbins]))
    
    mi = 0.
    
    for i in range(nbins):
        for j in range(nbins):
            if pxy[i,j] != 0:    
                mi = mi + pxy[i,j] *np.log2(pxy[i,j]/qxy[i,j])
    
    return mi

def pointwise_mutual_information(X,Y, nbins = 10, prange = [[0,1],[0,1],[0,1]]):
    X,Y = uf.keep_finite(X,Y)
    
    heatmap, xedges, yedges = np.histogram2d(X,Y, bins=nbins, range = [[prange[0][0], prange[0][1]],[prange[1][0], prange[1][1]]])
    
    pxy = heatmap/np.sum(heatmap) 

    px = np.sum(heatmap,axis = 1)/np.sum(heatmap)
    
    py = np.sum(heatmap,axis =0)/np.sum(heatmap)
    
    py = np.sum(heatmap,axis =0)/np.sum(heatmap)

    qxy = np.dot(np.reshape(px,[nbins,1]),np.reshape(py,[1,nbins]))
    
    
    pmi = np.zeros([nbins, nbins])
    
    for i in range(nbins):
        for j in range(nbins):
            if pxy[i,j] != 0:    
                pmi[i,j] = np.log2(pxy[i,j]/qxy[i,j])
            else:
                pmi[i,j] = np.nan
    
    return pmi


def conditional_mutual_information(X,Y,Z, nbins = 10, prange = [[0,1],[0,1],[0,1]]):
    X,Y,Z = keep_finite3(X,Y,Z)
    
    edges, heatmap = histogram3d(X,Y,Z, nbins=nbins, prange = [[prange[0][0], prange[0][1]],[prange[1][0], prange[1][1]],[prange[2][0], prange[2][1]]])
    
    pxyz = heatmap/np.sum(heatmap) 

    px = np.sum(heatmap,axis = (1,2))/np.sum(heatmap)
    py = np.sum(heatmap,axis =(0,2))/np.sum(heatmap)
    pz = np.sum(heatmap,axis =(0,1))/np.sum(heatmap)

    pxz = np.sum(heatmap,axis =(1))/np.sum(heatmap)
    pyz= np.sum(heatmap,axis =(0))/np.sum(heatmap)

    qxy = np.dot(np.reshape(px,[nbins,1]),np.reshape(py,[1,nbins]))
    
    qxyz = np.dot(np.reshape(qxy,[nbins, nbins, 1]), np.reshape(pz,[1,nbins]))
    
    mi = 0.
    
    for i in range(nbins):
        for j in range(nbins):
            for k in range(nbins):
                if pxyz[i,j,k] != 0:    
                    mi = mi + pxyz[i,j,k] *np.log2((pz[k]*pxyz[i,j,k])/(pxz[i,k] * pyz[j,k]))
    
    return mi

def pointwise_conditional_mutual_information(X,Y,Z, nbins = 10, prange = [[0,1],[0,1],[0,1]], sumoverz = True):
    X,Y,Z = keep_finite3(X,Y,Z)
    
    edges, heatmap = histogram3d(X,Y,Z, nbins=nbins, prange = [[prange[0][0], prange[0][1]],[prange[1][0], prange[1][1]],[prange[2][0], prange[2][1]]])
    
    pxyz = heatmap/np.sum(heatmap) 

    px = np.sum(heatmap,axis = (1,2))/np.sum(heatmap)
    py = np.sum(heatmap,axis =(0,2))/np.sum(heatmap)
    pz = np.sum(heatmap,axis =(0,1))/np.sum(heatmap)

    pxz = np.sum(heatmap,axis =(1))/np.sum(heatmap)
    pyz= np.sum(heatmap,axis =(0))/np.sum(heatmap)

    qxy = np.dot(np.reshape(px,[nbins,1]),np.reshape(py,[1,nbins]))
    
    qxyz = np.dot(np.reshape(qxy,[nbins, nbins, 1]), np.reshape(pz,[1,nbins]))
    
    pcmi = np.zeros([nbins, nbins, nbins])
    
    for i in range(nbins):
        for j in range(nbins):
            for k in range(nbins):
                if pxyz[i,j,k] != 0:    
                    pcmi[i,j,k] = np.log2((pz[k]*pxyz[i,j,k])/(pxz[i,k] * pyz[j,k]))
                else:
                    pcmi[i,j,k] = np.nan
    
    if sumoverz == True:
        pzz = np.zeros([nbins, nbins, nbins])
        for i in range(nbins):
            for j in range(nbins):
                pzz[i,j,:] = pz
        
        pcmi = np.nansum(pzz * pcmi, axis = 2)
    
    return pcmi


def gauss(x, *p):
    ''' Calculates a gaussian for a set of x values '''
    A, mu, sigma,c = p
    return np.abs(A)*np.exp(-(x-mu)**2/(2.*sigma**2))+c



def plot_pmi(pmi, title = 'INSERT TITLE HERE',xlabel = "INSERT A LABEL HERE", ylabel = "INSERT A LABEL HERE" , vrange = 0.3, prange = [[-90,90.],[0,1000.]], cbar = True, cbar_title = 'INSERT A TITLE HERE'):
    a = np.flip(pmi.T, axis = 0)
    masked_array = np.ma.array (a, mask=np.isnan(a))
    p = plt.imshow(masked_array, vmin = -1 * vrange, vmax = vrange, cmap = plot_color, extent = [prange[0][0], prange[0][1],prange[1][0],prange[1][1]], aspect = 'auto')
    #p = plt.imshow(np.flip(pmi.T, axis = 0), vmin = -1 * vrange, vmax = vrange, cmap = plt.get_cmap('plasma'), extent = [prange[0][0], prange[0][1],prange[1][0],prange[1][1]], aspect = 'auto')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if cbar == True:
        plt.colorbar(label = cbar_title)
    return p