# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:52:47 2019

@author: razieh
"""

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import IPython.display
from skimage.feature import plot_matches
from matplotlib import cm

def display(mstr_amp, slv_amp, lineno='0', loc="0", vmin=0, vmax=255):
    PATH = "./images/original/"
    ### Master
    fig = plt.figure('master')
    ax = fig.add_subplot(111)
    cax = ax.imshow(mstr_amp, interpolation='nearest', cmap=cm.gray, vmin=vmin, vmax=vmax)
    ax.set_xlabel('Range')
    ax.set_ylabel('Azimuth')
    
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[0, mstr_amp.max()])
    plt.axis('tight')
    plt.savefig(PATH +"(master_" + loc + ")"+ lineno)
    ### Slave
    fig = plt.figure('slave')
    ax = fig.add_subplot(111)
    cax = ax.imshow(slv_amp, interpolation='nearest', cmap=cm.gray, vmin=vmin, vmax=vmax)
    ax.set_xlabel('Range')
    ax.set_ylabel('Azimuth')
    
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[0, slv_amp.max()])
    plt.axis('tight')
    plt.savefig(PATH +"(salve_" + loc + ")" + lineno)
    plt.show()


def draw_points(img, p, loc="None", name="None", method_name="None", sp=None, counter = "0", spf=False, flag=True): # p:detected points sp:detected points (sub-pixels)
    if flag==True:
        PATH = "./images/points/"
        fig, ax = plt.subplots()
        #ax.set(title=p.tostring)     
        ax.imshow(img, interpolation='nearest', cmap=cm.gray)
        ax.plot(p[:, 1], p[:, 0], 'ro', markersize=4)
        if spf!= False:
            ax.plot(sp[:, 1], sp[:, 0], '.b', markersize=2)
        plt.title(method_name+ np.str(counter))
        plt.savefig (PATH+name+loc+'_'+method_name + np.str(counter))
        plt.show()

def draw_points_4test(img, p, loc, name="None", lineno='0', method_name="None", sp=None, counter = "0"): # p:detected points sp:detected points (sub-pixels)
    PATH = "./images/points/"
    fig, ax = plt.subplots()
    #ax.set(title=p.tostring)     
    ax.imshow(img, interpolation='nearest', cmap=cm.gray)
    ax.plot(p[:, 0], p[:, 1], 'ro', markersize=4)
    if sp.any()!= None:
        ax.plot(sp[:, 1], sp[:, 0], '.b', markersize=2)
    plt.savefig (PATH+name+loc+'_'+method_name+ lineno + np.str(counter))
    
    plt.show()
def draw_fv(img, coord, direction, PATH):
    fig, ax = plt.subplots()
    ax.imshow(img, interpolation='nearest', cmap=cm.gray)
    plt.axis('tight')
    plt.title(PATH)
    ax.quiver(coord[:,1], coord[:, 0], direction[:, 0], -direction[:, 1],
                  color='red',
                  width=0.0012, headwidth=3,
                  scale=None)
    PATH = "./images/flow_vec_quiver/" + PATH
    plt.savefig (PATH)
    plt.show()
    
def draw_fv_4test(img, coord, direction, PATH):
    fig, ax = plt.subplots()
    ax.imshow(img, interpolation='nearest', cmap=cm.gray)
    plt.axis('tight')
    plt.title(PATH)
    ax.quiver(coord[:,0], coord[:, 1], direction[:, 0], -direction[:, 1],
                  color='red',
                  width=0.0015, headwidth=3,
                  scale=None)
    PATH = "./images/flow_vec_quiver/" + PATH
    plt.savefig (PATH)
    plt.show()
def plot_point_matches(src, dest, img1, img2):
    index = np.arange(0,src.shape[0],1).T
    fig, ax = plt.subplots(nrows=1, ncols=1)

    plt.gray()
    plot_matches(ax, img1, img2, src, dest, 
                 np.column_stack((index, index)), matches_color='b', alignment='vertical')
    ax.axis('off')
    ax.set_title('Correct correspondences')

def plot_hist(vec, title='No title', labels=None):
    "vec : flowvec magnitude (ndarray n*1)"
    plt.figure(title)
    if labels != None:
        plt.hist(vec, bins=100, label=labels)
        plt.legend()
    else :
        plt.hist(vec, bins=100)
        
    plt.xlabel('flow_vec_magnitude')
    plt.ylabel('freq')
    if title!=None:
        plt.title(title)
    plt.savefig('./histPlots/'+title)
    plt.show()
    
def plot_bar_hist(partition, histo,  labels=None, title="No Title"):
    plt.figure("Histogram of number of points in each tile")    
    x = np.arange(partition)
    plt.bar(x, histo[0,:])
    plt.title("Histogram of number of points in each tile")
    ti = []
    for i in range(partition):
        ti.append("tile"+str(i+1))
    ti = tuple(ti)
    plt.xticks(x,ti)
    plt.legend()
    plt.savefig('./histPlots/'+title)
    plt.show()
    
def hist_notile_longw_img_two_p(img, points, partition):
    """ this function only plot the histogram based on no tiling considering points
    in 2 different partition (however they are extracted on one tile/partition)"""
    p_hist_h = np.zeros((1,2))
    h, w = img.shape
    counter = 0
    for i in range(points.shape[0]):
        if points[i,1]<w/2:
            counter = counter+1    
    p_hist_h[0,0] = counter
    p_hist_h[0,1] = points.shape[0] - counter
    plot_bar_hist(partition, p_hist_h)
    
def hist_notile_long_h_img():
    pass
def fill_p_hist_8(points, h, w):
    
    p_hist_h = np.zeros((1,8))
    for i in range(points.shape[0]):
        if points[i,1]<w/4 and points[i,0]<h/2:
            p_hist_h[0,0]=p_hist_h[0,0]+1
            
        if points[i,1]<w/4 and points[i,0]>h/2:
            p_hist_h[0,1]=p_hist_h[0,1]+1
            
        if points[i,1]<w/2 and points[i,1]>w/4 and points[i,0]<h/2:
            p_hist_h[0,2]=p_hist_h[0,2]+1
            
        if points[i,1]<w/2 and points[i,1]>w/4 and points[i,0]>h/2:
            p_hist_h[0,3]=p_hist_h[0,3]+1
            
        if points[i,1]<3*(w/4) and points[i,1]>w/2 and points[i,0]<h/2:
            p_hist_h[0,4]=p_hist_h[0,4]+1
            
        if points[i,1]<3*(w/4) and points[i,1]>w/2 and points[i,0]>h/2:
            p_hist_h[0,5]=p_hist_h[0,5]+1
            
        if points[i,1]>3*(w/4) and points[i,0]<h/2:
            p_hist_h[0,6]=p_hist_h[0,6]+1
            
        if points[i,1]>3*(w/4) and points[i,0]>h/2:
            p_hist_h[0,7]=p_hist_h[0,7]+1
            
    return p_hist_h

def plot_hist_all_detectors_kerman(img, pointsh, 
                                   pointsf,
                                   pointsshi,
                                   pointsfa, tiling=False):
    """this is for 8 partitions on kerman data for 4 points detectors
       input img only matters if the image becomes square or alongated 
       Note: look into tiling of new image in tiling technique to make sure order of tiling        
    """
    """    
           pointsh : points detected by harris
           pointsf : points detected by foestner
           pointsshi : points detected by shi_tomasi
           pointsfa : points detected by fast 
    """
    partition = 8
    h, w = img.shape
    if tiling==False:    
        p_hist_harris = np.zeros((1,8))
        p_hist_foe = np.zeros((1,8))
        p_hist_shi = np.zeros((1,8))
        p_hist_fast = np.zeros((1,8))
                
        p_hist_harris=fill_p_hist_8(pointsh, h, w)
        p_hist_foe=fill_p_hist_8(pointsf, h, w)
        p_hist_shi=fill_p_hist_8(pointsshi, h, w)
        p_hist_fast=fill_p_hist_8(pointsfa, h, w)
        title_fig = "Histogram of detected points on Kerman case study without tiling"
    else:
        # in case of having tilied image histogram of num of points are passed to the image (1*8)
        p_hist_harris = pointsh
        p_hist_foe = pointsf
        p_hist_shi = pointsshi
        p_hist_fast = pointsfa
        title_fig = "Histogram of detected points on Kerman case study with tiling"
    
    ti = []
    for i in range(partition):
        ti.append("tile "+str(i+1))
    ti = tuple(ti)
    x = ti #np.arange(partition)
            
    fig, ((ax1,ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, num="Histogram of detected points on Kerman case study without tiling")
    fig.suptitle(title_fig, fontsize=14)

    
    ax1.bar(x, p_hist_harris[0,:])
    ax1.set(title="Harris Point Detector")#, xticks=(x,ti))
    ax1.set_xticks(x,ti)
    
    ax2.bar(x, p_hist_foe[0,:])
    ax2.set(title="Foestner Point Detector")
    ax2.set_xticks(x, ti)
    
    ax3.bar(x, p_hist_shi[0,:])
    ax3.set(title="Shi-Tomasi Point Detector")
    ax3.set_xticks(x, ti)
    
    ax4.bar(x, p_hist_fast[0,:])
    ax4.set(title="Fast Point Detector")
    ax4.set_xticks(x, ti)
    plt.savefig('./histPlots/'+"hist_all_detectors")
    plt.show()

def scatter_flow_vec(img, fv, position, dirc, path="None"):
    title_fig = "Offset map of the study area " + path
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, num=title_fig)
    fig.suptitle(title_fig, fontsize=14)
    plt.imshow(img, interpolation='nearest', cmap=cm.gray)
    plt.axis('tight')

    cax1=ax1.scatter(position[:,0], position[:,1], c=fv, cmap=cm.jet)
    ax1.figure.colorbar(cax1)
    if dirc=="range":
        ax1.set(title=title_fig + "in range direction")
    else:
       ax1.set(title=title_fig + "in azimoth direction")
#    cax2=ax2.scatter(position[:,0], position[:,1], c=fv[:,1], cmap=cm.jet)
#    ax2.set(title="Harris Point Detector")
#    ax2.figure.colorbar(cax2)
#    ax2.set(title=title_fig + "in range direction")
    
    plt.show()
 
    
# following function is taken from Image Analysis I 
# Institute of photogrametry and Informatics - leibniz university of hannover
def imshow3D(*I):
    """Shows the array representation of one or more images in a jupyter notebook.

        Parameters
        ----------
        I : ndarray of float64
            Array representation of an image
            Concatenates multiple images

        Returns
        -------
        out : none

        Notes
        -----
        The given array must have 3 dimensions,
        where the length of the last dimension is either 1 or 3.
    """

    if len(I) == 1:
        I = I[0]
    else:
        channels = [i.shape[2] for i in I]
        heights = [i.shape[0] for i in I]
        max_height = max(heights)
        max_channels = max(channels)

        if min(channels) != max_channels:  # if one image has three channels ..
            I = list(I)
            for i in range(len(I)):
                dim = channels[i]
                if dim == 1:  # .. and another has one channel ..
                    I[i] = np.dstack((I[i], I[i], I[i]))  # .. expand that image to three channels!

        if min(heights) != max_height:  # if heights of some images differ ..
            I = list(I)
            for i in range(len(I)):
                h, w, d = I[i].shape
                if h < max_height:  # .. expand by 'white' rows!
                    I_expanded = np.ones((max_height, w, d), dtype=np.float64) * 255
                    I_expanded[:h, :, :] = I[i]
                    I[i] = I_expanded

        seperator = np.ones((max_height, 3, max_channels), dtype=np.float64) * 255
        seperator[:, 1, :] *= 0
        I_sep = []
        for i in range(len(I)):
            I_sep.append(I[i])
            if i < (len(I) - 1):
                I_sep.append(seperator)
        I = np.hstack(I_sep)  # stack all images horizontally

    assert I.ndim == 3
    h, w, d = I.shape
    assert d in {1, 3}
    if d == 1:
        I = I.reshape(h, w)
    IPython.display.display(Image.fromarray(I.astype(np.ubyte)))
