#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 22:54:16 2025

@author: marvinknoller
"""

import ufl
import numpy as np

tol = 1e-14
def create_fun(cos_coeffs, sin_coeffs, x):
    """
    Creates a linear combination of cosine and sine terms so that it can be used in forms.

    Parameters
    ----------
    cos_coeffs : fem.Constant
        The coefficients corresponding to the cosine terms.
    sin_coeffs : fem.Constant
        The coefficients corresponding to the sine terms.
    x : ufl.SpatialCoordinate
        The spatial coordinates corresponding to some mesh.

    Returns
    -------
    fun : ufl function
        The linear combination of cosines and sines on the spatial coordinates 
        according to the arrays in cos_coeffs and sin_coeffs.

    """
    def classify_side(x, side):
        """ We have to make sure that we do not double the corners!"""
        if side == 'bottom':
            bottom_true = ufl.le(abs(x[1]),tol)
            return bottom_true
        if side == 'right':
            bottom_true = ufl.le(abs(x[1]),tol)
            right_true = ufl.le(abs(x[0]-1.0),tol)
            not_right_and_bottom = ufl.Not(ufl.And(bottom_true,right_true)) # true, unless we hit the bottom right corner
            right_true_without_bottom = ufl.And(right_true, not_right_and_bottom)
            return right_true_without_bottom
        if side == 'top':
            right_true = ufl.le(abs(x[0]-1.0),tol)
            top_true = ufl.le(abs(x[1]-1.0),tol)
            not_top_and_right = ufl.Not(ufl.And(right_true,top_true)) # true, unless we hit the top right corner
            top_true_without_right = ufl.And(top_true, not_top_and_right)
            return top_true_without_right
        if side == 'left':
            top_true = ufl.le(abs(x[1]-1.0),tol)
            bottom_true = ufl.le(abs(x[1]),tol)
            left_true = ufl.le(abs(x[0]-0.0),tol)
            not_left_and_top = ufl.Not(ufl.And(top_true,left_true)) # true, unless we hit the top left corner
            left_true_without_top = ufl.And(left_true, not_left_and_top)
            
            not_bottom_and_left = ufl.Not(ufl.And(left_true,bottom_true)) # true, unless we hit the bottom left corner
            left_true_without_top_and_bottom = ufl.And(left_true_without_top, not_bottom_and_left)
            return left_true_without_top_and_bottom
        
    
    fun = ufl.zero()
    for n in range(cos_coeffs.value.size):#ufl_shape[0]): #.size
        fun += 1.0/2.0 * cos_coeffs[n]*(
            ufl.conditional(classify_side(x,'bottom'), ufl.cos(n*np.pi/2*x[0]), 0.0)
            + ufl.conditional(classify_side(x,'right'), ufl.cos(n*np.pi/2*(x[1]+1)), 0.0)
            + ufl.conditional(classify_side(x,'top'), ufl.cos(n*np.pi/2*(3-x[0])), 0.0)
            + ufl.conditional(classify_side(x,'left'), ufl.cos(n*np.pi/2*(4-x[1])), 0.0)
            )
    for n in range(1, sin_coeffs.value.size+1):
        fun += 1.0/2.0 * sin_coeffs[n-1]*(
            ufl.conditional(classify_side(x,'bottom'), ufl.sin(n*np.pi/2*x[0]), 0.0)
            + ufl.conditional(classify_side(x,'right'), ufl.sin(n*np.pi/2*(x[1]+1)), 0.0)
            + ufl.conditional(classify_side(x,'top'), ufl.sin(n*np.pi/2*(3-x[0])), 0.0)
            + ufl.conditional(classify_side(x,'left'), ufl.sin(n*np.pi/2*(4-x[1])), 0.0)
            )
    return fun

def create_fun_interpol(cos_coeffs, sin_coeffs, x):
    """
    Creates a linear combination of cosine and sine terms so that it can be used with python's lambda function.

    Parameters
    ----------
    cos_coeffs : fem.Constant
        The coefficients corresponding to the cosine terms.
    sin_coeffs : fem.Constant
        The coefficients corresponding to the sine terms.
    x : input variable from lambda
        This is the x from lambda x : ...

    Returns
    -------
    fun : ufl function
        The linear combination of cosines and sines on the spatial coordinates 
        according to the arrays in cos_coeffs and sin_coeffs.

    """
    side = -1 * np.ones(x[0].size)
    side[abs(x[1]) < tol] = 0
    side[abs(x[0] - 1.0) < tol] = 1
    side[abs(x[1]-1.0) < tol] = 2
    side[abs(x[0] - 0.0) < tol] = 3 # now the association is unique, since "double" values are just overwritten!
    fun = ufl.zero()
    for n in range(cos_coeffs.size):
        fun += 1.0/2.0 * cos_coeffs[n]*(np.cos(n*np.pi/2*x[0]) * (side==0)
                                        + np.cos(n*np.pi/2*(x[1]+1)) * (side==1)
                                        + np.cos(n*np.pi/2*(3-x[0])) * (side==2)
                                        + np.cos(n*np.pi/2*(4-x[1])) * (side==3)
                                        )
    for n in range(1, sin_coeffs.size+1):
        fun += 1.0/2.0 * sin_coeffs[n-1]*(np.sin(n*np.pi/2*x[0]) * (side==0)
                                        + np.sin(n*np.pi/2*(x[1]+1)) * (side==1)
                                        + np.sin(n*np.pi/2*(3-x[0])) * (side==2)
                                        + np.sin(n*np.pi/2*(4-x[1])) * (side==3)
                                        )
    return fun


def create_fun_for_plot(cos_coeffs, sin_coeffs):
    """
    Creates a linear combination of cosine and sine terms just for easy plotting with matplotlib

    Parameters
    ----------
    cos_coeffs : np.ndarray
        The coefficients corresponding to the cosine terms.
    sin_coeffs : np.ndarray
        The coefficients corresponding to the sine terms.

    Returns
    -------
    xx : np.ndarray
        x values from 0 to 4
    vals : TYPE
        fun(x), where fun is the combination of cosines and sines
        according to the arrays in cos_coeffs and sin_coeffs.

    """
    nn = 1000
    xx = np.linspace(0,4,nn)
    vals = np.zeros(nn)
    
    for nn in range(cos_coeffs.size):
        vals += 1.0/2.0*cos_coeffs[nn] * np.cos(nn*np.pi/2*xx)
        
    for nn in range(1, sin_coeffs.size+1):
        vals += 1.0/2.0*sin_coeffs[nn-1] * np.sin(nn*np.pi/2*xx)
        
    return xx,vals


