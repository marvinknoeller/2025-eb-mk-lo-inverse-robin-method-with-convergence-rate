#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 16:35:22 2025

@author: marvinknoller
"""
import dolfinx
from dolfinx import default_scalar_type
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
import numpy
from mpi4py import MPI
from ufl import (SpatialCoordinate, TrialFunction, TestFunction,
                 dx, ds, grad, inner)
import ufl
import numpy as np
from basis_functions import create_fun
from petsc4py import PETSc
import basix.ufl
import types

def project_between_spaces(V_fine, V_coarse, u_fine):
    """
    Computes the projection of u_fine onto V_coarse. The result is u_coarse.

    Parameters
    ----------
    V_fine : fem.functionspace
        function space to project from 
    V_coarse : fem.functionspace
        function space to project onto
    u_fine : dolfinx.fem function
        function to project

    Returns
    -------
    u_coarse : dolfinx.fem function
        projected function on coarse grid.

    """
    degree = 4
    Qe = basix.ufl.quadrature_element(
        V_coarse.mesh.topology.cell_name(), degree=degree)
    V_quadrature = dolfinx.fem.functionspace(V_coarse.mesh, Qe)
    cells = np.arange(V_quadrature.mesh.topology.index_map(V_quadrature.mesh.topology.dim).size_local)
    nmmid = dolfinx.fem.create_interpolation_data(V_quadrature, V_fine, cells) 
    q_func = dolfinx.fem.Function(V_quadrature)
    q_func.interpolate_nonmatching(u_fine, cells, nmmid)

    # Project fine function at quadrature points to coarse grid
    u = ufl.TrialFunction(V_coarse)
    v = ufl.TestFunction(V_coarse)
    a_coarse = ufl.inner(u, v) * ufl.dx
    L_coarse = ufl.inner(q_func, v)*ufl.dx
    problem = dolfinx.fem.petsc.LinearProblem(a_coarse, L_coarse, petsc_options={
        "ksp_type": "gmres",
        "pc_type": 'gamg',
        "ksp_rtol": 1e-14,
        "ksp_atol": 1e-15,
        "ksp_max_it": 10000
        })
    u_coarse = problem.solve()
    return u_coarse


def solve_robin(domain, alpha, fhandle, ghandle, N=10, degree=1, LU = True):
    """
    Compute the approximation to the Robin problem on domain with right hand sides f and g.

    Parameters
    ----------
    domain : dolfinx.mesh
        the discrete mesh defining the discrete [0,1]^2 numerically
    alpha : ufl function
        alpha is the a, which is the Robin function in the left hand side of the boundary condition
    fhandle : types.FunctionType or ufl function
        the function (handle) that defines the right hand side f
    ghandle : types.FunctionType or ufl function
        the function (handle) that defines the right hand side g
    N : int
        number of points that discretize the square [0,1]^2. It is h~1/N. The default is 10.
    degree : int, optional
        The degree of the finite element approximation. The default is 1.
    LU : bool, optional
        do you want to use the LU decomposition or an interative solver? The default is True.

    Returns
    -------
    uh: dolfinx.fem function
        this is u_h^{(a)}, the numerical approximation to u^{(a)}

    """
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = SpatialCoordinate(domain)
    if isinstance(fhandle, types.FunctionType) and fhandle.__name__ == "<lambda>": # if fhandle is a lambda function handle
        f = fhandle(x[0], x[1])
    else:
        f = fhandle
    
    if isinstance(ghandle, types.FunctionType) and ghandle.__name__ == "<lambda>": # if ghandle is a lambda function handle
        g = ghandle(x[0], x[1])
    else:
        g = ghandle
    
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v)) * dx + inner(alpha*u, v) * ds 
    L = -inner(f, v) * dx + inner(g,v)*ds
    if LU == True:
        problem = LinearProblem(a, L, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    else:
        problem = LinearProblem(a, L, petsc_options={
            "ksp_type": "gmres",
            "pc_type": 'gamg',  # or "icc" for Incomplete Cholesky
            "ksp_rtol": 1e-14,
            "ksp_atol": 1e-15,
            "ksp_max_it": 10000
        })
    uh = problem.solve()
    return uh


def evaluate_F(alpha, num_cos, num_sin, N, domain, fhandle, ghandle, q, V, dofs, LU=True):
    """
    evaluate the function F, whose jth component is F_{h,j}(a) = <\phi_j u_h^{(a)}, z_h^{(a)}>_{L^2(\partial \Omega)}

    Parameters
    ----------
    alpha : ufl function
        alpha is the a, which is the Robin function in the left hand side of the boundary condition
    num_cos : int
        number of cosine functions
    num_sin : int
        number of sine functions
    N : int
        number of points that discretize the square [0,1]^2. It is h~1/N
    domain : dolfinx.mesh
        the discrete mesh defining the discrete [0,1]^2 numerically
    q : dolfinx.fem function
        the given data on the known domain \omega, which is q = u^{(\tilde{a})}|_\omega
    V : fem.functionspace
        the finite element space
    dofs : np.array
        the degrees of freedom that belong to \omega within \Omega
    LU : bool
        do you want to use the LU decomposition or an interative solver?

    Returns
    -------
    F : np.array(J)
        the vector containing the entries F_{h,j}
    u_ha : dolfinx.fem function
        this is u_h^{(a)}, the numerical approximation to u^{(a)}
    z_ha : dolfinx.fem function
        this is z_h^{(a)}, the numerical approximation to z^{(a)}

    """
    x = SpatialCoordinate(domain) # the spatial coordinates of the domain
    J = num_cos + num_sin # total number of basis functions
    F = np.zeros(J) # we initialize the vector F
    ''' compute u_h^{(a)} '''
    u_ha = solve_robin(domain = domain, 
                       alpha=alpha,
                       fhandle = fhandle,
                       ghandle = ghandle,
                       N = int(N),
                       degree = 1,
                       LU = LU
                       )
    
    uh_res = fem.Function(V)
    uh_res.x.array[dofs] = u_ha.x.array[dofs]
    uh_res.x.scatter_forward()
    z_ha = solve_robin(domain = domain,
                       alpha=alpha,
                       fhandle = q-uh_res,
                       ghandle = fem.Constant(domain, default_scalar_type(0.0)),
                       N = int(N),
                       degree = 1,
                       LU = LU
                       )
    cos_coeffs = fem.Constant(domain, default_scalar_type(np.zeros(num_cos)))
    sin_coeffs = fem.Constant(domain, default_scalar_type(np.zeros(num_sin)))
    phi_j = create_fun(cos_coeffs, sin_coeffs, x)
    Fform = fem.form(inner(phi_j * u_ha, z_ha)*ds)
    for nn in range(num_cos):
        cos_coeffs.value = 0.0
        sin_coeffs.value = 0.0
        cos_coeffs.value[nn] = 1.0
        Fform_ass = fem.assemble_scalar(Fform)
        F[nn] = domain.comm.allreduce(Fform_ass, op=MPI.SUM)
    
    for nn in range(num_sin):
        cos_coeffs.value = 0.0
        sin_coeffs.value = 0.0
        sin_coeffs.value[nn] = 1.0
        Fform_ass = fem.assemble_scalar(Fform)
        F[nn+num_cos] = domain.comm.allreduce(Fform_ass, op=MPI.SUM)
        
    return F, u_ha, z_ha


def evaluate_DF(alpha, u_ha, z_ha, num_cos, num_sin, domain, V, dofs, LU=True):
    """
    evaluate the derivative DF, whose jth component is 
    F'_{h,j}(a) = <\phi_j \dot{u}_h^{(a)}, z_h^{(a)}>_{L^2(\partial \Omega)} + <\phi_j u_h^{(a)}, \dot{z}_h^{(a)}>_{L^2(\partial \Omega)}

    Parameters
    ----------
    alpha : ufl function
        alpha is the a, which is the Robin function in the left hand side of the boundary condition
    u_ha : dolfinx.fem function
        this is u_h^{(a)}, the numerical approximation to u^{(a)}
    z_ha : dolfinx.fem function
        this is z_h^{(a)}, the numerical approximation to z^{(a)}
    num_cos : int
        number of cosine functions
    num_sin : int
        number of sine functions
    N : int
        number of points that discretize the square [0,1]^2. It is h~1/N
    domain : dolfinx.mesh
        the discrete mesh defining the discrete [0,1]^2 numerically
    q : dolfinx.fem function
        the given data on the known domain \omega, which is q = u^{(\tilde{a})}|_\omega
    V : fem.functionspace
        the finite element space
    dofs : np.array
        the degrees of freedom that belong to \omega within \Omega
    LU : bool
        do you want to use the LU decomposition or an interative solver?

    Returns
    -------
    DF : np.array((J,J))
        the Jacobian matrix corresponding to F, from evaluate_F

    """
    x = SpatialCoordinate(domain) # the spatial coordinates of the domain
    J = num_cos + num_sin # total number of basis functions
    DF = np.zeros((J,J)) # we initialize the Jacobi matrix
    '''  
    the rhs on the boundary of \dot{u} and \dot{z} is -\eta u_h and -\eta z_h,
    respectively. 
    '''
    cos_coeffs_eta = fem.Constant(domain, default_scalar_type(np.zeros(num_cos)))
    sin_coeffs_eta = fem.Constant(domain, default_scalar_type(np.zeros(num_sin)))
    eta = create_fun(cos_coeffs_eta, sin_coeffs_eta, x)
    
    ''' phi_j is a single basis function '''
    cos_coeffs = fem.Constant(domain, default_scalar_type(np.zeros(num_cos)))
    sin_coeffs = fem.Constant(domain, default_scalar_type(np.zeros(num_sin)))
    phi_j = create_fun(cos_coeffs, sin_coeffs, x)
    
    ''' 
    initialize the place holder functions for \dot{u}_h and \dot{z}_h as well as the form
    that will determine later the entries of the matrix.
    It is exactly the derivative of F_{h,j}
    '''
    u_ha_prime_PH = fem.Function(V)
    z_ha_prime_PH = fem.Function(V)
    DFform = fem.form(inner(phi_j * u_ha_prime_PH, z_ha)*ds + inner(phi_j * u_ha, z_ha_prime_PH)*ds)
    
    '''
    this is all for solving the pde. For both problems \dot{u}_h and \dot{z}_h there is the same bilinear form a,
    however, we set up two different right hand sides, due to different f and g.
    '''
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v)) * dx + inner(alpha*u, v) * ds
    f1 = fem.Function(V) # f1 remains zero
    f2 = fem.Function(V)
    g1 = -eta*u_ha
    Luh1 = -inner(f1, v) * dx + inner(g1,v)*ds
    a_compiled = dolfinx.fem.form(a)
    Luh1_compiled = dolfinx.fem.form(Luh1)
    A = fem.petsc.assemble_matrix(a_compiled, bcs=[])
    A.assemble()
    # Create solution functions
    u_ha_prime = A.createVecRight()
    z_ha_prime = A.createVecRight()
    ksp = PETSc.KSP().create(A.comm)
    ksp.setOperators(A)
    ''' 
    we need to write it like this, otherwise the preconditioner or the lu decomposition is not saved (I believe.)
    '''
    if LU == True:
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.setFromOptions()
    else:
        ksp.setType('gmres')
        ksp.setTolerances(rtol=1e-14)
        ksp.getPC().setType('hypre')
        
    g2 = -eta*z_ha
    Luh2 = -inner(f2, v) * dx + inner(g2,v)*ds
    Luh2_compiled = dolfinx.fem.form(Luh2)
    
    for nn in range(num_cos):
        """ 
        Cosine perturbation. This syntax works since everything is a ufl function. Meaning: The eta in the g and accordingly, in the form
        gets updated automatically 
        """
        cos_coeffs_eta.value = 0.0
        cos_coeffs_eta.value[nn] = 1.0
        sin_coeffs_eta.value = 0.0
        
        b = fem.petsc.assemble_vector(Luh1_compiled)
        ksp.solve(b, u_ha_prime)
        # update the place holder
        u_ha_prime_PH.x.array[:] = u_ha_prime.array[:]
        u_ha_prime_PH.x.scatter_forward()
        
        uhprime_res = fem.Function(V)
        uhprime_res.x.array[dofs] = u_ha_prime.array[dofs]
        uhprime_res.x.scatter_forward()
        f2.x.array[:] = -uhprime_res.x.array[:]
        b = fem.petsc.assemble_vector(Luh2_compiled)
        ksp.solve(b, z_ha_prime)
        # update the place holder
        z_ha_prime_PH.x.array[:] = z_ha_prime.array[:]
        z_ha_prime_PH.x.scatter_forward()
        
        for mm1 in range(num_cos):
            """ Here we fill the matrix: The cosine terms"""
            cos_coeffs.value = 0.0
            sin_coeffs.value = 0.0
            cos_coeffs.value[mm1] = 1.0
            DFform_ass = fem.assemble_scalar(DFform)
            DF[mm1,nn] = domain.comm.allreduce(DFform_ass, op=MPI.SUM)
            
        for mm2 in range(num_sin):
            """ Here we fill the matrix: The sine terms"""
            cos_coeffs.value = 0.0
            sin_coeffs.value = 0.0
            sin_coeffs.value[mm2] = 1.0
            DFform_ass = fem.assemble_scalar(DFform)
            DF[mm2+num_cos, nn] = domain.comm.allreduce(DFform_ass, op=MPI.SUM)
            
    for nn in range(num_sin):
        """ 
        Sine perturbation. This syntax works since everything is a ufl function. Meaning: The eta in the g and accordingly, in the form
        gets updated automatically 
        """
        cos_coeffs_eta.value = 0.0
        sin_coeffs_eta.value = 0.0
        sin_coeffs_eta.value[nn] = 1.0
        
        b = fem.petsc.assemble_vector(Luh1_compiled)
        ksp.solve(b, u_ha_prime)
        # update the place holder
        u_ha_prime_PH.x.array[:] = u_ha_prime.array[:]
        u_ha_prime_PH.x.scatter_forward()
        
        uhprime_res = fem.Function(V)
        uhprime_res.x.array[dofs] = u_ha_prime.array[dofs]
        uhprime_res.x.scatter_forward()
        f2.x.array[:] = -uhprime_res.x.array[:]
        b = fem.petsc.assemble_vector(Luh2_compiled)
        ksp.solve(b, z_ha_prime)
        # update the place holder
        z_ha_prime_PH.x.array[:] = z_ha_prime.array[:]  # or .setArray or .interpolate as appropriate
        z_ha_prime_PH.x.scatter_forward()  # sync ghost values for parallel
        for mm1 in range(num_cos):
            """ Here we fill the matrix: The cosine terms"""
            cos_coeffs.value = 0.0
            sin_coeffs.value = 0.0
            cos_coeffs.value[mm1] = 1.0
            DFform_ass = fem.assemble_scalar(DFform)
            DF[mm1,nn+num_cos] = domain.comm.allreduce(DFform_ass, op=MPI.SUM)
            
        for mm2 in range(num_sin):
            """ Here we fill the matrix: The sine terms"""
            cos_coeffs.value = 0.0
            sin_coeffs.value = 0.0
            sin_coeffs.value[mm2] = 1.0
            DFform_ass = fem.assemble_scalar(DFform)
            DF[mm2+num_cos, nn+num_cos] = domain.comm.allreduce(DFform_ass, op=MPI.SUM)
            
    return DF