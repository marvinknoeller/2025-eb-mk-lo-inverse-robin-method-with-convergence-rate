"""
@author: marvinknoller
"""
import numpy as np
import dolfinx
from basis_functions import create_fun, create_fun_for_plot, create_fun_interpol
from mpi4py import MPI
from dolfinx import fem, mesh, plot, default_scalar_type
import ufl
from ufl import (Circumradius, FacetNormal, SpatialCoordinate, TrialFunction, TestFunction,
                 div, dx, ds, grad, inner, dot)
import aux_files
import matplotlib.pyplot as plt
import matplotlib.tri as tri

""" 
-------------------------------------------------------------------------------------
We start with constructing the direct problem, which is to compute q = u_h|_{\omega}
-------------------------------------------------------------------------------------
"""

""" Want to see a plot of the solution?"""
plot_solution = 1
plot_iteration = 1

full_cos = [10.0, 1.0, -.5, 2.0, 1.0, -.5, 1.0, 1.0]
full_sin = [.2, 1.0, -.5, 2.0, 1.0, -.5, 1.0, 1.0]
nums = 17
err_vec = np.zeros(nums)
normuaH2 = np.zeros(nums)
const = np.zeros(nums)
for jj in range(4, nums):
    print(jj)
    """ The (approximate) mesh size"""
    N = 2000
    h = 1/N * np.sqrt(2)
    print('Maximal mesh size for reconstruction will be ' + str(h))
    """ domain must be defined outside here, otherwise it destroys the uniqueness (of the domain)!"""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, int(N), int(N),
                                         dolfinx.mesh.CellType.triangle,
                                         ghost_mode=dolfinx.mesh.GhostMode.shared_facet)
    
    domain_reference = mesh.create_unit_square(MPI.COMM_WORLD, int(N), int(N),
                                         dolfinx.mesh.CellType.triangle,
                                         ghost_mode=dolfinx.mesh.GhostMode.shared_facet)
    
    """ Determine the set \omega """
    # for V1 : 
    center = np.array([[.8, .4], [.8, .2]])
    radius = np.array([0.05, 0.1])
    def inner_disc(x, tol=1e-13):
        ind_x = np.zeros(x[0].size)
        for cc in range(center.shape[1]):
            ind_x += (x[0]-center[0,cc])**2 + (x[1]-center[1,cc])**2 <= radius[cc]**2
        return ind_x
    
    tdim = domain.topology.dim
    cell_map = domain.topology.index_map(tdim)
    num_cells = cell_map.size_local + cell_map.num_ghosts
    marker = np.ones(num_cells, dtype=np.int32)
    marker[dolfinx.mesh.locate_entities(domain, tdim, inner_disc)] = 2
    
    cell_tag = dolfinx.mesh.meshtags(domain, tdim, np.arange(num_cells, dtype=np.int32), marker)
    subdx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tag)
    inner_vol = dolfinx.fem.form(1 * subdx(2))
    local_volume = dolfinx.fem.assemble_scalar(inner_vol)

    """ boundary value problem for z """
    from dolfinx.fem import locate_dofs_geometrical
    V = fem.functionspace(domain, ("Lagrange", 1))
    dofs = locate_dofs_geometrical(V, inner_disc)
    LU = True
    x = SpatialCoordinate(domain)
    x_ref = SpatialCoordinate(domain_reference)
    
    cos_coeffs = fem.Constant(domain_reference, default_scalar_type(np.array(full_cos[:jj//2 + np.mod(jj,2)])))
    sin_coeffs = fem.Constant(domain_reference, default_scalar_type(np.array(full_sin[:jj//2])))
    if plot_iteration == 1:
        xx, vals = create_fun_for_plot(np.array(cos_coeffs.value), np.array(sin_coeffs.value))
    alpha_exact = create_fun(cos_coeffs, sin_coeffs, x_ref)
    
    c_coeffs = np.array(full_cos[:jj//2+ np.mod(jj,2)])
    s_coeffs = np.array(full_sin[:jj//2])
    dc_coeffs = np.arange(0,c_coeffs.size,1.) * c_coeffs
    ds_coeffs = np.arange(1,s_coeffs.size+1,1.) * s_coeffs
    
    dcos_coeffs = fem.Constant(domain, default_scalar_type(dc_coeffs))
    dsin_coeffs = fem.Constant(domain, default_scalar_type(ds_coeffs))
    
    der_alpha_exact = create_fun(dcos_coeffs, dsin_coeffs, x)
    fhandle = lambda x,y : - 10 * x * ufl.exp(ufl.sin(4*np.pi*y))
    ghandleref = lambda x,y : fem.Constant(domain_reference, default_scalar_type(0.0))
    ghandle = lambda x,y : fem.Constant(domain, default_scalar_type(0.0))
    V_reference = fem.functionspace(domain_reference, ("Lagrange", 1))
    u_exact = aux_files.solve_robin(domain=domain_reference,
                                    alpha = alpha_exact,  
                                    fhandle = fhandle,
                                    ghandle = ghandleref,
                                    N=int(N), 
                                    degree=2,
                                    LU = False)
    """ set q = uh_ex|_\omega"""
    delta = 0
    pert = np.random.rand(np.size(dofs)) -0.5
    
    u_interp = aux_files.project_between_spaces(V_reference, V, u_exact)
    q = fem.Function(V)
    q.x.array[dofs] = u_interp.x.array[dofs] + delta*np.linalg.norm(u_interp.x.array[dofs]) * pert/np.linalg.norm(pert)
    q.x.scatter_forward()
    
    def error_infinity(c,s, u_ex):
        # Interpolate exact solution, special handling if exact solution
        # is a ufl expression or a python lambda function
        u_h = fem.Function(V)
        u_h.interpolate(lambda x : create_fun_interpol(c, s, x))
        comm = u_h.function_space.mesh.comm
        u_ex_V = fem.Function(u_h.function_space)
        if isinstance(u_ex, ufl.core.expr.Expr):
            u_expr = dolfinx.fem.Expression(u_ex, u_h.function_space.element.interpolation_points())
            u_ex_V.interpolate(u_expr)
        else:
            u_ex_V.interpolate(u_ex)
        # Compute infinity norm, furst local to process, then gather the max
        # value over all processes
        error_max_local = np.max(np.abs(u_h.x.array - u_ex_V.x.array))
        error_max = comm.allreduce(error_max_local, op=MPI.MAX)
        return error_max
    
    if plot_solution == 1:
        fig, ax = plt.subplots(figsize=(5, 5))
        xd = domain.geometry.x
        cells = domain.topology.connectivity(domain.topology.dim, 0).array.reshape(-1, 3)
        triangulation = tri.Triangulation(xd[:, 0], xd[:, 1], cells)
        
        contour = plt.tricontourf(triangulation, u_interp.x.array,500, cmap="gnuplot2_r")
        plt.axis("equal")
        plt.rcParams['axes.labelsize'] = 18
        plt.rcParams['xtick.labelsize'] = 18
        plt.rcParams['ytick.labelsize'] = 18
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='both', which='major', length=8, width=1.5)
        # Customize minor ticks
        ax.tick_params(axis='both', which='minor', length=5, width=1.2, color='gray')
        
        cbar_ax = fig.add_axes([.95, 0.15, 0.03, 0.7])
        lims = contour.get_clim()
        fig.colorbar(contour, cax=cbar_ax)
        plt.savefig('exact_solution.png', bbox_inches='tight')
        
        plt.show()
        
        fig, ax = plt.subplots(figsize=(5, 5))
    
        # Create triangulation and plot
        contour = plt.tricontourf(triangulation, q.x.array,500, cmap="gnuplot2_r")
        plt.axis("equal")
        plt.rcParams['axes.labelsize'] = 18
        plt.rcParams['xtick.labelsize'] = 18
        plt.rcParams['ytick.labelsize'] = 18
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='both', which='major', length=8, width=1.5)
        # Customize minor ticks
        ax.tick_params(axis='both', which='minor', length=5, width=1.2, color='gray')
        contour.set_clim(lims)
        cbar_ax = fig.add_axes([.95, 0.15, 0.03, 0.7])
        fig.colorbar(contour, cax=cbar_ax)
        plt.savefig('q.png', bbox_inches='tight')
        
        plt.show()
    
    
    num_cos = jj//2 + np.mod(jj,2)
    num_sin = jj//2
    cos_coeffs = fem.Constant(domain, default_scalar_type(np.array(full_cos[:jj//2 + np.mod(jj,2)])))
    sin_coeffs = fem.Constant(domain, default_scalar_type(np.array(full_sin[:jj//2])))
    alpha_ell = create_fun(cos_coeffs, sin_coeffs, x)
    F_ell, u_ha_ell, z_ha_ell = aux_files.evaluate_F(alpha_ell, num_cos, num_sin, N, domain, fhandle, ghandle, q, V, dofs, LU = LU)
    DF_ell = aux_files.evaluate_DF(alpha_ell, u_ha_ell, z_ha_ell, num_cos, num_sin, domain, V, dofs, LU = LU)
    err_vec[jj] = np.linalg.cond(DF_ell)
    print(err_vec[jj])
    
    from scipy.io import savemat    
    data = {
            'err_vec' : err_vec
            }
    savemat('Example5_'+str(jj)+str(num_cos)+'_'+str(num_sin)+'delta'+str(int(delta*100))+'pz'+'.mat',data)