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
plot_solution = 0
plot_iteration = 0
    
deltavec = np.array([1e-6, 1e-5, 1e-4])

""" The maximal mesh size"""
num_h = 21
hh = np.logspace(-1.0, -3.0, num_h)
Nvec = 1.0/hh * np.sqrt(2)
err_vec = np.zeros(num_h)
Nref = 2000

domain_reference = mesh.create_unit_square(MPI.COMM_WORLD, int(Nref), int(Nref),
                                     dolfinx.mesh.CellType.triangle,
                                     ghost_mode=dolfinx.mesh.GhostMode.shared_facet)

""" Determine the set \omega """
# for V1 : 
center = np.array([[.8, .4], [.8, .2]])
radius = np.array([0.05, 0.1])

def inner_disc(x, tol=1e-13):
    ind_x = np.zeros(x[0].size) #fem.Constant(domain, default_scalar_type(0.0))
    for cc in range(center.shape[1]):
        ind_x += (x[0]-center[0,cc])**2 + (x[1]-center[1,cc])**2 <= radius[cc]**2
    return ind_x

""" boundary value problem for z """
from dolfinx.fem import locate_dofs_geometrical
LU = True
x_ref = SpatialCoordinate(domain_reference)

cos_coeffs = fem.Constant(domain_reference, default_scalar_type(np.array([10.0, 1.0, -.5, 2.0, 1.0, -.5])))
sin_coeffs = fem.Constant(domain_reference, default_scalar_type(np.array([.2, 1.0, -.5, 2.0, 1.0, -.5])))
if plot_iteration == 1:
    xx, vals = create_fun_for_plot(np.array(cos_coeffs.value), np.array(sin_coeffs.value))
alpha_exact = create_fun(cos_coeffs, sin_coeffs, x_ref)
fhandle = lambda x,y : - 10 * x * ufl.exp(ufl.sin(4*np.pi*y))
ghandleref = lambda x,y : fem.Constant(domain_reference, default_scalar_type(0.0))
V_reference = fem.functionspace(domain_reference, ("Lagrange", 2))
u_exact = aux_files.solve_robin(domain=domain_reference,
                                alpha = alpha_exact,  
                                fhandle = fhandle,
                                ghandle = ghandleref,
                                N=int(Nref),
                                degree=2,
                                LU = False)

for delta_ell in range(len(deltavec)):

    for h_ell in range(num_h):
        print('h-Iteration no. '+str(h_ell))
        N = int(Nvec[h_ell])
        """ domain must be defined outside here, otherwise it destroys the uniqueness (of the domain)!"""
        domain = mesh.create_unit_square(MPI.COMM_WORLD, int(N), int(N),
                                             dolfinx.mesh.CellType.triangle,
                                             ghost_mode=dolfinx.mesh.GhostMode.shared_facet)
        
        ghandle = lambda x,y : fem.Constant(domain, default_scalar_type(0.0))
        V = fem.functionspace(domain, ("Lagrange", 1))
        dofs = locate_dofs_geometrical(V, inner_disc)
        x = SpatialCoordinate(domain)
        
        c_coeffs = np.array([10.0, 1.0, -.5, 2.0, 1.0, -.5])
        s_coeffs = np.array([.2, 1.0, -.5, 2.0, 1.0, -.5])
        
        dc_coeffs = -np.arange(0,c_coeffs.size,1.) * c_coeffs
        ds_coeffs = np.arange(1,s_coeffs.size+1,1.) * s_coeffs
        
        dcos_coeffs = fem.Constant(domain, default_scalar_type(dc_coeffs))
        dsin_coeffs = fem.Constant(domain, default_scalar_type(ds_coeffs))
        
        der_alpha_exact = create_fun(dsin_coeffs, dcos_coeffs, x)
        
        """ set q = uh_ex|_\omega"""
        delta = deltavec[delta_ell]
        # Step 1: Get the mesh coordinates of all dofs (for scalar space)
        dof_coords = V.tabulate_dof_coordinates()
        
        # Step 2: Filter for the specific dofs you located
        target_coords = dof_coords[dofs][:,:2]
        delta_fun = fem.Function(V)
    
        # Create f(x) on those coordinates
        zz = np.array([10.0, 0.0]) + 1.0j * np.array([0.0, 10.0])
        for kk in range(target_coords.shape[0]):
            xx = target_coords[kk,:]
            value = (np.real(np.exp(1.0j * np.dot(zz,xx-center[:,0]))) * (np.linalg.norm(xx-center[:,0])<=radius[0]+.1) 
                     + np.real(np.exp(1.0j * np.dot(zz,xx-center[:,1]))) * (np.linalg.norm(xx-center[:,1])<=radius[1]+.1)
                     )
            delta_fun.x.array[dofs[kk]] = value
    
        u_interp = aux_files.project_between_spaces(V_reference, V, u_exact)
        q = fem.Function(V)
        q.x.array[dofs] = u_interp.x.array[dofs] + delta*delta_fun.x.array[dofs]
        q.x.scatter_forward()
        
        def error_infinity(c, s, u_ex):
            # Interpolate exact solution, special handling if exact solution
            # is a ufl expression or a python lambda function
            tol = 1e-12
            bdry = lambda x: (
                np.isclose(x[0], 0.0, atol=tol) |
                np.isclose(x[0], 1.0, atol=tol) |
                np.isclose(x[1], 0.0, atol=tol) |
                np.isclose(x[1], 1.0, atol=tol)
            )
            boundary_dofs = locate_dofs_geometrical(V, bdry)
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
            error_max_local = np.max(np.abs(u_h.x.array[boundary_dofs] - u_ex_V.x.array[boundary_dofs]))
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
            # contour.set_clim(lims)
            cbar_ax = fig.add_axes([.95, 0.15, 0.03, 0.7])
            fig.colorbar(contour, cax=cbar_ax)
            plt.savefig('q.png', bbox_inches='tight')
            
            plt.show()
            
        """ 
        -------------------------------------------------------------------------------------
        End of the direct problem.
        Now the inverse problem starts
        -------------------------------------------------------------------------------------
        """
        
        """ define an initial guess """
        num_cos = 6
        num_sin = 6
        
        cos_coeffs_initial = np.array([2.0])
        sin_coeffs_initial = np.array([-0])
        
        cos_coeffs_ell = np.zeros(num_cos)
        sin_coeffs_ell = np.zeros(num_sin)
        cos_coeffs_ell[:cos_coeffs_initial.size] = cos_coeffs_initial
        sin_coeffs_ell[:sin_coeffs_initial.size] = sin_coeffs_initial
        
        cos_coeffs_ell_ufl = fem.Constant(domain, default_scalar_type(cos_coeffs_ell))
        sin_coeffs_ell_ufl = fem.Constant(domain, default_scalar_type(sin_coeffs_ell))
        
        x_ell = np.concatenate((cos_coeffs_ell, sin_coeffs_ell))
        
        coeff_history = x_ell[:,np.newaxis]
        
        xx, vals_ell = create_fun_for_plot(cos_coeffs_ell, sin_coeffs_ell)
        vals_history = vals_ell[:,np.newaxis]
        if plot_iteration == 1:
            fig, ax = plt.subplots()
            
            plt.plot(xx,vals, color='blue', linewidth=2)
            plt.plot(xx,vals_ell, color='red', linewidth=2)
            plt.show()
            
        x_ell0 = np.zeros(num_cos+num_sin)
        tol = 1e-6
        
        for ell in range(10000):
            if np.linalg.norm(x_ell - x_ell0)<tol:
                err_vec[h_ell] = (error_infinity(cos_coeffs_ell, sin_coeffs_ell,alpha_exact) + 
                                  error_infinity(np.arange(1,sin_coeffs_ell.size+1,1.)*sin_coeffs_ell,
                                                 -np.arange(0,cos_coeffs_ell.size,1.)*cos_coeffs_ell, 
                                                 der_alpha_exact)
                                  )
                print('Final Linfty error '+str(err_vec[h_ell]))
                break
            
            alpha_ell = create_fun(cos_coeffs_ell_ufl, sin_coeffs_ell_ufl, x)
            F_ell, u_ha_ell, z_ha_ell = aux_files.evaluate_F(alpha_ell, num_cos, num_sin, N, domain, fhandle, ghandle, q, V, dofs, LU = LU)
            DF_ell = aux_files.evaluate_DF(alpha_ell, u_ha_ell, z_ha_ell, num_cos, num_sin, domain, V, dofs, LU = LU)
            """ Newton Step V1 """
            update_ell = np.linalg.solve(DF_ell, -F_ell)
            # Armijo search
            f0 = np.linalg.norm(F_ell)
            alpha = .5
            kk = 0
            for kk in range(0,30):
                if kk == 29:
                    break
                cos_coeffs_A = cos_coeffs_ell + alpha**kk*update_ell[:num_cos]
                sin_coeffs_A = sin_coeffs_ell + alpha**kk*update_ell[num_cos:num_cos+num_sin]
                cos_coeffs_ell_ufl_A = fem.Constant(domain, default_scalar_type(cos_coeffs_A))
                sin_coeffs_ell_ufl_A = fem.Constant(domain, default_scalar_type(sin_coeffs_A))
                alpha_A = create_fun(cos_coeffs_ell_ufl_A, sin_coeffs_ell_ufl_A, x)
                F_ell_A, _, _ = aux_files.evaluate_F(alpha_A, num_cos, num_sin, N, domain,fhandle, ghandle, q, V, dofs, LU = LU)
                xx, vals_pos = create_fun_for_plot(cos_coeffs_A, sin_coeffs_A)
                if (np.linalg.norm(F_ell_A)< f0) and (min(vals_pos>0) == True):
                    break
                    
            cos_coeffs_ell += alpha**kk*update_ell[:num_cos]
            sin_coeffs_ell += alpha**kk*update_ell[num_cos:num_cos+num_sin]
            cos_coeffs_ell_ufl.value = cos_coeffs_ell
            sin_coeffs_ell_ufl.value = sin_coeffs_ell
            x_ell0 = x_ell
            x_ell = np.concatenate((cos_coeffs_ell, sin_coeffs_ell))
            coeff_history = np.concatenate((coeff_history, x_ell[:,np.newaxis]),axis=1)
            xx, vals_ell = create_fun_for_plot(cos_coeffs_ell, sin_coeffs_ell)
            vals_history = np.concatenate((vals_history, vals_ell[:,np.newaxis]),axis=1)
            
            if plot_iteration == 1:
                fig, ax = plt.subplots()
                
                plt.plot(xx,vals, color='blue', linewidth=2)
                plt.plot(xx,vals_ell, color='red', linewidth=2)
                plt.show()
                
    # Sample data
    
    # loglog plot
    fig, ax = plt.subplots()
    plt.loglog(hh, err_vec, color='blue', linewidth=2, marker='o', label='$\Vert a_h-\widetilde{a} \Vert_{C^1(\partial\Omega)}$')
    plt.loglog(hh, hh**2*1e3, color='black',linestyle='--', linewidth=2)
    # Labels and grid
    plt.xlabel("h",fontsize=18)
    plt.ylabel("error",fontsize=18)
    #plt.title("error plot",fontsize=18)
    plt.grid(True, which="both", linestyle="--",alpha=.7)
    plt.rcParams['axes.labelsize'] = 14       # Axis label font size
    plt.rcParams['xtick.labelsize'] = 14      # X-axis tick font size
    plt.rcParams['ytick.labelsize'] = 14      # Y-axis tick font size
    # ax.tick_params(axis='both', length=6, width=1.4)
    # Customize major ticks
    ax.tick_params(axis='both', which='major', length=8, width=1.5)
    # Customize minor ticks
    ax.tick_params(axis='both', which='minor', length=5, width=1.2, color='gray')
    plt.legend(fontsize=14, loc='lower right')
    plt.savefig('order_a.eps', bbox_inches='tight')  # Save as EPS file
    plt.show()
    
    from tabulate import tabulate  # for nice table printing
    # Compute EOC values
    eocs = [None]  # First entry has no EOC
    for i in range(1, len(hh)):
        rate = np.log(err_vec[i]/err_vec[i-1]) / np.log(hh[i]/hh[i-1])
        eocs.append(rate)
    
    # Create table
    table = []
    for i in range(len(hh)):
        table.append([hh[i], err_vec[i], eocs[i]])
    
    # Print table
    headers = ["h", "Error", "EOC"]
    print(tabulate(table, headers=headers, floatfmt=".4e"))
    
    from scipy.io import savemat    
    data = {'hh' : hh,
            'err_vec' : err_vec,
            'xx' : xx,
            'vals_history' : vals_history,
            'coeff_history' : coeff_history
            }
    savemat('order_delta'+str(delta)+'.mat',data)
