from time import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy as sc

from helper_functions import two_field_divergence
from wing_class import Wing


class System:
    # -------------
    # System setup
    # -------------

    def __init__(self, X_len, Y_len, nx, ny, rho, nu, umax):
        """X_len, Y_len: domain size (m)
        nx, ny: grid points
        rho: density (kg/m^3)
        nu: kinematic viscosity (m^2/s)"""

        self.X_len, self.Y_len = X_len, Y_len

        # discretization
        self.nx, self.ny = nx, ny
        self.dx = self.X_len / self.nx
        self.dy = self.Y_len / self.ny
        self.cv_vol = self.dx * self.dy
        self.dt = 0.001 # initialized to constant, although adjusted dynamically throughout run
        self.t_elapsed = 0.0
        self.step_num = 0

        # fluid properties
        self.rho, self.nu = rho, nu
        self.cv_mass = self.rho * self.cv_vol

        # coordinates
        self.x = np.linspace(0.0, self.X_len, self.nx+2)
        self.y = np.linspace(-self.Y_len / 2, self.Y_len / 2, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

        # fields
        self.u = np.zeros((self.nx+2, self.ny)) # x velocities
        self.v = np.zeros_like(self.u) # y velocities
        self.p = np.zeros_like(self.u) # pressures
        self.ax = np.zeros_like(self.u) # x body acceleration
        self.ay = np.zeros_like(self.u) # y body acceleration

        # Speed ramping
        self.umax = umax
        self.ramping = True
        self.unstable = False

        # pressure solving
        self.has_jacobi_matrix = False
        self.jacobi_matrix = None

        # obstacle definition
        self.has_wing = False
        self.Wing = None
        self.wing_mask = None
        self.boundary_mask = None
        self.simple_points = None
        self.has_influence_sections = False
        self.wing_dS = None
        self.unit_tangents = None


    def set_ics(self, u, v, p):
        self.u[1:-1] = u
        self.u[0] = u[0] # Equal pad of 1 on left and right to simplify BC implementation
        self.u[-1] = u[-1]

        self.v[1:-1] = v
        self.v[0] = v[0]
        self.v[-1] = v[-1]

        self.p[1:-1] = p
        self.p[0] = p[0]
        self.p[-1] = p[-1]


    def set_body_accelerations(self, ax, ay):
        self.ax = ax
        self.ay = ay


    def add_wing(self, **kwargs):
        self.has_wing = True
        self.Wing = Wing(**kwargs)
        self.wing_mask = self.Wing.interior_mask(self.X, self.Y)
        self.boundary_mask = self.Wing.grid_boundary(self.X, self.Y)
        self.simple_points = ~(self.wing_mask | self.boundary_mask)


    # ------------------
    # Velocity Stepping
    # ------------------

    def motion_step(self):
        """Provisional velocity update (advection + pressure grad + diffusion + body force)."""

        if np.mean(self.u[self.simple_points]) > self.umax:
            self.ramping=False

        # variable time stepping
        max_vel = np.max(np.hypot(self.u, self.v))
        if max_vel <= 1e-10:
            max_vel = 1e-10

        self.dt = 0.3 * np.min((self.dx**2 / self.nu, self.dx / max_vel))
        if max_vel > 20 * self.umax:
            self.unstable = True
            print('WARNING: Max velocity exceeds 20 * intended average final velocity')

        self.t_elapsed += self.dt

        u_star = np.empty_like(self.u)
        v_star = np.empty_like(self.u)

        # Explicit update (mixed: upwind-x for u,v in x; centered in y)

        # centers
        u_star[1:-1, 1:-1] = self.u[1:-1, 1:-1] - self.dt * (
            self.u[1:-1, 1:-1] * (self.u[1:-1, 1:-1] - self.u[:-2, 1:-1]) / (self.dx)
            + self.v[1:-1, 1:-1] * (self.u[1:-1, 2:] - self.u[1:-1, :-2]) / (2 * self.dy)
            + (self.p[2:, 1:-1] - self.p[:-2, 1:-1]) / (self.rho * 2 * self.dx)
            - self.nu * (self.u[2:, 1:-1] - 2 * self.u[1:-1, 1:-1] + self.u[:-2, 1:-1]) / (self.dx ** 2)
            - self.nu * (self.u[1:-1, 2:] - 2 * self.u[1:-1, 1:-1] + self.u[1:-1, :-2]) / (self.dy ** 2)
            - self.ax[1:-1, 1:-1])

        v_star[1:-1, 1:-1] = self.v[1:-1, 1:-1] - self.dt * (
            self.u[1:-1, 1:-1] * (self.v[1:-1, 1:-1] - self.v[:-2, 1:-1]) / (self.dx)
            + self.v[1:-1, 1:-1] * (self.v[1:-1, 2:] - self.v[1:-1, :-2]) / (2 * self.dy)
            + (self.p[1:-1, 2:] - self.p[1:-1, :-2]) / (self.rho * 2 * self.dy)
            - self.nu * (self.v[2:, 1:-1] - 2 * self.v[1:-1, 1:-1] + self.v[:-2, 1:-1]) / (self.dx ** 2)
            - self.nu * (self.v[1:-1, 2:] - 2 * self.v[1:-1, 1:-1] + self.v[1:-1, :-2]) / (self.dy ** 2)
            - self.ay[1:-1, 1:-1])

        # edges
        u_star[1:-1, 0] = self.u[1:-1, 0] - self.dt * (
            self.u[1:-1, 0] * (self.u[1:-1, 0] - self.u[:-2, 0]) / (self.dx)
            + self.v[1:-1, 0] * (self.u[1:-1, 1] - self.u[1:-1, -1]) / (2 * self.dy)
            + (self.p[2:, 0] - self.p[:-2, 0]) / (self.rho * 2 * self.dx)
            - self.nu * (self.u[2:, 0] - 2 * self.u[1:-1, 0] + self.u[:-2, 0]) / (self.dx ** 2)
            - self.nu * (self.u[1:-1, 1] - 2 * self.u[1:-1, 0] + self.u[1:-1, -1]) / (self.dy ** 2)
            - self.ax[1:-1, 0])

        u_star[1:-1, -1] = self.u[1:-1, -1] - self.dt * (
            self.u[1:-1, -1] * (self.u[1:-1, -1] - self.u[:-2, -1]) / (self.dx)
            + self.v[1:-1, -1] * (self.u[1:-1, 0] - self.u[1:-1, -2]) / (2 * self.dy)
            + (self.p[2:, -1] - self.p[:-2, -1]) / (self.rho * 2 * self.dx)
            - self.nu * (self.u[2:, -1] - 2 * self.u[1:-1, -1] + self.u[:-2, -1]) / (self.dx ** 2)
            - self.nu * (self.u[1:-1, 0] - 2 * self.u[1:-1, -1] + self.u[1:-1, -2]) / (self.dy ** 2)
            - self.ax[1:-1, -1])

        v_star[1:-1, 0] = self.v[1:-1, 0] - self.dt * (
            self.u[1:-1, 0] * (self.v[1:-1, 0] - self.v[:-2, 0]) / (self.dx)
            + self.v[1:-1, 0] * (self.v[1:-1, 1] - self.v[1:-1, -1]) / (2 * self.dy)
            + (self.p[1:-1, 1] - self.p[1:-1, -1]) / (self.rho * 2 * self.dy)
            - self.nu * (self.v[2:, 0] - 2 * self.v[1:-1, 0] + self.v[:-2, 0]) / (self.dx ** 2)
            - self.nu * (self.v[1:-1, 1] - 2 * self.v[1:-1, 0] + self.v[1:-1, -1]) / (self.dy ** 2)
            - self.ay[1:-1, 0])

        v_star[1:-1, -1] = self.v[1:-1, -1] - self.dt * (
            self.u[1:-1, -1] * (self.v[1:-1, -1] - self.v[:-2, -1]) / (self.dx)
            + self.v[1:-1, -1] * (self.v[1:-1, 0] - self.v[1:-1, -2]) / (2 * self.dy)
            + (self.p[1:-1, 0] - self.p[1:-1, -2]) / (self.rho * 2 * self.dy)
            - self.nu * (self.v[2:, -1] - 2 * self.v[1:-1, -1] + self.v[:-2, -1]) / (self.dx ** 2)
            - self.nu * (self.v[1:-1, 0] - 2 * self.v[1:-1, -1] + self.v[1:-1, -2]) / (self.dy ** 2)
            - self.ay[1:-1, -1])

        # update wrap padding
        u_star[0] = u_star[-2]
        u_star[-1] = u_star[1]
        v_star[0] = v_star[-2]
        v_star[-1] = v_star[1]

        if self.has_wing:
            # No-slip on boundary ring & interior
            u_star[self.boundary_mask] = 0.0
            v_star[self.boundary_mask] = 0.0
            u_star[self.wing_mask] = 0.0
            v_star[self.wing_mask] = 0.0

        # Commit provisional velocities (pressure correction happens after pressure solve)
        self.u = u_star
        self.v = v_star

        self.step_num += 1 # Advance step count

        # Sanity
        a = np.sum(self.u); b = np.sum(self.v)
        assert not any((np.isnan(a), np.isnan(b))), f"NaN in velocity at step {self.step_num}"
        assert not any((np.isinf(a), np.isneginf(a), np.isinf(b), np.isneginf(b))), \
            f"Inf in velocity at step {self.step_num}"


    # -----------------
    # Pressure Solving
    # -----------------

    def build_jacobi_step_matrix(self, verbose=False):
        """Builds a matrix representation that takes in a flattened vector of the current pressure iteration
        and applies one Jacobi step involving the four spatial neighbors. Matrix is very sparse, so it is
        implemented using scipy.sparse. Already takes care of BCs: Neumann on wing, periodic in y, equal pad in x"""

        diff_sum = self.dx ** 2 + self.dy ** 2

        nx = self.u.shape[0]
        ny = self.u.shape[1]

        c = sc.sparse.lil_array((nx*ny, nx*ny))

        center_mask = np.zeros_like(self.u, dtype=bool)
        bottom_row = np.zeros_like(self.u, dtype=bool)
        top_row = np.zeros_like(self.u, dtype=bool)

        center_mask[1:-1, 1:-1] = 1
        bottom_row[1:-1, 0] = 1
        top_row[1:-1, -1] = 1

        start = time()
        for i in range(nx):
            for j in range(ny):
                ind = i * ny + j
                ip_ind = ind + ny
                im_ind = ind - ny
                jp_ind = ind + 1
                jm_ind = ind - 1

                if center_mask[i, j]:
                    if not self.wing_mask[i, j]:
                        right = self.dy  ** 2 / (2 * diff_sum)
                        left = self.dy  ** 2 / (2 * diff_sum)
                        top = self.dx  ** 2 / (2 * diff_sum)
                        bottom = self.dx  ** 2 / (2 * diff_sum)
                        center = 0

                        if self.wing_mask[i, j-1]:
                            center += bottom
                            bottom = 0
                        if self.wing_mask[i, j+1]:
                            center += top
                            top = 0
                        if self.wing_mask[i-1, j]:
                            center += left
                            left = 0
                        if self.wing_mask[i+1, j]:
                            center += right
                            right = 0

                        c[ind, ip_ind] = right
                        c[ind, im_ind] = left
                        c[ind, jp_ind] = top
                        c[ind, jm_ind] = bottom
                        c[ind, ind] = center

                elif bottom_row[i, j]:
                    c[ind, jp_ind] = self.dx  ** 2 / (2 * diff_sum) # top neighbor
                    c[ind, jm_ind + ny] = self.dx  ** 2 / (2 * diff_sum) # bottom neighbor (periodic)

                    c[ind, ip_ind] = self.dy  ** 2 / (2 * diff_sum)
                    c[ind, im_ind] = self.dy  ** 2 / (2 * diff_sum)

                elif top_row[i, j]:
                    c[ind, jp_ind - ny] = self.dx  ** 2 / (2 * diff_sum)  # top neighbor (periodic)
                    c[ind, jm_ind] = self.dx  ** 2 / (2 * diff_sum)  # bottom neighbor

                    c[ind, ip_ind] = self.dy  ** 2 / (2 * diff_sum)
                    c[ind, im_ind] = self.dy  ** 2 / (2 * diff_sum)

                elif i == 0:
                    right = self.dy ** 2 / (2 * diff_sum)
                    left = self.dy ** 2 / (2 * diff_sum)
                    top = self.dx ** 2 / (2 * diff_sum)
                    bottom = self.dx ** 2 / (2 * diff_sum)

                    c[ind, ip_ind] = right
                    c[ind, ind + (nx-2)*ny] = left # periodic left

                    if j == 0:
                        c[ind, jm_ind + ny] = bottom
                    else:
                        c[ind, jm_ind] = bottom

                    if j == self.ny-1:
                        c[ind, jp_ind - ny] = top
                    else:
                        c[ind, jp_ind] = top

                elif i == nx-1:
                    left = self.dy ** 2 / (2 * diff_sum)
                    right = self.dy ** 2 / (2 * diff_sum)
                    top = self.dx ** 2 / (2 * diff_sum)
                    bottom = self.dx ** 2 / (2 * diff_sum)

                    c[ind, im_ind] = left
                    c[ind, ind - (nx-2)*ny] = right

                    if j == 0:
                        c[ind, jm_ind + ny] = bottom
                    else:
                        c[ind, jm_ind] = bottom

                    if j == self.ny-1:
                        c[ind, jp_ind - ny] = top
                    else:
                        c[ind, jp_ind] = top

        end = time()
        self.jacobi_matrix = c.tocsr()
        self.has_jacobi_matrix = True
        if verbose:
            print(f"Jacobi matrix too {end - start:.2f} seconds to build")
            print(f"Jacobi stepping matrix takes {(self.jacobi_matrix.data.nbytes + self.jacobi_matrix.indptr.nbytes + self.jacobi_matrix.indices.nbytes)/2**20:.3f} Mb")


    def pressure_poisson_RHS(self):
        """We rearrange the Navier Stokes and discretize the velocity time derivative. Taking the divergence
        of the resulting vector equation and assuming ∇·u(n+1) = 0, we obtain a Poisson equation for the pressure
        field. This function builds the constant right hand side (RHS) to fit the pressure Poisson field to"""

        star_x = np.empty_like(self.u)
        star_y = np.empty_like(self.u)

        # Building components of the vector equation rearranged for u_n+1

        # central block
        star_x[1:-1, 1:-1] = self.u[1:-1, 1:-1] / self.dt - (
            self.u[1:-1, 1:-1] * (self.u[1:-1, 1:-1] - self.u[:-2, 1:-1]) / (self.dx)
            + self.v[1:-1, 1:-1] * (self.u[1:-1, 2:] - self.u[1:-1, :-2]) / (2 * self.dy)
            - self.nu * ((self.u[2:, 1:-1] - 2 * self.u[1:-1, 1:-1] + self.u[:-2, 1:-1]) / (self.dx ** 2))
            - self.nu * ((self.u[1:-1, 2:] - 2 * self.u[1:-1, 1:-1] + self.u[1:-1, :-2]) / (self.dy ** 2))
            - self.ax[1:-1, 1:-1])

        star_y[1:-1, 1:-1] = self.v[1:-1, 1:-1] / self.dt - (
            self.u[1:-1, 1:-1] * (self.v[1:-1, 1:-1] - self.v[:-2, 1:-1]) / (self.dx)
            + self.v[1:-1, 1:-1] * (self.v[1:-1, 2:] - self.v[1:-1, :-2]) / (2 * self.dy)
            - self.nu * ((self.v[2:, 1:-1] - 2 * self.v[1:-1, 1:-1] + self.v[:-2, 1:-1]) / (self.dx ** 2))
            - self.nu * ((self.v[1:-1, 2:] - 2 * self.v[1:-1, 1:-1] + self.v[1:-1, :-2]) / (self.dy ** 2))
            - self.ay[1:-1, 1:-1])

        # edges
        star_x[1:-1, 0] = self.u[1:-1, 0] / self.dt - (
            self.u[1:-1, 0] * (self.u[1:-1, 0] - self.u[:-2, 0]) / (self.dx)
            + self.v[1:-1, 0] * (self.u[1:-1, 1] - self.u[1:-1, -1]) / (2 * self.dy)
            - self.nu * ((self.u[2:, 0] - 2 * self.u[1:-1, 0] + self.u[:-2, 0]) / (self.dx ** 2))
            - self.nu * ((self.u[1:-1, 1] - 2 * self.u[1:-1, 0] + self.u[1:-1, -1]) / (self.dy ** 2))
            - self.ax[1:-1, 0])

        star_x[1:-1, -1] = self.u[1:-1, 0] / self.dt - (
            self.u[1:-1, -1] * (self.u[1:-1, -1] - self.u[:-2, -1]) / (self.dx)
            + self.v[1:-1, -1] * (self.u[1:-1, 0] - self.u[1:-1, -2]) / (2 * self.dy)
            - self.nu * ((self.u[2:, -1] - 2 * self.u[1:-1, -1] + self.u[:-2, -1]) / (self.dx ** 2))
            - self.nu * ((self.u[1:-1, 0] - 2 * self.u[1:-1, -1] + self.u[1:-1, -2]) / (self.dy ** 2))
            - self.ax[1:-1, -1])

        star_y[1:-1, 0] = self.v[1:-1, 0] / self.dt - (
            self.u[1:-1, 0] * (self.v[1:-1, 0] - self.v[:-2, 0]) / (self.dx)
            + self.v[1:-1, 0] * (self.v[1:-1, 1] - self.v[1:-1, -1]) / (2 * self.dy)
            - self.nu * ((self.v[2:, 0] - 2 * self.v[1:-1, 0] + self.v[:-2, 0]) / (self.dx ** 2))
            - self.nu * ((self.v[1:-1, 1] - 2 * self.v[1:-1, 0] + self.v[1:-1, -1]) / (self.dy ** 2))
            - self.ay[1:-1, 0])

        star_y[1:-1, -1] = self.v[1:-1, -1] / self.dt - (
            self.u[1:-1, -1] * (self.v[1:-1, -1] - self.v[:-2, -1]) / (self.dx)
            + self.v[1:-1, -1] * (self.v[1:-1, 0] - self.v[1:-1, -2]) / (2 * self.dy)
            - self.nu * ((self.v[2:, -1] - 2 * self.v[1:-1, -1] + self.v[:-2, -1]) / (self.dx ** 2))
            - self.nu * ((self.v[1:-1, 0] - 2 * self.v[1:-1, -1] + self.v[1:-1, -2]) / (self.dy ** 2))
            - self.ay[1:-1, -1])

        # Wrap pad on left and right
        star_x[0] = star_x[-2]
        star_x[-1] = star_x[1]
        star_y[0] = star_y[-2]
        star_y[-1] = star_y[0]

        assert np.sum(star_x[self.wing_mask]) + np.sum(star_y[self.wing_mask]) < 0.0001, f"Star RHS values in wing growing at step {self.step_num}"

        # Pressure poisson has RHS = rho * divergence of vector components above
        RHS = self.rho * two_field_divergence(star_x, star_y, self.dx, self.dy)

        a = np.sum(RHS)
        assert not np.isnan(a), f"NaN in Poisson RHS at step {self.step_num}"
        assert not any((np.isinf(a), np.isneginf(a))), f"Inf in Poisson RHS at step {self.step_num}"
        return RHS


    def pressure_step(self, tol=1e-7, rel_tol=1e-3, max_iters=1000):
        """
        Jacobi/ω-Jacobi iteration on pressure Poisson equation:
        ∇²p = RHS with homogeneous Neumann at the solid (zero normal gradient).
        Implemented using matrix multiplication to optimize runtime.
        """
        if not self.has_jacobi_matrix:
            raise ValueError('Must call self.build_jacobi_step_matrix() once before attempting to calculate pressures')

        RHS = self.pressure_poisson_RHS()
        p_n = self.p.copy().flatten()
        useable_RHS = np.where(self.wing_mask, 0.0, RHS).flatten() # Enforce 0 RHS in wing. Flatten to match pressures

        diff_prod = self.dx ** 2 * self.dy ** 2
        diff_sum = self.dx ** 2 + self.dy ** 2

        if self.has_wing:
            fluid = ~self.wing_mask
        else:
            fluid = np.ones_like(self.u, dtype=bool)

        it = 0
        while it < max_iters:
            # Jacobi step using matrix representation
            p_new = self.jacobi_matrix @ p_n - (diff_prod / (2 * diff_sum) * useable_RHS)

            max_delta = np.max(np.abs(p_new - p_n))
            pressure_scale = np.percentile(p_new, 95) - np.percentile(p_new, 5)
            if pressure_scale < 1e-10: # safety against divide by 0
                pressure_scale=1e-10

            p_n = p_new
            it += 1

            if max_delta <= tol or max_delta/pressure_scale <= rel_tol: # Convergence check
                break

        p_n = p_n.reshape(self.u.shape)

        # Save pressure, removing pressure drift
        p_n -= np.mean(p_n)
        self.p = p_n

        # Sanity
        a = np.sum(self.p)
        assert not np.isnan(a), f"NaN in pressure at step {self.step_num}"
        assert not any((np.isinf(a), np.isneginf(a))), f"Inf in pressure at step {self.step_num}"

        return it


    # --------------------
    # On-the-fly plotting
    # --------------------

    def plot_pressures(self, contour_style=False, show_plot=False, cmap='viridis'):
        fig, ax = plt.subplots(figsize=(10, 8))
        cf = ax.contourf(self.X, self.Y, self.p, levels=50, alpha=0.9, cmap=cmap)
        cbar = plt.colorbar(cf); cbar.set_label("Pressure")
        if contour_style:
            ax.contour(self.X, self.Y, self.p, colors='k', linewidths=0.5)
        ax.set_aspect('equal')
        if show_plot:
            plt.show()
        return fig


    def plot_velocities(self, vector_style=True, show_plot=False, stride_factor=20,
                        stream_density=(1, 1), color=False, cmap='inferno', save_plot=False):
        sx = max(1, self.nx // stride_factor)
        sy = max(1, self.ny // stride_factor)

        Xp, Yp = self.X[1:-1][::sx, ::sy], self.Y[1:-1][::sx, ::sy]
        up, vp = self.u[1:-1][::sx, ::sy], self.v[1:-1][::sx, ::sy]

        fig, ax = plt.subplots(figsize=(10, 8))
        if color:
            C = np.hypot(up, vp)
            if vector_style:
                q = ax.quiver(Xp, Yp, up, vp, C, cmap=cmap)
                plt.colorbar(q, ax=ax, label='|u|')
            else:
                ax.streamplot(Xp, Yp, up, vp, density=stream_density, color=C, cmap=cmap)
        else:
            if vector_style:
                ax.quiver(Xp, Yp, up, vp)
            else:
                ax.streamplot(Xp, Yp, up, vp, density=stream_density, color='k')

        if self.has_wing:
            path = mpl.path.Path(np.array(self.Wing.all_points))
            patch = mpl.patches.PathPatch(path, facecolor='blue', lw=1, alpha=0.7)
            ax.add_patch(patch)
        ax.set_aspect('equal')

        if show_plot:
            plt.show()
        if save_plot:
            plt.savefig(f"./Plots/Velocities_t{self.step_num:05d}.png", dpi=150)
        return fig


    def plot_state(self, p_cmap='bwr', stride_factor=20, stream_density=(1, 1),
                   vector_style=True, contour_style=False, show_plot=False, save_plot=False, fname=None):
        sx = max(1, self.nx // stride_factor)
        sy = max(1, self.ny // stride_factor)

        Xp, Yp = self.X[1:-1][::sx, ::sy], self.Y[1:-1][::sx, ::sy]
        up, vp = self.u[1:-1][::sx, ::sy], self.v[1:-1][::sx, ::sy]

        fig, ax = plt.subplots(figsize=(10, 8))
        cf = ax.contourf(self.X[1:-1], self.Y[1:-1], self.p[1:-1], levels=50, alpha=0.5, cmap=p_cmap)
        if contour_style:
            ax.contour(self.X[1:-1], self.Y[1:-1], self.p[1:-1], colors='k', linewidths=0.5)
        cbar = plt.colorbar(cf, ax=ax)

        if vector_style:
            ax.quiver(Xp, Yp, up, vp)
        else:
            ax.streamplot(Xp, Yp, up, vp, density=stream_density, color='black')

        if self.has_wing:
            path = mpl.path.Path(np.vstack((self.Wing.all_points, [self.Wing.x_start, 0.0])))
            patch = mpl.patches.PathPatch(path, facecolor='grey', lw=1, alpha=0.7)
            ax.add_patch(patch)

        ax.set_aspect('equal')

        if save_plot:
            if not fname:
                fname = f"./Plots/State_t{self.step_num:05d}.png"
            plt.savefig(fname, dpi=300)
        if show_plot:
            plt.show()
        plt.close(fig)
        return fig


    def plot_wing(self, grid=False, show_plot=False, save_plot=False, fname=None):
        """Visualize wing shape on the grid we use"""
        if not self.has_wing:
            raise ValueError('System must have a wing to plot. Call self.add_wing() to add a wing')

        fig, ax = plt.subplots(figsize=(9, 6))

        if grid:
            ax.scatter(self.X[1:-1], self.Y[1:-1], s=5, c='grey', alpha=0.5)
        path = mpl.path.Path(np.vstack((self.Wing.all_points, [self.Wing.x_start, 0.0])))
        patch = mpl.patches.PathPatch(path, facecolor='grey', lw=1, alpha=0.5)
        ax.add_patch(patch)

        ax.set_aspect('equal')
        ax.set_xlim(0, self.X_len)
        ax.set_ylim(-self.Y_len / 2, self.Y_len / 2)

        if show_plot:
            plt.show()
        if save_plot:
            if not fname:
                fname = "wing_shape.png"
            plt.savefig(fname, dpi=300)

        return fig

    # ---------------------------
    # Force on wing calculations
    # ---------------------------

    def set_influence_sections(self):
        dS, unit_tangents = self.Wing.influence_sections(self.X, self.Y)
        self.has_influence_sections = True
        self.wing_dS = dS
        self.unit_tangents = unit_tangents


    def compute_forces(self):
        if not self.has_influence_sections:
            raise ValueError('Must call self.set_influence_sections() once before attempting to calculate forces')

        pressure_forces = self.p[self.boundary_mask] * self.wing_dS * self.unit_tangents
        total_force = np.sum(pressure_forces, axis=1)

        return total_force # (drag, lift)


    # --------
    # Metrics
    # --------

    def system_ke(self):
        return 0.5 * np.sum(self.cv_mass * (self.u ** 2 + self.v ** 2))


    def system_momentum(self):
        return np.sum(self.cv_mass * self.u), np.sum(self.cv_mass * self.v)


    # -------------
    # State saving
    # -------------

    def save_state(self, fname=False):
        """Saves all details needed to reconstruct and plot the current state and obstacle"""

        if not fname:
            fname = f"State_save_t{self.step_num}.npz"

        np.savez(fname, u=self.u, v=self.v, p=self.p, X=self.X, Y=self.Y, wing_points=self.Wing.all_points)


    def save_final_stats(self, fname=False):
        """Saves the hyperparameters that generated the current run, along with the final force stats at umax"""

        if not fname:
            fname = "Final_stats.txt"

        drag, lift = self.compute_forces()

        message = [f"Final lift:\t{lift}\n", f"Final drag:\t{drag}\n", f"Final LdR: \t{lift / drag}\n",
                   f"Ended on step number {self.step_num} at u-speed {self.umax}\n"
                   f"Sigmoid scaling: {self.Wing.scale}\n",
                   f"Thickness: {self.Wing.thickness}\n",
                   f"Interpolation method: {self.Wing.interpolation_method}\n",
                   f"Inverted: {self.Wing.inverted}\n"]

        if self.Wing.naca_code is None:
            message.append(f"Upper parameters: {self.Wing.upper_height_params}\n")
            message.append(f"Lower parameters: {self.Wing.lower_height_params}\n")

            if self.Wing.boundary_condition is not None:
                message.append(f"Upper boundary condition: {self.Wing.boundary_condition[0]}\n")
                message.append(f"Lower boundary condition: {self.Wing.boundary_condition[1]}\n")
        else:
            message.append(f"Wing defined with naca code {self.Wing.naca_code}\n")

        with open(fname, "w") as f:
            f.writelines(message)
