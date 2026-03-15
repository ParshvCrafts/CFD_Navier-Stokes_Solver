import numpy as np
from system_class import System
from helper_functions import two_field_divergence
from time import time

# Domain & fluid
X_len, Y_len = 2, 1
nx, ny = 800, 400

air_rho = 0.90925 # air density, kg/m^3
nu = 1.8630e-5 # air kinematic viscosity, m^2/s

umax = 0.1

syst = System(X_len, Y_len, nx, ny, air_rho, nu, umax)
print("Starting run...", flush=True)

# Initial fields
u0 = np.zeros((nx, ny))
v0 = np.zeros_like(u0)
p0 = np.zeros_like(u0)

syst.set_ics(u0, v0, p0)


upper = np.array([0.3, 0.5, 0.25, 0.05, -0.2])
lower = np.array([-0.2, -0.1, -0.15, -0.2, -0.4])

# Wing
# syst.add_wing(naca_code='5315', naca_points=200, thickness=0.2, interpolation_method='p', scale=False)
syst.add_wing(upper_height_params=upper, lower_height_params=lower, interpolation_method='c', scale=True, thickness=1.5)

syst.build_jacobi_step_matrix()
syst.set_influence_sections()

# Body accelerations
ax = np.zeros_like(syst.u)
ax[syst.simple_points] = 0.005
ay = np.zeros_like(syst.v)
syst.set_body_accelerations(ax, ay)


start = time()
while syst.ramping:
    syst.motion_step()
    iters = syst.pressure_step(tol=1e-7, rel_tol=1e-3, max_iters=1000)
    average_x_speed = np.mean(syst.u[syst.simple_points])

    if syst.step_num % 10 == 0 and syst.step_num != 0:
        div_avg = np.mean(np.abs(two_field_divergence(syst.u, syst.v, syst.dx, syst.dy)[syst.simple_points]))
        print(f"Step {syst.step_num:5d} | Jacobi iters: {iters:4d} | <|div u|>={div_avg:.3e} | <|u|>={average_x_speed:.3e} | sim time={syst.t_elapsed:.3f}")
        end = time()
        # syst.plot_state(show_plot=False, save_plot=True, stride_factor=100, p_cmap='bwr')
        print(f"Time elapsed: {(end-start):.2f}")

        if syst.step_num % 500 == 0:
            drag, lift = syst.compute_forces()
            print(f'Force calculations: Lift={lift}, Drag={drag}, L/D={lift/drag:.3f}')


# Saving final details
syst.save_state('test_run.npz')
syst.save_final_stats('test_run.txt')
syst.plot_state(show_plot=True, stride_factor=100, p_cmap='bwr')

