from time import time
from helper_functions import *
from system_class import System


X_len, Y_len = 2, 1
nx, ny = 800, 400

# Values from standard atmosphere model @ 3000m: https://www.ngdc.noaa.gov/stp/space-weather/online-publications/miscellaneous/us-standard-atmosphere-1976/us-standard-atmosphere_st76-1562_noaa.pdf
air_rho = 0.90925 # ai density, kg/m^3
nu = 1.8630e-5 # air kinematic viscosity, m^2/s

umax = 0.1


best_ldr = 0.0
best_ldr_code = None
wings_to_test = 108
wing_ind = 0

start = time()
for camber in range(1, 8):
    for camber_pos in range(1, 6):
        for thickness in range(10, 45, 5):
            code = f"{camber}{camber_pos}{thickness}"
            xu, yu, xl, yl = generate_NACA4(code, n_points=200)

            # monotonically increasing x indices are needed for interpolation
            if np.sum(np.diff(xu) < 0) == 0 and np.sum(np.diff(xl) < 0) == 0:
                wing_ind += 1
                print(f'Beginning testing of airfoil {code}')
                syst = System(X_len, Y_len, nx, ny, air_rho, nu, umax)

                # Initial fields
                u0 = np.zeros((nx, ny))
                v0 = np.zeros_like(u0)
                p0 = np.zeros_like(u0)

                syst.set_ics(u0, v0, p0)

                # Wing
                syst.add_wing(naca_code=code, naca_points=200, thickness=0.2, interpolation_method='p', scale=False)
                syst.build_jacobi_step_matrix()
                syst.set_influence_sections()

                # body accelerations
                ax = np.zeros_like(syst.u)
                ax[syst.simple_points] = 0.005

                ay = np.zeros_like(syst.v)
                syst.set_body_accelerations(ax, ay)

                while syst.ramping and not syst.unstable:
                    syst.motion_step()
                    _ = syst.pressure_step(tol=1e-7, rel_tol=1e-3, max_iters=1000)

                if syst.unstable:
                    with open('unstable_nacas_second', 'a') as f:
                        f.write(code + '\n')

                else:
                    with open('stable_nacas_second', 'a') as f:
                        f.write(code + '\n')

                    drag, lift = syst.compute_forces()
                    if lift / drag > best_ldr:
                        print(f'\t\t\t\t**** NACA {code} is the current best wing ****')
                        best_ldr = lift / drag
                        best_ldr_code = code

                    syst.save_state(fname=f'./NACA_testing/second_states/Naca_{code}.npz')
                    syst.save_final_stats(fname=f'./NACA_testing/second_stats/Naca_{code}.txt')

                end = time()
                print(f'\t{wing_ind/wings_to_test*100:.1f}% done')
                print(f'\tT elapsed: {better_time(end-start)}')
                print()


print(best_ldr_code)
print(best_ldr)
