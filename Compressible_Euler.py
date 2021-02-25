#Importing Numpy, Matplotlib, and Math
import numpy as np
import matplotlib.pyplot as plt
import math

#Setting required variables and initial conditions
#n cells along x
#m cells along y
n       = 200
m       = 200

#Setting the domain and range
x_min   = -0.5
x_max   = 0.5

y_min   = -0.5
y_max   = 0.5

#Gamme and CFL
gamma   = 1.4
CFL     = 0.5

#Evaluating delta x and delta y
delta_x = (x_max-x_min)/n
delta_y = (y_max-y_min)/m

#Intializing delta t and the current time
delta_t = 0.0
current_t = 0.0

#Creating arrays for U vector and the temporary Utilda vector
U        = np.zeros((4,n,m))
Utilda   = np.zeros((4,n,m))

#Creating an array for the Flux vector
F        = np.zeros((4,n,m))

#Initializing a list to store x and y values
x        = np.zeros(n)
y        = np.zeros(m)

#Initializing the x domain
for i in range(n):
    x[i] = x_min + (i+0.5)*delta_x

#Initializing the y range
for j in range(m):
    y[j] = y_min + (j+0.5)*delta_y

#Finding the U vector given states in a single cell
def find_U(rho, u, v, p):
    U_now = np.zeros(4)
    U_now[0] = rho
    U_now[1] = rho*u
    U_now[2] = rho*v
    U_now[3] = p/(gamma - 1.0) + 0.5*rho*(u**2.0 + v**2.0)
    return U_now

#Finding the flux vector given states in a single cell
def find_F(rho, u, v, p):
    F_now = np.zeros(4)
    F_now[0] = rho*u
    F_now[1] = rho*u**2.0+p
    F_now[2] = rho*u*v
    F_now[3] = u*(gamma*p/(gamma - 1.0) + 0.5*rho*(u**2.0 + v**2.0))
    return F_now

#Solving for a
def find_a(rho, p):
    return math.sqrt(gamma*p/rho)

#Finding the states in a single cell given the U vector
def find_rho_u_v_p_a(U_state):
    rho = U_state[0]
    u   = U_state[1]/U_state[0]
    v   = U_state[2]/U_state[0]
    p   = (U_state[3] - 0.5*rho*(u**2.0 + v**2.0))*(gamma - 1.0)
    a   = find_a(rho, p)
    return rho, u, v, p, a

def is_realizable(U_state):
    if U_state[0] <= 0.0:
        return False
    
    p = (U_state[3] - 0.5*(U_state[1]**2.0 + U_state[2]**2.0)/(U_state[0]))*(gamma - 1.0)

    if p <= 0:
        return False

    return True

#Rotating the standard U vector
def rotConversionU(U_rot, nx, ny):
    U1_new = U_rot[1]*nx + U_rot[2]*ny
    U2_new = -U_rot[1]*ny + U_rot[2]*nx
    U_rot[1] = U1_new
    U_rot[2] = U2_new
    return U_rot

#Converting the rotated Flux vector back to standard form
def rotConversionF(F_rot, nx, ny):
    F1_new = F_rot[1]*nx - F_rot[2]*ny
    F2_new = F_rot[1]*ny + F_rot[2]*nx
    F_rot[1] = F1_new
    F_rot[2] = F2_new
    return F_rot

#Finding h
def find_h(rho, u, v, p):
    return gamma*p/((gamma - 1.0)*rho) + 0.5*(u**2.0+v**2.0)

#Finding the roe average states
def roe_averageStates(rho_L, u_L, v_L, p_L, a_L, rho_R, u_R, v_R, p_R, a_R):
    h_L   = find_h(rho_L, u_L, v_L, p_L)
    h_R   = find_h(rho_R, u_R, v_R, p_R)
    rho_hat = math.sqrt(rho_L*rho_R)
    u_hat = (u_L*rho_L**0.5 + u_R*rho_R**0.5)/(rho_L**0.5+rho_R**0.5)
    v_hat = (v_L*rho_L**0.5 + v_R*rho_R**0.5)/(rho_L**0.5+rho_R**0.5)
    h_hat = (h_L*rho_L**0.5 + h_R*rho_R**0.5)/(rho_L**0.5+rho_R**0.5)
    p_hat = ((h_hat-0.5*(u_hat**2.0 + v_hat**2.0))*(gamma-1.0)*rho_hat)/gamma
    a_hat = find_a(rho_hat, p_hat)

    return rho_hat, u_hat, h_hat, p_hat, a_hat

#Using HLLE to find the approximate flux
def hlle(rho_L, u_L, v_L, p_L, a_L, rho_R, u_R, v_R, p_R, a_R):
    rho_hat, u_hat, h_hat, p_hat, a_hat = roe_averageStates(rho_L, u_L, v_L, p_L, a_L, rho_R, u_R, v_R, p_R, a_R)

    LM = min(u_L - a_L, u_hat - a_hat)
    LP = max(u_R + a_R, u_hat + a_hat)

    if LM > 0.0:
        Fl = find_F(rho_L, u_L, v_L, p_L)
        return Fl
    elif LP < 0.0:
        Fr = find_F(rho_R, u_R, v_R, p_R)
        return Fr
    else:
        Ul_mat = find_U(rho_L, u_L, v_L, p_L)
        Ur_mat = find_U(rho_R, u_R, v_R, p_R)

        Fl = find_F(rho_L, u_L, v_L, p_L)
        Fr = find_F(rho_R, u_R, v_R, p_R)

        Fm = (LP*Fl-LM*Fr)/(LP-LM)+LP*LM*(Ur_mat-Ul_mat)/(LP-LM)
        return Fm

def get_psi_krishnan(Ul, Ui, Umax, Umin):
    tol = 10e-4
    if Ul > Ui + tol:
        y = (Umax-Ui)/(Ul-Ui)
        psi = (y**2.0+2.0*y)/(y**2.0+y+2.0)
    elif Ul < Ui - tol: 
        y = (Umin-Ui)/(Ul-Ui)
        psi = (y**2.0+2.0*y)/(y**2.0+y+2.0)
    else:
        psi = 1.0

    return psi

def get_psi_minmod(Ul, Ui, Umax, Umin):
    tol = 10e-4
    if Ul > Ui + tol:
        psi = min((Umax-Ui)/(Ul-Ui), 1.0)
    elif Ul < Ui - tol: 
        psi = min((Umin-Ui)/(Ul-Ui), 1.0)
    else:
        psi = 1.0

    return psi

#Finding a(dUdx)*delta_x and b(dUdy)*delta_y
def limiter_of_slopes(Ukata, i, j):

    #Initializing limiter psi aka "phi"
    psi = np.zeros(4) 

    psi_1 = np.zeros(4) 
    psi_2 = np.zeros(4) 
    psi_3 = np.zeros(4) 
    psi_4 = np.zeros(4) 

    Ui = np.zeros(4)
    Ui = Ukata[:, i, j]

    Ul_1 = np.zeros(4)
    Ul_2 = np.zeros(4)
    Ul_3 = np.zeros(4)
    Ul_4 = np.zeros(4)

    #Initializing a (dudx), aR(uR), and aL(uL)
    a   = np.zeros(4)
    aR  = np.zeros(4)
    aL  = np.zeros(4)

    #Initializing b (dudy), bR(uT), and bL(uB)
    b   = np.zeros(4)
    bR  = np.zeros(4)
    bL  = np.zeros(4)

    #Intializing an array for the maximum U and minimum U in the stencil
    Umax = np.zeros(4)
    Umin = np.zeros(4)

    apsi = np.zeros(4)
    bpsi = np.zeros(4)

    #Far left bound cell
    if i == 0:
        #Lower bound cell (Bottom Left "2 ghost neighbours")
        if j == 0:
            aR = Ukata[:, i+1, j]
            aL = Ukata[:, i, j]
            bR = Ukata[:, i, j+1]
            bL = Ukata[:, i, j]
        #Upper bound cell (Top Left "2 ghost neighbours")
        elif j == m-1:
            aR = Ukata[:, i+1, j]
            aL = Ukata[:, i, j]
            bR = Ukata[:, i, j]
            bL = Ukata[:, i, j-1]
        #Otherwise (Far Left "1 ghost neighbour")
        else:
            aR = Ukata[:, i+1, j]
            aL = Ukata[:, i, j]
            bR = Ukata[:, i, j+1]
            bL = Ukata[:, i, j-1]
    #Far right bound cell
    elif i == n-1:
        #Lower bound cell (Bottom right "2 ghost neighbours")
        if j == 0:
            aR = Ukata[:, i, j]
            aL = Ukata[:, i-1, j]
            bR = Ukata[:, i, j+1]
            bL = Ukata[:, i, j]
        #Upper bound cell (Top right "2 ghost neighbours")
        elif j == m-1:
            aR = Ukata[:, i, j]
            aL = Ukata[:, i-1, j]
            bR = Ukata[:, i, j]
            bL = Ukata[:, i, j-1]
        #Otherwise (Far right "1 ghost neighbour")
        else:
            aR = Ukata[:, i, j]
            aL = Ukata[:, i-1, j]
            bR = Ukata[:, i, j+1]
            bL = Ukata[:, i, j-1]
    #Otherwise
    else:
        #Lower bound cell (Lower end "1 ghost neighbour")
        if j == 0:
            aR = Ukata[:, i+1, j]
            aL = Ukata[:, i-1, j]
            bR = Ukata[:, i, j+1]
            bL = Ukata[:, i, j]
        #Upper bound cell (Top end "1 ghost neighbour")
        elif j == m-1:
            aR = Ukata[:, i+1, j]
            aL = Ukata[:, i-1, j]
            bR = Ukata[:, i, j]
            bL = Ukata[:, i, j-1]
        #Otherwise (Away from any bound "0 ghost neighbours")
        else:
            aR = Ukata[:, i+1, j]
            aL = Ukata[:, i-1, j]
            bR = Ukata[:, i, j+1]
            bL = Ukata[:, i, j-1]

    #Getting Ul in each neighbour
    Ul_1 = Ui + a*delta_x/2.0
    Ul_2 = Ui - a*delta_x/2.0
    Ul_3 = Ui + b*delta_y/2.0
    Ul_4 = Ui - b*delta_y/2.0

    #Making our lives simpler...
    for c in range(4):
        #Collecting the maximum of every U entry in the stencil
        Umax[c] = max(aR[c], aL[c], bR[c], bL[c], Ui[c])
        #Collecting the minimum of every U entry in the stencil
        Umin[c] = min(aR[c], aL[c], bR[c], bL[c], Ui[c])

        #Calculating a (dudx) and b (dudy)
        a[c] = (aR[c]-aL[c])/(2.0*delta_x)
        b[c] = (bR[c]-bL[c])/(2.0*delta_y)

        #Obtaining psi values for neighbours (minmod/krishnan)
        psi_1[c] = get_psi_krishnan(Ul_1[c], Ui[c], Umax[c], Umin[c])
        psi_2[c] = get_psi_krishnan(Ul_2[c], Ui[c], Umax[c], Umin[c])
        psi_3[c] = get_psi_krishnan(Ul_3[c], Ui[c], Umax[c], Umin[c])
        psi_4[c] = get_psi_krishnan(Ul_4[c], Ui[c], Umax[c], Umin[c])

        #Keeping minimum psi value
        psi[c] = min(psi_1[c], psi_2[c], psi_3[c], psi_4[c])

        find_rho_u_v_p_a

        apsi[c] = a[c]*psi[c]*(delta_x/2.0)
        bpsi[c] = b[c]*psi[c]*(delta_y/2.0)

    Ul_1 = Ui + apsi
    Ul_2 = Ui - apsi
    Ul_3 = Ui + bpsi
    Ul_4 = Ui - bpsi

    if not is_realizable(Ul_1) or not is_realizable(Ul_2) or not is_realizable(Ul_3) or not is_realizable(Ul_4):
        apsi.fill(0.0)
        bpsi.fill(0.0)

    #Returning the slope limiters in x and y
    return apsi, bpsi

#Function to calculate Flux in X and Flux in Y
def flux_calc(this_U):
    #Calculating Fluxes in X
    for i in range(n+1):
        for j in range(m):
            iL, iR = 0,0

            #Boundary at the far left bound
            if i == 0:
                #Setting the left (ghost) and right index
                iL, iR = i, i 

                #Regular rotation conversion of left and right cells
                UL = rotConversionU(np.copy(this_U[:, iL, j]), 1, 0)
                UR = rotConversionU(np.copy(this_U[:, iR, j]), 1, 0)

            #Boundary at the far right bound
            elif i == n:
                #Setting the left and right (ghost) index
                iL, iR = i-1, i-1

                #Regular rotation conversion of left and right cells
                UL = rotConversionU(np.copy(this_U[:, iL, j]), 1, 0)
                UR = rotConversionU(np.copy(this_U[:, iR, j]), 1, 0)

            else:
                #Setting the left and right index
                iL, iR = i-1, i

                #Regular rotation conversion of left and right cells
                '''UL = rotConversionU(np.copy(this_U[:, iL, j]), 1, 0)
                UR = rotConversionU(np.copy(this_U[:, iR, j]), 1, 0)'''

                #Slope limited rotation conversion of left and right cells
                psia, psib = limiter_of_slopes(this_U, i-1, j)
                UL = rotConversionU(np.copy(this_U[:, iL, j])+psia, 1, 0)
                psia, psib = limiter_of_slopes(this_U, i, j)
                UR = rotConversionU(np.copy(this_U[:, iR, j])-psia, 1, 0)

            #Getting the rotated states in the left and right cells
            rho_ijL, u_ijL, v_ijL, p_ijL, a_ijL = find_rho_u_v_p_a(UL) 
            rho_ijR, u_ijR, v_ijR, p_ijR, a_ijR = find_rho_u_v_p_a(UR) 

            #Calculating the rotated flux vector and then rotating it back to standard
            flux = hlle(rho_ijL, u_ijL, v_ijL, p_ijL, a_ijL, rho_ijR, u_ijR, v_ijR, p_ijR, a_ijR)
            flux = rotConversionF(flux, 1, 0)

            #Evaluating fluxes at the far left bound, far right bound, or within the domain
            if i == 0:
                F[:, i, j] += flux
            elif i == n:
                F[:, i-1, j] -= flux
            else:
                F[:, i, j] += flux
                F[:, i-1, j] -= flux

    #Calculating Fluxes in Y
    for i in range(n):
        for j in range(m+1):
            jL, jR = 0,0

            #Boundary at the far low bound
            if j == 0:
                #Setting the lower (ghost) and upper index
                jL, jR = j, j 

                #Regular rotation conversion of the lower and upper cells
                UL = rotConversionU(np.copy(this_U[:, i, jL]), 0, 1)
                UR = rotConversionU(np.copy(this_U[:, i, jR]), 0, 1)

            #Boundary at the far up bound
            elif j == m:
                #Setting the lower and upper (ghost) index
                jL, jR = j-1, j-1

                #Regular rotation conversion of the lower and upper cells
                UL = rotConversionU(np.copy(this_U[:, i, jL]), 0, 1)
                UR = rotConversionU(np.copy(this_U[:, i, jR]), 0, 1)

            else:
                #Setting the lower and upper index
                jL, jR = j-1, j

                #Regular rotation conversion of the lower and upper cells
                '''UL = rotConversionU(np.copy(this_U[:, i, jL]), 0, 1)
                UR = rotConversionU(np.copy(this_U[:, i, jR]), 0, 1)'''

                #Slope limited rotation conversion of lower and upper cells
                psia, psib = limiter_of_slopes(this_U, i, j-1)
                UL = rotConversionU(np.copy(this_U[:, i, jL])+psib, 0, 1)
                psia, psib = limiter_of_slopes(this_U, i, j)
                UR = rotConversionU(np.copy(this_U[:, i, jR])-psib, 0, 1)

            #Getting the rotated states in the lower and upper cells
            rho_ijL, u_ijL, v_ijL, p_ijL, a_ijL = find_rho_u_v_p_a(UL) 
            rho_ijR, u_ijR, v_ijR, p_ijR, a_ijR = find_rho_u_v_p_a(UR) 

            #Calculating the rotated flux vector and then rotating it back to standard
            flux = hlle(rho_ijL, u_ijL, v_ijL, p_ijL, a_ijL, rho_ijR, u_ijR, v_ijR, p_ijR, a_ijR)
            flux = rotConversionF(flux, 0, 1)

            #Evaluating fluxes at the lower bound, upper bound, or within the domain
            if j == 0:
                F[:, i, j] += flux
            elif j == m:
                F[:, i, j-1] -= flux
            else:
                F[:, i, j] += flux
                F[:, i, j-1] -= flux

#Defining a function to carry out contour plotting
def plot_fig(fig_number, x_value, y_value, plot_value, plot_name, x_name, y_name):
    fig = plt.figure(fig_number)
    x_plot, y_plot = np.meshgrid(x_value, y_value)
    cp = plt.contourf(y_plot, x_plot, plot_value, cmap = 'jet')
    plt.colorbar(cp)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(plot_name)
    plt.savefig(plot_name)

#Initial conditions for Problem 1 are contained in this function
def problem1():

    t = 0.00075

    rho_1 = 1.225
    u_1   = 0.0
    v_1   = 0.0
    p_1   = 101325.0

    rho_2 = 0.30625
    u_2   = 0.0
    v_2   = 0.0
    p_2   = 25331.25

    #Setting initial conditions in each cell
    for i in range(n):
        for j in range(m):
            if x[i] < 0.0 and y[j] < 0.0:
                U[:, i, j] = find_U(rho_2, u_2, v_2, p_2) 
            else:
                U[:, i, j] = find_U(rho_1, u_1, v_1, p_1) 
    return U, t

#Initial conditions for Problem 2 are contained in this function
def problem2():

    t = 0.00253

    rho_1 = 1.5
    u_1   = 0.0
    v_1   = 0.0
    p_1   = 150000.0

    rho_2 = 0.532258064516129
    u_2   = 381.385035698
    v_2   = 0.0
    p_2   = 30000.0

    rho_3 = 0.137992831541219
    u_3   = 381.385035698
    v_3   = 381.385035698
    p_3   = 2903.2258064516

    rho_4 = 0.532258064516129
    u_4   = 0.0
    v_4   = 381.385035698 
    p_4   = 30000.0

    #Setting initial conditions in each cell
    for i in range(n):
        for j in range(m):
            if x[i] > 0.3 and y[j] > 0.3:
                U[:, i, j] = find_U(rho_1, u_1, v_1, p_1) 
            elif x[i] < 0.3 and y[j] > 0.3:
                U[:, i, j] = find_U(rho_2, u_2, v_2, p_2) 
            elif x[i] < 0.3 and y[j] < 0.3:
                U[:, i, j] = find_U(rho_3, u_3, v_3, p_3) 
            else:
                U[:, i, j] = find_U(rho_4, u_4, v_4, p_4)
    return U, t

#Initial conditions for Problem 3 are contained in this function
def problem3():

    t = 0.0019

    rho_1 = 1.0
    u_1   = 237.170824513
    v_1   = -158.113883008
    p_1   = 100000.0

    rho_2 = 3.0
    u_2   = 237.170824513
    v_2   = 158.113883008
    p_2   = 100000.0

    rho_3 = 1.0
    u_3   = -237.170824513
    v_3   = 158.113883008
    p_3   = 100000.0

    rho_4 = 3.0
    u_4   = -237.170824513
    v_4   = -158.113883008
    p_4   = 100000.0

    #Setting initial conditions in each cell
    for i in range(n):
        for j in range(m):
            if x[i] > 0.0 and y[j] > 0.0:
                U[:, i, j] = find_U(rho_1, u_1, v_1, p_1) 
            elif x[i] < 0.0 and y[j] > 0.0:
                U[:, i, j] = find_U(rho_2, u_2, v_2, p_2) 
            elif x[i] < 0.0 and y[j] < 0.0:
                U[:, i, j] = find_U(rho_3, u_3, v_3, p_3) 
            else:
                U[:, i, j] = find_U(rho_4, u_4, v_4, p_4)
    return U, t

#Initializing the problem we wish to solve
U, t = problem3()

#Creating arrays to store a, u, v, p, and rho
a   = np.zeros((n, m))
u   = np.zeros((n, m))
v   = np.zeros((n, m))
p   = np.zeros((n, m))
rho = np.zeros((n, m))

#Marching in time until we hit the final specified time t
while current_t < t:
    #Resetting all entries in F (dUdt) to zero
    F.fill(0.0)

    #Setting delta t to a very high number and max lambda to a very small number
    delta_t = 1e12
    max_lambda = 1e-12

    #Looping through every cell in the array
    for i in range(n):
        for j in range(m):
            #Checking the maximum lambda in the x-direction
            U_X = np.copy(U[:, i, j])
            U_rotX = rotConversionU(U_X, 1, 0)
            rho_ij, u_ij, v_ij, p_ij, a_ij = find_rho_u_v_p_a(U_rotX)
            max_lambda = max(abs(u_ij)+a_ij, max_lambda)

            #Checking the maximum lambda in the y-direction
            U_Y = np.copy(U[:, i, j])
            U_rotY = rotConversionU(U_Y, 0, 1)
            rho_ij, u_ij, v_ij, p_ij, a_ij = find_rho_u_v_p_a(U_rotY)
            max_lambda = max(abs(u_ij)+a_ij, max_lambda)

    #Determining the new delta t based on the previous and current max lambda
    delta_t = min(CFL*delta_x/max_lambda, delta_t)

    #If the new delta t will take us passed the final time t (adjust to reach t)
    if current_t + delta_t > t:
        delta_t = t - current_t

    #Grabbing U in this time step and evaluating dUdt
    flux_calc(U)
                
    #Godunov Scheme (Predictor Corrector)
    Utilda = U + delta_t*F/delta_x
    flux_calc(Utilda)
    U += 0.5*delta_t*F/delta_x

    #Adjusting the current t based on delta t and printing status to completionS
    current_t += delta_t

    percentComplete = (current_t/t)*100.0
    percentComplete = int(percentComplete)
    print(str(percentComplete) + "% " + "To Completion")

#Storing our states of interest
for i in range(n):
    for j in range(m):
        rho[i, j], u[i, j], v[i, j], p[i, j], a[i, j] = find_rho_u_v_p_a(U[:, i, j])

#Displaying the resulting figures
plot_fig(1, x, y, u,   "Velocity x", "x (m)", "y (m)")
plot_fig(2, x, y, v,   "Velocity y", "x (m)", "y (m)")
plot_fig(3, x, y, p,   "Pressure"  , "x (m)", "y (m)")
plot_fig(4, x, y, rho, "Density"   , "x (m)", "y (m)")