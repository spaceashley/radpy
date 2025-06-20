import numpy as np

#function to calculate temperature from bolemetric flux and angular diameter
def temp(Fbol,dF,theta,dtheta):
    T = 2341*(Fbol/theta**2)**(1/4)
    dT = np.sqrt(
          (2341 / (4 * Fbol**(3/4) * theta**(1/2)) * dF)**2
        + (2341 * Fbol**(1/4) / (2* theta**(3/2)) * dtheta)**2
    )
    # dT = T*np.sqrt(((dF/Fbol)*(1/4))**2 + ((dtheta/theta)*(1/2))**2)
    #return(print("Temperature: ", T ,"+/-",dT,"[K]"))
    return((T,dT))

#calculates the distance given parallax
def dist_calc(p, p_err, zpc):
    pc = (p+zpc)/1000
    D = 1/(pc)
    dD = D*((p_err/1000)/pc)
    print("Corrected parallax:", pc*1000)
    print("Distance:",D,"+/-", dD, "[pc]")
    return((D,dD))

#Luminosity function
def luminosity(D,dD, Fbol, dF):
    pc_conv = 3.091e16          #meters in a parsec
    d = D*pc_conv*100;
    L = (4*np.pi*(d**2)*Fbol)/(3.846e33)
    dL = L*np.sqrt(((2*dD)/D)**2 + (dF/Fbol)**2)
    return(print("Luminosity: ", L ,"+/-",dL, "[L_solar]"))
    # return((L,dL))


# Physical radius function
def ang_to_lin(theta, dtheta, D, dD):
    Rs = 6.957e8;  # meters in a solar radii
    pc_conv = 3.091e16  # meters in a parsec
    mas_conv = 206265 * 1000  # radians to mas conversion

    R = (theta * (D * pc_conv)) / (2 * mas_conv * Rs)
    # dR = R*np.sqrt((dtheta/theta)**2 + (dD/D)**2)                         #my equation same as Jonas just different derivation
    dR = (pc_conv / (2 * mas_conv * Rs)) * np.sqrt(
        (dtheta * D) ** 2 + (dD * theta) ** 2)  # jonas equation matches the first one
    return (print("Linear Radius: ", R, "+/-", dR, "[R_solar]"))

