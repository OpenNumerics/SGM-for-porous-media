def getPDEParameters():
    t_plus = 0.4

    # Build a Gaussian process for eps(x)
    a_s = 5e5
    k_rn = 0.1
    G = 1.e3
    phi_s = lambda x: -G*x

    # Run the FVM simulator
    parameters = {"t_plus" : t_plus, "a_s" : a_s, "k_rn" : k_rn}
    return parameters, phi_s