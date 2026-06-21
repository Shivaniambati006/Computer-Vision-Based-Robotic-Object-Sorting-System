import numpy as np
from sympy import symbols, summation, cos, apart, solve, simplify

# Define z symbol for Z-Transform
z = symbols('z')

# Helper function to classify a discrete-time signal (periodic/non-periodic, energy/power)
def classify_signal(signal):
    periodicity = 'Periodic' if np.array_equal(signal, np.roll(signal, len(signal) // 2)) else 'Non-periodic'
    energy = np.sum(np.abs(signal) ** 2)
    power = np.mean(np.abs(signal) ** 2)
    
    if energy < np.inf and power == 0:
        signal_type = 'Energy Signal'
    elif power < np.inf:
        signal_type = 'Power Signal'
    else:
        signal_type = 'Neither Energy nor Power Signal'

    return periodicity, signal_type, energy, power

# Z-Transform computation for a symbolic expression
def z_transform(signal_expr, n_limit=10):
    n = symbols('n')
    return summation(signal_expr * z**(-n), (n, 0, n_limit))

# Manually compute inverse Z-transform
def inverse_z_trans(X_z):
    X_z = apart(X_z)  # Perform partial fraction decomposition
    terms = X_z.as_ordered_terms()  # Decompose into individual terms
    n = symbols('n')
    result = 0

    for term in terms:
        if z in term.free_symbols:
            coefficient = term.as_coeff_exponent(z)[0]
            exponent = term.as_coeff_exponent(z)[1]
            result += coefficient * z**(-exponent).subs(z, 1)
        else:
            result += term

    return simplify(result)

# Classify system based on transfer function (poles, zeros, stability)
def classify_system(H_z):
    numerator, denominator = H_z.as_numer_denom()
    zeros = solve(numerator, z)
    poles = solve(denominator, z)
    stable = all(abs(p) < 1 for p in poles)
    return zeros, poles, stable

# Example usage of the program

# 1. Signal Classification and Z-Transform
n_vals = np.arange(0, 10)
signal = np.cos(2 * n_vals)  # Example discrete-time signal cos(2n)
periodicity, signal_type, energy, power = classify_signal(signal)
print(f"Signal Classification:")
print(f"Periodicity: {periodicity}, Signal Type: {signal_type}, Energy: {energy}, Power: {power}")

# Z-transform of the signal cos(2n)u[n]
n = symbols('n')  # Define n symbol for symbolic expression
signal_expr = cos(2 * n)
z_trans = z_transform(signal_expr, n_limit=10)  # Limit summation to 10 terms
print(f"Z-Transform of signal: {z_trans}")

# 2. System Classification (Z-domain analysis)
H_z_example = (z + 1) / (z**2 - 0.5*z + 0.25)
zeros, poles, stable = classify_system(H_z_example)
print(f"System Classification:")
print(f"Zeros: {zeros}, Poles: {poles}, Stable: {stable}")

# 3. Impulse and Step Responses
impulse_resp = inverse_z_trans(H_z_example)
print(f"Impulse Response: {impulse_resp}")