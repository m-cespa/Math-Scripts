import sympy as sp
from sympy import sin, tan, cos, pretty, latex
from IPython.display import display, Math
from sympy.core.symbol import Symbol

theta, phi = sp.symbols('theta phi')
coords = [theta, phi]

g = sp.Matrix([[1, 0],
               [0, sin(theta)**2]])

g_inv = g.inv()

n = len(coords)
Gamma = sp.MutableDenseNDimArray.zeros(n, n, n)
Riemann = sp.MutableDenseNDimArray.zeros(n, n, n, n)

# notation convention here is 
# Gamma[upper, lower, lower] = Γ[ρ, μ, ν]
# Riemann[upper, lower, lower, lower] = R[d, a, b, c]

for rho in range(n):
    for mu in range(n):
        for nu in range(n):
            # christoffel symbol formula in terms of metric
            out = sum(g_inv[rho, sigma] * (
                sp.diff(g[sigma, nu], coords[mu]) +
                sp.diff(g[sigma, mu], coords[nu]) -
                sp.diff(g[mu, nu], coords[sigma])
            ) for sigma in range(n))

            Gamma[rho, mu, nu] = sp.simplify(0.5 * out)

for d in range(n):
    for a in range(n):
        for b in range(n):
            for c in range(n):
                # Riemann curvature tensor formula in terms of christoffel
                term1 = sp.diff(Gamma[d, b, c], coords[a])
                term2 = sp.diff(Gamma[d, a, c], coords[b])
                term3 = sum(Gamma[e, a, c] * Gamma[d, b, e] for e in range(n))
                term4 = sum(Gamma[e, b, c] * Gamma[d, a, e] for e in range(n))
            
                Riemann[d, a, b, c] = sp.simplify(-term1 + term2 + term3 - term4)

def convert_to_latex(coord: Symbol):
    greek_letters = {
        'alpha': '\\alpha', 'beta': '\\beta', 'gamma': '\\gamma', 'delta': '\\delta',
        'epsilon': '\\epsilon', 'zeta': '\\zeta', 'eta': '\\eta', 'theta': '\\theta',
        'iota': '\\iota', 'kappa': '\\kappa', 'lambda': '\\lambda', 'mu': '\\mu',
        'nu': '\\nu', 'xi': '\\xi', 'omicron': '\\omicron', 'pi': '\\pi',
        'rho': '\\rho', 'sigma': '\\sigma', 'tau': '\\tau', 'upsilon': '\\upsilon',
        'phi': '\\phi', 'chi': '\\chi', 'psi': '\\psi', 'omega': '\\omega'
    }

    # checks if coordinate is greek letter
    coord_name = coord.name
    if coord_name in greek_letters:
        return greek_letters[coord_name]
    else:
        return coord_name

display(Math(r"The\ Connection\ coefficients\ are:"))
connections_lst = ''
for rho in range(n):
    for mu in range(n):
        for nu in range(n):
            # converting coordinate symbols to latex format
            rho_latex = convert_to_latex(coords[rho])
            mu_latex = convert_to_latex(coords[mu])
            nu_latex = convert_to_latex(coords[nu])

            symbol = f"\\Gamma^{{{rho_latex}}}_{{{mu_latex} {nu_latex}}}"
            value = latex(Gamma[rho, mu, nu])
            equation = f"{symbol} = {value} ; "

            connections_lst += equation

display(Math(connections_lst))

display(Math(r"The\ Riemann\ Curvature\ Tensor\ coefficients\ are:"))
riemman_lst = ''
for d in range(n):
    for a in range(n):
        for b in range(n):
            for c in range(n):
                # converting coordinate symbols to latex format
                a_latex = convert_to_latex(coords[a])
                b_latex = convert_to_latex(coords[b])
                c_latex = convert_to_latex(coords[c])
                d_latex = convert_to_latex(coords[d])

                symbol = f"R^{{{d_latex}}}_{{{a_latex} {b_latex} {c_latex}}}"
                value = latex(Riemann[d, a, b, c])
                equation = f"{symbol} = {value} ; "

                riemman_lst += equation

display(Math(riemman_lst))
