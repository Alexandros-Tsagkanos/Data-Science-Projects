import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def plot_armijo():
    phi   =    lambda alpha: (alpha - 2) **2 + 1

    phi_prime_0   =  - 4

    alpha   =    np.linspace(0, 4, 400)
    phi_values   =    phi(alpha)

    c   =    0.3
    armijo_line   =    phi(0) + c *alpha *phi_prime_0

    plt.figure(figsize  =   (10, 6))
    plt.plot(
        alpha, phi_values, 'b-', linewidth  =   2, label  =   r'$\phi(\alpha)   =    f(x_k + \alpha p_k)$'
    )
    plt.plot(
            alpha, armijo_line, 'r--', linewidth=2, label=r'$\phi(0) + c\alpha\phi^\prime(0)$'
    )

    plt.plot(0, phi(0), "go", markersize  =   10, label  =   r"$\phi(0)$")

    plt.fill_between(alpha, phi_values, armijo_line,
                     where  =   (phi_values   <=  armijo_line),
                     alpha  =   0.3, color  =   "green",
                     label  =   'Αποδεκτή περιοχή (Armijo)')

    plt.xlabel(r"Μήκος βήματος $\alpha$", fontsize  =   12)
    plt.ylabel(r'$\phi(\alpha)$', fontsize  =   12)
    plt.title(
        'Συνθήκη Armijo για Line Search', fontsize  =   14, fontweight  =   "bold"
    )
    plt.legend(fontsize  =   10)
    plt.grid(bool(1), alpha  =   0.3)
    plt.xlim([0, 4])
    plt.ylim([0, 6])

    plt.tight_layout()
    plt.savefig("armijo_plot.png", dpi  =   300, bbox_inches  =   "tight")
    plt.close()
    print("✓ Το γράφημα 'armijo_plot.png' δημιουργήθηκε επιτυχώς!")

def plot_descent_direction():
    fig, ax   =    plt.subplots(figsize  =   (10, 8))

    gradient   =    np.array([3, 2])

    gradient_norm   =    gradient /np.linalg.norm(gradient)

    steepest_descent   =  - gradient_norm

    descent_dir1   =    np.array([ - 0.8, - 0.3])
    descent_dir1   =    descent_dir1 /np.linalg.norm(descent_dir1)

    descent_dir2   =    np.array([ - 0.5, - 0.7])
    descent_dir2   =    descent_dir2 / np.linalg.norm(descent_dir2)

    non_descent   =    np.array([0.6, 0.5])
    non_descent   =    non_descent / np.linalg.norm(non_descent)

    origin  =   np.array([0, 0])

    ax.quiver( *origin, *gradient_norm, color =  'red', scale =  2, width =  0.015,
              label =  r'$\nabla f(x_k)$ (Gradient)', zorder =  5)

    ax.quiver( *origin, *steepest_descent, color =  'green', scale =  2, width =  0.02,
              label =  r'$ - \nabla f(x_k)$ (Steepest Descent)', zorder =  5)

    ax.quiver( *origin, *descent_dir1, color =  "blue", scale =  2, width =  0.012,
              alpha =  0.7, label =  'Κατεύθυνση καθόδου 1', zorder =  4)
    ax.quiver( *origin, *descent_dir2, color =  "blue", scale =  2, width =  0.012,
              alpha =  0.7, label =  "Κατεύθυνση καθόδου 2", zorder =  4)

    ax.quiver( *origin, *non_descent, color =  'gray', scale =  2, width =  0.012,
              alpha =  0.5, label =  "Μη - κατεύθυνση καθόδου", zorder =  3)

    t  =   np.linspace( - 1.5, 1.5, 100)
    perp_line_x =  t
    perp_line_y = - (gradient_norm[0]/gradient_norm[1]) *t
    ax.plot(perp_line_x, perp_line_y, "k--", linewidth= 1.5, alpha= 0.5,
            label= r"Υπερεπίπεδο $\nabla f^T d =  0$")

    ax.text( - 0.7, - 0.5, "Περιοχή καθόδου\n" + r'($\nabla f^T d < 0$)',
            fontsize= 11, ha= "center",
            bbox= dict(boxstyle= "round", facecolor= "lightgreen", alpha= 0.3))

    ax.set_xlim([ - 1.2, 1.2])
    ax.set_ylim([ - 1.2, 1.2])
    ax.set_aspect('equal')
    ax.axhline(y= 0, color= 'k', linewidth= 0.5, alpha= 0.3)
    ax.axvline(x= 0, color= "k", linewidth= 0.5, alpha= 0.3)
    ax.grid(bool(1), alpha= 0.3)
    ax.set_xlabel('$d_1$', fontsize= 12)
    ax.set_ylabel('$d_2$', fontsize= 12)
    ax.set_title("Κατευθύνσεις Καθόδου και Steepest Descent",
                 fontsize= 14, fontweight= "bold")
    ax.legend(loc= 'upper right', fontsize= 9)

    plt.tight_layout()
    plt.savefig('descent_direction.png', dpi= 300, bbox_inches= 'tight')
    plt.close()
    print("✓ Το γράφημα 'descent_direction.png' δημιουργήθηκε επιτυχώς!")

def freudenstein_roth(x1, x2):
    term1 =  x1 - 13 + ((5 - x2) *x2 - 2) *x2
    term2 =  x1 - 29 + ((x2 + 1) *x2 - 14) *x2
    return term1 **2 + term2 **2

def plot_freudenstein_3d():
    x1 =  np.linspace(-10, 10, 400)
    x2 =  np.linspace(-10, 10, 400)
    X1, X2 =  np.meshgrid(x1, x2)

    Z =  freudenstein_roth(X1, X2)

    fig =  plt.figure(figsize= (12, 9))
    ax =  fig.add_subplot(111, projection= '3d')

    surf =  ax.plot_surface(X1, X2, Z, cmap= 'viridis',
                           edgecolor= "none", alpha= 0.9,
                           vmin= 0, vmax=5000)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label("$f(x_1, x_2)$", fontsize=12)

    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)
    ax.set_zlabel('$f(x_1, x_2)$', fontsize=12)
    ax.set_title("Συνάρτηση Freudenstein and Roth - 3D Απεικόνιση",
                 fontsize=14, fontweight='bold', pad=20)

    ax.view_init(elev=25, azim=45)

    ax.set_zlim([0, 5000])

    plt.tight_layout()
    plt.savefig("freudenstein_3d.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Το γράφημα 'freudenstein_3d.png' δημιουργήθηκε επιτυχώς!", end="\n")

def plot_freudenstein_contour():
    x1 = np.linspace(-10, 10, 500)
    x2 = np.linspace(-10, 10, 500)
    X1, X2 = np.meshgrid(x1, x2)

    Z = freudenstein_roth(X1, X2)

    plt.figure(figsize=(12, 10))

    levels = np.logspace(0, 4, 30)

    contourf = plt.contourf(X1, X2, Z, levels=levels, cmap="viridis", alpha=0.8)

    contour = plt.contour(X1, X2, Z, levels=levels, colors="black",
                          linewidths=0.5, alpha=0.4)

    cbar = plt.colorbar(contourf)
    cbar.set_label('$f(x_1, x_2)$', fontsize=12)

    plt.plot(0.5, -2, 'r*', markersize=20, label="$x^{(0)} = (0.5, -2)$",
             markeredgecolor="white", markeredgewidth=1.5)

    plt.xlabel("$x_1$", fontsize=12)
    plt.ylabel("$x_2$", fontsize=12)
    plt.title("Συνάρτηση Freudenstein and Roth - Contour Plot",
              fontsize=14, fontweight="bold")
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(bool(1), alpha=0.3)
    plt.axis("equal")

    plt.tight_layout()
    plt.savefig('freudenstein_contour.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Το γράφημα 'freudenstein_contour.png' δημιουργήθηκε επιτυχώς!")

def gradient_freudenstein(x1, x2):
    term1 = x1 - 13 + ((5 - x2) *x2 - 2) *x2
    term2 = x1 - 29 + ((x2 + 1) *x2 - 14)* x2

    dterm1_dx2 = (5 - x2)* x2 - 2 + (5 - 2* x2)* x2
    dterm2_dx2 = (x2 + 1)* x2 - 14 + (2* x2 + 1)* x2

    df_dx1 = 2* term1 + 2* term2
    df_dx2 = 2* term1* dterm1_dx2 + 2* term2 * dterm2_dx2

    return np.array([df_dx1, df_dx2])

def hessian_freudenstein(x1, x2):
    term1 = x1 - 13 + ((5 - x2) * x2 - 2) * x2
    term2 = x1 - 29 + ((x2 + 1) * x2 - 14) * x2

    dterm1_dx2 = (5 - x2) * x2 - 2 + (5 - 2*x2) * x2
    dterm2_dx2 = (x2 + 1) * x2 - 14 + (2*x2 + 1) * x2

    d2term1_dx2_2 = 2 * (5 - 2*x2) - 2*x2
    d2term2_dx2_2 = 2 * (2*x2 + 1) + 2*x2

    d2f_dx1_2 = 4
    d2f_dx1_dx2 = 2 * dterm1_dx2 + 2 * dterm2_dx2
    d2f_dx2_2 = (2 * dterm1_dx2**2 + 2 * term1 * d2term1_dx2_2 +
                 2 * dterm2_dx2**2 + 2 * term2 * d2term2_dx2_2)

    H = np.array([[d2f_dx1_2, d2f_dx1_dx2],
                  [d2f_dx1_dx2, d2f_dx2_2]])

    return (H)
def one_iteration_methods():
    x0 = np.array([0.5, -2.0])

    print("="*70)
    print('ΥΠΟΛΟΓΙΣΜΟΣ ΜΙΑΣ ΕΠΑΝΑΛΗΨΗΣ - EXERCISE 3b')
    print("="*70, end="\n")
    print(f'Αρχικό σημείο: x^(0) = {x0}')
    print(f"f(x^(0)) = {freudenstein_roth(x0[0], x0[1]):.6f}")
    print()

    grad0 = gradient_freudenstein(x0[0], x0[1])
    H0 = hessian_freudenstein(x0[0], x0[1])

    print(f"∇f(x^(0)) = {grad0}")
    print(f"||∇f(x^(0))|| = {np.linalg.norm(grad0):.6f}")
    print()
    print("Hessian H(x^(0)):")
    print(H0, end="\n")
    print()

    print("-"*70)
    print("1. STEEPEST DESCENT")
    print('-'*70)

    d_sd = -grad0
    print(f'Κατεύθυνση: d = -∇f(x^(0)) = {d_sd}')

    def phi_sd(alpha):
        x_new = x0 + alpha * d_sd
        return freudenstein_roth(x_new[0], x_new[1])

    result_sd = minimize_scalar(phi_sd, bounds=(0, 10), method='bounded')
    alpha_sd = result_sd.x
    x1_sd = x0 + alpha_sd * d_sd

    print(f'Βέλτιστο μήκος βήματος: α* = {alpha_sd:.6f}')
    print(f"Νέο σημείο: x^(1) = {x1_sd}")
    print(f"f(x^(1)) = {freudenstein_roth(x1_sd[0], x1_sd[1]):.6f}")
    print()

    print('-'*70)
    print('2. NEWTON METHOD', end="\n")
    print("-"*70, end="\n")

    try:
        H_inv = np.linalg.inv(H0)
        d_newton = -H_inv @ grad0
        print(f"Κατεύθυνση: d = -H^(-1)∇f(x^(0)) = {d_newton}")

        def phi_newton(alpha):
            x_new = x0 + alpha * d_newton
            return freudenstein_roth(x_new[0], x_new[1])

        result_newton = minimize_scalar(
            phi_newton, bounds=(0, 10), method='bounded'
        )
        alpha_newton = result_newton.x
        x1_newton = x0 + alpha_newton * d_newton

        print(f'Βέλτιστο μήκος βήματος: α* = {alpha_newton:.6f}')
        print(f'Νέο σημείο: x^(1) = {x1_newton}')
        print(f'f(x^(1)) = {freudenstein_roth(x1_newton[0], x1_newton[1]):.6f}')
    except np.linalg.LinAlgError:
        print("ΠΡΟΣΟΧΗ: Ο Hessian δεν είναι αντιστρέψιμος!")
    print()

    print('-'*70)
    print("3. BFGS (Quasi-Newton)")
    print('-'*70)

    B0 = np.eye(2)
    d_bfgs = -B0 @ grad0
    print(f"Αρχικός πίνακας: B_0 = I (ταυτοτικός)")
    print(f'Κατεύθυνση: d = -B_0^(-1)∇f(x^(0)) = -∇f(x^(0)) = {d_bfgs}')
    print(f"(Για την 1η επανάληψη, το BFGS δίνει το ίδιο με Steepest Descent)")
    print(f'x^(1) = {x1_sd}')
    print(f'f(x^(1)) = {freudenstein_roth(x1_sd[0], x1_sd[1]):.6f}')
    print()

    print('-'*70)
    print('4. FLETCHER-REEVES (Conjugate Gradient)')
    print('-'*70, end="\n")

    print(f"Για την 1η επανάληψη: β_0 = 0")
    print(f'Κατεύθυνση: d = -∇f(x^(0)) = {d_sd}')
    print(f"(Για την 1η επανάληψη, το FR δίνει το ίδιο με Steepest Descent)")
    print(f"x^(1) = {x1_sd}")
    print(f"f(x^(1)) = {freudenstein_roth(x1_sd[0], x1_sd[1]):.6f}")
    print()

    print("="*70)
    print('ΣΥΝΟΨΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ')
    print('='*70)
    print(f"{'Μέθοδος':<25} {'x1^(1)':<12} {'x2^(1)':<12} {'f(x^(1))':<15}")
    print('-'*70, end="\n")
    print(f"{'Steepest Descent':<25} {x1_sd[0]:<12.6f} {x1_sd[1]:<12.6f} "
          f"{freudenstein_roth(x1_sd[0], x1_sd[1]):<15.6f}")
    print(f"{'Newton':<25} {x1_newton[0]:<12.6f} {x1_newton[1]:<12.6f} "
          f'{freudenstein_roth(x1_newton[0], x1_newton[1]):<15.6f}')
    print(f"{'BFGS':<25} {x1_sd[0]:<12.6f} {x1_sd[1]:<12.6f} "
          f'{freudenstein_roth(x1_sd[0], x1_sd[1]):<15.6f}')
    print(f"{'Fletcher-Reeves':<25} {x1_sd[0]:<12.6f} {x1_sd[1]:<12.6f} "
          f"{freudenstein_roth(x1_sd[0], x1_sd[1]):<15.6f}")
    print('='*70)
    print()

    print('='*70)
    print('ΣΥΝΘΗΚΕΣ ΣΥΓΚΛΙΣΗΣ ΓΙΑ FLETCHER-REEVES', end="\n")
    print('='*70, end="\n")
    print("Η μέθοδος Fletcher-Reeves συγκλίνει υπό τις εξής προϋποθέσεις:", end="\n")
    print("1. Η συνάρτηση f είναι δύο φορές συνεχώς διαφορίσιμη")
    print('2. Η συνάρτηση f είναι κυρτή (convex)')
    print("3. Το line search ικανοποιεί τις Strong Wolfe conditions")
    print("4. Η κλίση ∇f είναι Lipschitz συνεχής")
    print()
    print('Για την συνάρτηση Freudenstein and Roth:')
    print("- Είναι δύο φορές συνεχώς διαφορίσιμη ✓")
    print('- ΔΕΝ είναι κυρτή (έχει πολλαπλά τοπικά ελάχιστα) ✗')
    print("- Άρα η σύγκλιση στο global minimum ΔΕΝ είναι εγγυημένη")
    print("="*70)

def main():
    print('\n' + '='*70)
    print('ΔΗΜΙΟΥΡΓΙΑ ΓΡΑΦΗΜΑΤΩΝ ΚΑΙ ΥΠΟΛΟΓΙΣΜΩΝ')
    print("="*70 + '\n')

    print('1. Δημιουργία γραφήματος Armijo...')
    plot_armijo()
    print()

    print("2. Δημιουργία γραφήματος κατευθύνσεων καθόδου...")
    plot_descent_direction()
    print()

    print("3. Δημιουργία 3D γραφήματος Freudenstein and Roth...")
    plot_freudenstein_3d()
    print()

    print('4. Δημιουργία contour plot Freudenstein and Roth...', end="\n")
    plot_freudenstein_contour()
    print()

    print('5. Υπολογισμός επαναλήψεων μεθόδων βελτιστοποίησης...')
    one_iteration_methods()

    print('\n' + '='*70)
    print('ΟΛΟΚΛΗΡΩΣΗ! Όλα τα γραφήματα δημιουργήθηκαν επιτυχώς.')
    print("="*70 + "\n")

if __name__    =='__main__':
    main()