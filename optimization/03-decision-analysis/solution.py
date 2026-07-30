# -*- coding: utf-8 -*-
# Εργασία 3 - Τσαγκάνος Αλέξανδρος
# Ανάλυση Αποφάσεων & Βελτιστοποίηση, 2025-26

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# gonna need these for the plots to look decent
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150


# make sure the figures folder exists
figdir = './figures/'
if not os.path.exists(figdir):
    os.makedirs(figdir)

# helper functions for drawing the decision tree
# tetragono = square (decision node)
def vale_tetragono(ax, cx, cy, platos=0.15):
    r = plt.Rectangle((cx - platos/2, cy - platos/2), platos, platos,
                    facecolor='#4CAF50', edgecolor='k', lw=1.5, zorder=5)
    ax.add_patch(r)

# kyklos = circle (chance node)
def vale_kyklo(ax, cx, cy, aktina=0.08):
    k = plt.Circle((cx, cy), aktina, facecolor='#FF9800', edgecolor='k', lw=1.5, zorder=5)
    ax.add_patch(k)

# for terminal nodes
def vale_telos(ax, cx, cy, poso, megethos=7):
    ax.plot(cx, cy, 's', color='#2196F3', markersize=6, zorder=5)
    ax.text(cx + 0.02, cy, f'{poso:,.2f}', fontsize=megethos, va='center', fontweight='bold')

# ok so im gonna alias these to english names too cause i keep mixing them up lol
draw_decision_node = vale_tetragono
draw_chance_node = vale_kyklo
chance_circle = vale_kyklo  # same thing basically
decision_box = vale_tetragono

# also need this for compatibility
FIGDIR = figdir   # uppercase version

print("=" * 70)
print("ΑΣΚΗΣΗ 1A: Chelsea Bush - Πρόβλημα Προεκλογικής Στρατηγικής")
print("=" * 70)

# ---- askisi 1A ----

# basic probabilities
p_kala_ST = 0.4
p_asxima_ST = 0.6
kerdos_ST = 16
zimia_ST = -10

#conditional probs given NH result
p_kala_ST_dedom_kala_NH = 4/7
p_kala_ST_dedom_asxima_NH = 1/7
p_kala_NH = 7/15
p_asxima_NH = 1 - p_kala_NH

kostos_NH = 1.6   # million dollars

# calculate the complement probs
p_asxima_ST_dedom_kala_NH = 1 - p_kala_ST_dedom_kala_NH
p_asxima_ST_dedom_asxima_NH = 1 - p_kala_ST_dedom_asxima_NH

print("\n--- Δεδομένες Πιθανότητες ---")
print("P(Καλά S.T.) =", p_kala_ST)
print("P(Άσχημα S.T.) =", p_asxima_ST)
print(f"P(Καλά S.T. | Καλά N.H.) = {p_kala_ST_dedom_kala_NH:.4f} = 4/7")
print(f"P(Καλά S.T. | Άσχημα N.H.) = {p_kala_ST_dedom_asxima_NH:.4f} = 1/7")
print(f"P(Καλά N.H.) = {p_kala_NH:.4f} = 7/15")
print(f"P(Άσχημα N.H.) = {p_asxima_NH:.4f} = 8/15")

# without NH primary - just go directly to super tuesday
ev_symmetoxh_xoris_NH = p_kala_ST * kerdos_ST + p_asxima_ST * zimia_ST
print("\n--- Χωρίς N.H. ---")
print(f"EV(Συμμετοχή S.T.) = {p_kala_ST}×{kerdos_ST} + {p_asxima_ST}×({zimia_ST}) = {ev_symmetoxh_xoris_NH:.2f} εκατ. $")
print("EV(Μη Συμμετοχή S.T.) = 0")
ev_xoris_NH = max(ev_symmetoxh_xoris_NH, 0)
if ev_symmetoxh_xoris_NH > 0:
    print("Βέλτιστη απόφαση χωρίς N.H.: Συμμετοχή στο S.T.")
else:
    print("Βέλτιστη απόφαση χωρίς N.H.: Μη Συμμετοχή")
print(f"EV = {ev_xoris_NH:.2f} εκατ. $")

# if we do NH and it goes well...
ev_symm_kala_NH = p_kala_ST_dedom_kala_NH * kerdos_ST + p_asxima_ST_dedom_kala_NH * zimia_ST
ev_opt_kala_NH = max(ev_symm_kala_NH, 0)
print("\n--- Μετά από Καλά στο N.H. ---")
print(f"EV(Συμμετοχή S.T. | Καλά N.H.) = (4/7)×16 + (3/7)×(-10) = {ev_symm_kala_NH:.4f}")
apof_kala = "Συμμετοχή" if ev_symm_kala_NH > 0 else "Μη Συμμετοχή"
print("Βέλτιστη απόφαση:", apof_kala)

# save this for the tree later - ill need it
ev_kala_NH = ev_opt_kala_NH

# and if NH goes badly...
ev_symm_asxima_NH = p_kala_ST_dedom_asxima_NH * kerdos_ST + p_asxima_ST_dedom_asxima_NH * zimia_ST
ev_opt_asxima_NH = max(ev_symm_asxima_NH, 0)
print("\n--- Μετά από Άσχημα στο N.H. ---")
print(f"EV(Συμμετοχή S.T. | Άσχημα N.H.) = (1/7)×16 + (6/7)×(-10) = {ev_symm_asxima_NH:.4f}")
apof_asxima = "Συμμετοχή" if ev_symm_asxima_NH > 0 else "Μη Συμμετοχή"
print("Βέλτιστη απόφαση:", apof_asxima)

ev_asxima_NH = ev_opt_asxima_NH  # for tree

# overall EV with NH
ev_me_NH = p_kala_NH * ev_opt_kala_NH + p_asxima_NH * ev_opt_asxima_NH - kostos_NH
print("\n--- Συνολική Αξιολόγηση N.H. ---")
print("EV(N.H.) = P(Καλά N.H.)×EV(Καλά) + P(Άσχημα N.H.)×EV(Άσχημα) - Κόστος N.H.")
print(f"EV(N.H.) = ({p_kala_NH:.4f})×({ev_opt_kala_NH:.4f}) + ({p_asxima_NH:.4f})×({ev_opt_asxima_NH:.4f}) - {kostos_NH}")
print(f"EV(N.H.) = {ev_me_NH:.4f} εκατ. $")

print("\n--- ΤΕΛΙΚΗ ΑΠΟΦΑΣΗ ---")
print(f"EV(Μη συμμετοχή N.H.) = {ev_xoris_NH:.4f} εκατ. $")
print(f"EV(Συμμετοχή N.H.) = {ev_me_NH:.4f} εκατ. $")
optimal_1a = max(ev_xoris_NH, ev_me_NH)
if ev_me_NH > ev_xoris_NH:
    print("ΒΕΛΤΙΣΤΗ ΠΟΛΙΤΙΚΗ: Συμμετοχή στο N.H., μετά:")
    print("  - Αν καλά στο N.H. → Συμμετοχή στο S.T.")
    print("  - Αν άσχημα στο N.H. → Μη συμμετοχή στο S.T.")
else:
    print("ΒΕΛΤΙΣΤΗ ΠΟΛΙΤΙΚΗ: Απευθείας στο S.T. χωρίς N.H.")

# --- sensitivity analysis ---
print("\n--- Ανάλυση Ευαισθησίας (Μέρος b) ---")

gain_range = np.linspace(12, 20, 100)   # kerdos +-25%
loss_range = np.linspace(-12.5, -7.5, 100)  # zimia +-25%

# first vary gain, keep loss fixed at -10
best_g = []
xoris_g = []
me_g = []
i = 0
while i < len(gain_range):
    g = gain_range[i]
    val_no = max(p_kala_ST * g + p_asxima_ST * zimia_ST, 0)
    xoris_g.append(val_no)
    w = max(p_kala_ST_dedom_kala_NH * g + p_asxima_ST_dedom_kala_NH * zimia_ST, 0)
    p = max(p_kala_ST_dedom_asxima_NH * g + p_asxima_ST_dedom_asxima_NH * zimia_ST, 0)
    val_nh = p_kala_NH * w + p_asxima_NH * p - kostos_NH
    me_g.append(val_nh)
    if val_no > val_nh:
        best_g.append(val_no)
    else:
        best_g.append(val_nh)
    i += 1

fig_s, axes = plt.subplots(1, 2, figsize=(14, 5))
ar = axes[0]   # left plot
de = axes[1]   # right plot

ar.plot(gain_range, best_g, 'b-', lw=2, label='Βέλτιστη EV')
ar.plot(gain_range, xoris_g, 'g--', lw=1.5, label='Χωρίς N.H.')
ar.plot(gain_range, me_g, 'r--', lw=1.5, label='Με N.H.')
ar.axvline(x=16, color='gray', ls=':', alpha=0.7, label='Βασικό σενάριο ($16M)')
ar.set_xlabel('Κέρδος αν πάει καλά στο S.T. (εκατ. $)', fontsize=11)
ar.set_ylabel('Αναμενόμενη Αποδοχή (εκατ. $)', fontsize=11)
ar.set_title('Ανάλυση Ευαισθησίας - Μεταβολή Κέρδους', fontsize=12, fontweight='bold')
ar.legend(fontsize=9)
ar.grid(True, alpha=0.3)

# now vary loss, keep gain fixed at 16
best_l = []
xoris_l = []
me_l = []
for lo in loss_range:
    val_no = max(p_kala_ST * kerdos_ST + p_asxima_ST * lo, 0)
    xoris_l.append(val_no)
    w = max(p_kala_ST_dedom_kala_NH * kerdos_ST + p_asxima_ST_dedom_kala_NH * lo, 0)
    p = max(p_kala_ST_dedom_asxima_NH * kerdos_ST + p_asxima_ST_dedom_asxima_NH * lo, 0)
    val_nh = p_kala_NH * w + p_asxima_NH * p - kostos_NH
    me_l.append(val_nh)
    best_l.append(max(val_no, val_nh))

de.plot(loss_range, best_l, 'b-', lw=2, label='Βέλτιστη EV')
de.plot(loss_range, xoris_l, 'g--', lw=1.5, label='Χωρίς N.H.')
de.plot(loss_range, me_l, 'r--', lw=1.5, label='Με N.H.')
de.axvline(x=-10, color='gray', ls=':', alpha=0.7, label='Βασικό σενάριο (-$10M)')
de.set_xlabel('Ζημία αν πάει άσχημα στο S.T. (εκατ. $)', fontsize=11)
de.set_ylabel('Αναμενόμενη Αποδοχή (εκατ. $)', fontsize=11)
de.set_title('Ανάλυση Ευαισθησίας - Μεταβολή Ζημίας', fontsize=12, fontweight='bold')
de.legend(fontsize=9)
de.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(figdir + 'ex1a_sensitivity.png', dpi=150, bbox_inches='tight')
plt.close()
print("Αποθηκεύτηκε: ex1a_sensitivity.png")

# draw the decision tree for 1A
# this is gonna be a bit messy but whatever
fig_tree, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(-0.5, 10)
ax.set_ylim(-1, 9)
ax.axis('off')
ax.set_title('Δέντρο Απόφασης - Chelsea Bush (Άσκηση 1A)',
             fontsize=14, fontweight='bold', pad=20)

# D1 root node
vale_tetragono(ax, 0.5, 4.5, 0.3)
ax.text(0.5, 4.5, 'D1', ha='center', va='center',
        fontsize=8, fontweight='bold', color='white', zorder=6)

# branch to NH
ax.annotate('', xy=(2.5, 7), xytext=(0.65, 4.65),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.text(1.2, 6.2, 'Συμμετοχή N.H.\n(-$1.6M)', fontsize=8, ha='center', color='#1565C0')

# no NH branch
ax.annotate('', xy=(2.5, 2), xytext=(0.65, 4.35),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.text(1.2, 2.8, 'Όχι N.H.', fontsize=8, ha='center', color='#1565C0')

# chance node for NH result
chance_circle(ax, 2.7, 7, 0.15)
ax.text(2.7, 7, 'C1', ha='center', va='center', fontsize=7, fontweight='bold', zorder=6)

# good NH outcome
ax.annotate('', xy=(4.5, 8.2), xytext=(2.85, 7.1),
            arrowprops=dict(arrowstyle='->', color='green', lw=1.2))
ax.text(3.4, 8.0, 'Καλά N.H.\nP=7/15', fontsize=7, ha='center', color='green')

# bad NH outcome
ax.annotate('', xy=(4.5, 5.8), xytext=(2.85, 6.9),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.2))
ax.text(3.4, 6.1, 'Άσχημα N.H.\nP=8/15', fontsize=7, ha='center', color='red')

# D2: decision after good NH
decision_box(ax, 4.7, 8.2, 0.25)
ax.text(4.7, 8.2, 'D2', ha='center', va='center', fontsize=7, fontweight='bold', color='white', zorder=6)

# participate in ST after good NH
ax.annotate('', xy=(6.5, 8.7), xytext=(4.85, 8.35),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
ax.text(5.5, 8.8, 'Συμμ. S.T.', fontsize=7, ha='center')

chance_circle(ax, 6.7, 8.7, 0.12)

ax.annotate('', xy=(8.5, 8.9), xytext=(6.82, 8.78),
            arrowprops=dict(arrowstyle='->', color='green', lw=1))
ax.text(7.5, 9.05, 'Καλά P=4/7', fontsize=6, color='green')
ax.text(8.6, 8.9, '+$16M', fontsize=7, fontweight='bold', color='green')

ax.annotate('', xy=(8.5, 8.4), xytext=(6.82, 8.62),
            arrowprops=dict(arrowstyle='->', color='red', lw=1))
ax.text(7.5, 8.35, 'Άσχημα P=3/7', fontsize=6, color='red')
ax.text(8.6, 8.4, '-$10M', fontsize=7, fontweight='bold', color='red')

# dont participate after good NH
ax.annotate('', xy=(8.5, 7.7), xytext=(4.85, 8.05),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
ax.text(6.5, 7.6, 'Μη Συμμ. S.T.', fontsize=7, ha='center')
ax.text(8.6, 7.7, '$0', fontsize=7, fontweight='bold')

# D3: decision after bad NH
decision_box(ax, 4.7, 5.8, 0.25)
ax.text(4.7, 5.8, 'D3', ha='center', va='center', fontsize=7, fontweight='bold', color='white', zorder=6)

ax.annotate('', xy=(6.5, 6.3), xytext=(4.85, 5.95),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
ax.text(5.5, 6.4, 'Συμμ. S.T.', fontsize=7, ha='center')

chance_circle(ax, 6.7, 6.3, 0.12)

ax.annotate('', xy=(8.5, 6.5), xytext=(6.82, 6.38),
            arrowprops=dict(arrowstyle='->', color='green', lw=1))
ax.text(7.5, 6.6, 'Καλά P=1/7', fontsize=6, color='green')
ax.text(8.6, 6.5, '+$16M', fontsize=7, fontweight='bold', color='green')

ax.annotate('', xy=(8.5, 6.0), xytext=(6.82, 6.22),
            arrowprops=dict(arrowstyle='->', color='red', lw=1))
ax.text(7.5, 5.85, 'Άσχημα P=6/7', fontsize=6, color='red')
ax.text(8.6, 6.0, '-$10M', fontsize=7, fontweight='bold', color='red')

# dont participate after bad NH
ax.annotate('', xy=(8.5, 5.3), xytext=(4.85, 5.65),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
ax.text(6.5, 5.2, 'Μη Συμμ. S.T.', fontsize=7, ha='center')
ax.text(8.6, 5.3, '$0', fontsize=7, fontweight='bold')

# D4: no NH branch
decision_box(ax, 2.7, 2, 0.25)
ax.text(2.7, 2, 'D4', ha='center', va='center', fontsize=7, fontweight='bold', color='white', zorder=6)

ax.annotate('', xy=(4.5, 2.8), xytext=(2.85, 2.1),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
ax.text(3.5, 2.8, 'Συμμ. S.T.', fontsize=7, ha='center')

chance_circle(ax, 4.7, 2.8, 0.12)
ax.annotate('', xy=(6.5, 3.1), xytext=(4.82, 2.88),
            arrowprops=dict(arrowstyle='->', color='green', lw=1))
ax.text(5.5, 3.2, 'Καλά P=0.4', fontsize=6, color='green')
ax.text(6.6, 3.1, '+$16M', fontsize=7, fontweight='bold', color='green')

ax.annotate('', xy=(6.5, 2.5), xytext=(4.82, 2.72),
            arrowprops=dict(arrowstyle='->', color='red', lw=1))
ax.text(5.5, 2.35, 'Άσχημα P=0.6', fontsize=6, color='red')
ax.text(6.6, 2.5, '-$10M', fontsize=7, fontweight='bold', color='red')

ax.annotate('', xy=(6.5, 1.2), xytext=(2.85, 1.9),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
ax.text(4.5, 1.2, 'Μη Συμμ. S.T.', fontsize=7, ha='center')
ax.text(6.6, 1.2, '$0', fontsize=7, fontweight='bold')

# add EV labels
ax.text(0.5, 3.9, f'EV={max(ev_xoris_NH, ev_me_NH):.2f}M', fontsize=7, ha='center',
        fontweight='bold', color='purple', bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow'))
ax.text(2.7, 7.4, f'EV={p_kala_NH*ev_kala_NH+p_asxima_NH*ev_asxima_NH:.2f}M', fontsize=6, ha='center',
        color='purple', bbox=dict(boxstyle='round,pad=0.15', facecolor='lightyellow'))
ax.text(2.7, 1.5, f'EV={ev_xoris_NH:.2f}M', fontsize=6, ha='center',
        color='purple', bbox=dict(boxstyle='round,pad=0.15', facecolor='lightyellow'))

legend_items = [
    mpatches.Patch(facecolor='#4CAF50', edgecolor='black', label='Κόμβος Απόφασης'),
    mpatches.Patch(facecolor='#FF9800', edgecolor='black', label='Κόμβος Τύχης'),
]
ax.legend(handles=legend_items, loc='lower right', fontsize=9)

plt.savefig(FIGDIR + 'ex1a_tree.png', dpi=150, bbox_inches='tight')
plt.close()
print("Αποθηκεύτηκε: ex1a_tree.png")


# ==== EXERCISE 2 ====
print("\n" + "=" * 70)
print("ΑΣΚΗΣΗ 2: Steeley Associates, Inc.")
print("=" * 70)

# data
timi_polisis = 900_000
p_anaptiksi = 0.70
grafeia_anaptiksi = 1_300_000
grafeia_xoris = 200_000

p_adeia_ok = 0.10
apodosi_diamerismata = 3_000_000
p_adeia_reject = 0.90

polisi_meta_aporripsi = 700_000

dikastika = 300_000
p_niki = 0.40
p_ekremi = 0.10
p_itta = 1 - p_niki - p_ekremi   # should be 0.50

apozimioosi = 1_000_000
diam_meta_niki = 3_000_000
epipleon_exoda = 200_000

# after losing the lawsuit, new decision
# these values seem weird but whatever, thats what the problem says
p_mellontiki_anaptiksi = 0.50
polisi_me_anaptiksi = 900000
polisi_xoris_anaptiksi = 500000
grafeia_me_anaptiksi = 1200000
grafeia_xoris_anaptiksi = 100000

print("\n--- Υπολογισμοί Steeley Associates ---")

ev_polisi = timi_polisis
print(f"EV(Πώληση) = ${ev_polisi:,.0f}")

ev_grafeia = p_anaptiksi * grafeia_anaptiksi + (1 - p_anaptiksi) * grafeia_xoris
print(f"EV(Κτίριο Γραφείων) = 0.70×$1,300,000 + 0.30×$200,000 = ${ev_grafeia:,.0f}")

# need these for tree - save as english names too
sell_price = timi_polisis
ev_office = ev_grafeia

# calculate after losing lawsuit
ev_polisi_meta_itta = p_mellontiki_anaptiksi * polisi_me_anaptiksi + (1 - p_mellontiki_anaptiksi) * polisi_xoris_anaptiksi
print(f"EV(Πώληση μετά ήττα) = 0.50×$900,000 + 0.50×$500,000 = ${ev_polisi_meta_itta:,.0f}")

ev_grafeia_meta_itta = p_mellontiki_anaptiksi * grafeia_me_anaptiksi + (1 - p_mellontiki_anaptiksi) * grafeia_xoris_anaptiksi
print(f"EV(Γραφεία μετά ήττα) = 0.50×$1,200,000 + 0.50×$100,000 = ${ev_grafeia_meta_itta:,.0f}")

if ev_polisi_meta_itta >= ev_grafeia_meta_itta:
    ev_meta_itta = ev_polisi_meta_itta
    apof_meta_itta = "Πώληση"
else:
    ev_meta_itta = ev_grafeia_meta_itta
    apof_meta_itta = "Γραφεία"
print(f"Βέλτιστη απόφαση μετά ήττα: {apof_meta_itta} = ${ev_meta_itta:,.0f}")

# EV of lawsuit
# hmm let me think about this formula...
# if we win: get compensation + can build apartments
# if pending: just extra costs
# if lose: best option after
ev_minisi = p_niki * (apozimioosi + diam_meta_niki) + \
            p_ekremi * (0 - epipleon_exoda) + \
            p_itta * ev_meta_itta - dikastika
print(f"EV(Μήνυση) = 0.40×$4,000,000 + 0.10×(-$200,000) + 0.50×${ev_meta_itta:,.0f} - $300,000")
print(f"EV(Μήνυση) = ${ev_minisi:,.0f}")

ev_suit = ev_minisi  # for tree

# after rejection - find best option
epiloges_meta_aporr = [polisi_meta_aporripsi, ev_grafeia, ev_minisi]
kalitero_meta_aporr = max(epiloges_meta_aporr)
print("\nΜετά απόρριψη:")
print(f"  Πώληση: ${polisi_meta_aporripsi:,.0f}")
print(f"  Γραφεία: ${ev_grafeia:,.0f}")
print(f"  Μήνυση: ${ev_minisi:,.0f}")
print(f"  Βέλτιστη: ${kalitero_meta_aporr:,.0f}")

best_after_reject_val = kalitero_meta_aporr   # save for tree

# permit application
ev_aitisi = p_adeia_ok * apodosi_diamerismata + p_adeia_reject * kalitero_meta_aporr
print(f"\nEV(Αίτηση Άδειας) = 0.10×$3,000,000 + 0.90×${kalitero_meta_aporr:,.0f} = ${ev_aitisi:,.0f}")

ev_permit = ev_aitisi  # for tree

# final decision for steeley
print("\n--- ΤΕΛΙΚΗ ΑΠΟΦΑΣΗ Steeley ---")
print(f"EV(Πώληση) = ${ev_polisi:,.0f}")
print(f"EV(Γραφεία) = ${ev_grafeia:,.0f}")
print(f"EV(Αίτηση Άδειας) = ${ev_aitisi:,.0f}")
kalitero_steeley = max(ev_polisi, ev_grafeia, ev_aitisi)
best_steeley = kalitero_steeley  # alias for tree
if kalitero_steeley == ev_polisi:
    print("ΒΕΛΤΙΣΤΗ ΑΡΧΙΚΗ ΑΠΟΦΑΣΗ: Πώληση ακινήτου")
elif kalitero_steeley == ev_grafeia:
    print("ΒΕΛΤΙΣΤΗ ΑΡΧΙΚΗ ΑΠΟΦΑΣΗ: Κατασκευή κτιρίου γραφείων")
else:
    print("ΒΕΛΤΙΣΤΗ ΑΡΧΙΚΗ ΑΠΟΦΑΣΗ: Αίτηση άδειας διαμερισμάτων")
print(f"Αναμενόμενη αξία: ${kalitero_steeley:,.0f}")

# draw the tree for steeley (simplified)
fig, ax = plt.subplots(figsize=(18, 12))
ax.set_xlim(-1, 16)
ax.set_ylim(-1, 13)
ax.axis('off')
ax.set_title('Δέντρο Απόφασης - Steeley Associates (Άσκηση 2)', fontsize=14, fontweight='bold')

# D1 - root
draw_decision_node(ax, 0.5, 6.5, 0.4)
ax.text(0.5, 6.5, 'D1', ha='center', va='center', fontsize=9, fontweight='bold', color='white', zorder=6)
ax.text(0.5, 5.9, f'EV=${best_steeley/1e6:.2f}M', fontsize=7, ha='center', color='purple',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow'))

# Branch 1: Sell
ax.annotate('', xy=(3, 11), xytext=(0.7, 6.7),
            arrowprops=dict(arrowstyle='->', lw=1.5))
ax.text(1.5, 9.5, 'Πώληση', fontsize=8, ha='center', color='#1565C0', fontweight='bold')
ax.text(3.2, 11, f'${sell_price/1e6:.1f}M', fontsize=8, fontweight='bold', color='green')

# Branch 2: Office building
ax.annotate('', xy=(3, 9), xytext=(0.7, 6.6),
            arrowprops=dict(arrowstyle='->', lw=1.5))
ax.text(1.5, 8.2, 'Γραφεία', fontsize=8, ha='center', color='#1565C0', fontweight='bold')
draw_chance_node(ax, 3.2, 9, 0.15)
ax.text(3.2, 9, 'C', fontsize=7, ha='center', va='center', fontweight='bold', zorder=6)

ax.annotate('', xy=(5, 9.8), xytext=(3.35, 9.1),
            arrowprops=dict(arrowstyle='->', color='green', lw=1))
ax.text(4, 9.7, 'Ανάπτ. 70%', fontsize=6, color='green')
ax.text(5.1, 9.8, '$1.3M', fontsize=7, fontweight='bold', color='green')

ax.annotate('', xy=(5, 8.2), xytext=(3.35, 8.9),
            arrowprops=dict(arrowstyle='->', color='red', lw=1))
ax.text(4, 8.2, 'Όχι ανάπτ. 30%', fontsize=6, color='red')
ax.text(5.1, 8.2, '$0.2M', fontsize=7, fontweight='bold', color='red')

ax.text(3.2, 8.5, f'EV=${ev_office/1e6:.2f}M', fontsize=6, ha='center', color='purple',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='lightyellow'))

# Branch 3: Permit application
ax.annotate('', xy=(3, 4), xytext=(0.7, 6.3),
            arrowprops=dict(arrowstyle='->', lw=1.5))
ax.text(1.2, 4.8, 'Αίτηση\nΆδειας', fontsize=8, ha='center', color='#1565C0', fontweight='bold')

draw_chance_node(ax, 3.2, 4, 0.15)
ax.text(3.2, 4, 'C', fontsize=7, ha='center', va='center', fontweight='bold', zorder=6)
ax.text(3.2, 3.5, f'EV=${ev_permit/1e6:.2f}M', fontsize=6, ha='center', color='purple',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='lightyellow'))

# Approval
ax.annotate('', xy=(5.5, 5.5), xytext=(3.35, 4.1),
            arrowprops=dict(arrowstyle='->', color='green', lw=1))
ax.text(4, 5.2, 'Έγκριση 10%', fontsize=6, color='green')
ax.text(5.6, 5.5, '$3.0M', fontsize=7, fontweight='bold', color='green')

# Rejection -> D2
ax.annotate('', xy=(5.5, 2.5), xytext=(3.35, 3.9),
            arrowprops=dict(arrowstyle='->', color='red', lw=1))
ax.text(4, 2.8, 'Απόρριψη 90%', fontsize=6, color='red')

draw_decision_node(ax, 5.7, 2.5, 0.3)
ax.text(5.7, 2.5, 'D2', ha='center', va='center', fontsize=8, fontweight='bold', color='white', zorder=6)
ax.text(5.7, 1.9, f'EV=${best_after_reject_val/1e6:.2f}M', fontsize=6, ha='center', color='purple',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='lightyellow'))

# after rejection: sell
ax.annotate('', xy=(8, 4.5), xytext=(5.85, 2.65),
            arrowprops=dict(arrowstyle='->', lw=1))
ax.text(6.7, 4.0, 'Πώληση', fontsize=7, ha='center')
ax.text(8.1, 4.5, '$0.7M', fontsize=7, fontweight='bold')

# after rejection: office
ax.annotate('', xy=(8, 3.2), xytext=(5.85, 2.55),
            arrowprops=dict(arrowstyle='->', lw=1))
ax.text(6.7, 3.1, 'Γραφεία', fontsize=7, ha='center')
ax.text(8.1, 3.2, f'EV=${ev_office/1e6:.2f}M', fontsize=7, fontweight='bold')

# after rejection: lawsuit
ax.annotate('', xy=(8, 1.5), xytext=(5.85, 2.35),
            arrowprops=dict(arrowstyle='->', lw=1))
ax.text(6.7, 1.6, f'Μήνυση\n(-$300K)', fontsize=7, ha='center')

draw_chance_node(ax, 8.2, 1.5, 0.15)
ax.text(8.2, 1.0, f'EV=${ev_suit/1e6:.2f}M', fontsize=6, ha='center', color='purple',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='lightyellow'))

# lawsuit outcomes
ax.annotate('', xy=(10.5, 2.5), xytext=(8.35, 1.6),
            arrowprops={'arrowstyle': '->', 'color': 'green', 'lw': 1})
ax.text(9.2, 2.3, 'Κέρδος 40%', fontsize=6, color='green')
ax.text(10.6, 2.5, '$4.0M', fontsize=7, fontweight='bold', color='green')

ax.annotate('', xy=(10.5, 1.5), xytext=(8.35, 1.5),
            arrowprops={'arrowstyle': '->', 'color': 'gray', 'lw': 1})
ax.text(9.2, 1.55, 'Εκκρεμεί 10%', fontsize=6, color='gray')
ax.text(10.6, 1.5, '-$0.2M', fontsize=7, fontweight='bold', color='gray')

ax.annotate('', xy=(10.5, 0.5), xytext=(8.35, 1.4),
            arrowprops={'arrowstyle': '->', 'color': 'red', 'lw': 1})
ax.text(9.2, 0.6, 'Ήττα 50%', fontsize=6, color='red')

vale_tetragono(ax, 10.7, 0.5, 0.25)
ax.text(10.7, 0.5, 'D3', ha='center', va='center', fontsize=7,
        fontweight='bold', color='white', zorder=6)
ax.text(10.7, 0.0, f'EV=${ev_meta_itta/1e6:.2f}M', fontsize=6,
        ha='center', color='purple',
        bbox={'boxstyle': 'round,pad=0.15', 'facecolor': 'lightyellow'})

ypomnima = [
    mpatches.Patch(facecolor='#4CAF50', edgecolor='k', label='Κόμβος Απόφασης'),
    mpatches.Patch(facecolor='#FF9800', edgecolor='k', label='Κόμβος Τύχης'),
]
ax.legend(handles=ypomnima, loc='lower right', fontsize=9)

plt.savefig(figdir + 'ex2_steeley_tree.png', dpi=150, bbox_inches='tight')
plt.close()
print("Αποθηκεύτηκε: ex2_steeley_tree.png")

# ====
print("\n" + "=" * 70)
print("ΑΣΚΗΣΗ 3: BAAG - Smart Steering Support (DSS)")
print("=" * 70)
# ====

# Problem data
research_cost = 300000
dev_cost = 800000
marketing_cost = 200000
p_research_success = 0.80
p_dev_failure = 0.35
p_dev_success = 0.65

sell_current_dss = 2000000  # sell DSS as is
gm_tech_sale = 200000      # sell tech to GM
gm_concept_sale = 1000000  # sell concept after successful dev

# Market acceptance
p_high = 0.30
p_medium = 0.50
p_low = 0.20
profit_high = 8000000
profit_medium = 4000000
profit_low = 2200000

print("\n--- Υπολογισμοί BAAG/DSS ---")

# If no research: sell DSS as is
ev_no_research = sell_current_dss
print(f"EV(Χωρίς Έρευνα) = Πώληση DSS = ${ev_no_research:,.0f}")

# With research (cost $300K):
# - Research fails (20%): sell DSS = $2M
# - Research succeeds (80%):
#   - Don't develop: sell DSS + GM tech = $2M + $200K = $2.2M
#   - Develop (cost $800K):
#     - Dev fails (35%): sell DSS + GM tech = $2.2M
#     - Dev succeeds (65%):
#       - Don't launch: sell concept = $1M
#       - Launch (cost $200K):
#         High (30%): $8M, Medium (50%): $4M, Low (20%): $2.2M

# EV(Marketing)
ev_market = p_high * profit_high + p_medium * profit_medium + p_low * profit_low
print(f"EV(Αποδοχή αγοράς) = 0.30×$8M + 0.50×$4M + 0.20×$2.2M = ${ev_market:,.0f}")

# If we launch (after marketing cost)
ev_launch = ev_market - marketing_cost
print(f"EV(Κυκλοφορία) = ${ev_market:,.0f} - ${marketing_cost:,.0f} = ${ev_launch:,.0f}")

# Decision: Launch or sell concept ($1M)
ev_after_dev_success = max(ev_launch, gm_concept_sale)
decision_dev_success = "Κυκλοφορία" if ev_launch >= gm_concept_sale else "Πώληση concept"
print(f"Μετά επιτυχή ανάπτυξη: max(${ev_launch:,.0f}, ${gm_concept_sale:,.0f}) → {decision_dev_success} = ${ev_after_dev_success:,.0f}")

# EV(Development)
ev_dev_fail = sell_current_dss + gm_tech_sale  # $2.2M
ev_development = p_dev_success * ev_after_dev_success + p_dev_failure * ev_dev_fail - dev_cost
print(f"EV(Ανάπτυξη) = 0.65×${ev_after_dev_success:,.0f} + 0.35×${ev_dev_fail:,.0f} - ${dev_cost:,.0f} = ${ev_development:,.0f}")

# Decision after successful research
ev_no_develop = sell_current_dss + gm_tech_sale  # $2.2M
ev_after_research_success = max(ev_development, ev_no_develop)
decision_research = "Ανάπτυξη" if ev_development >= ev_no_develop else "Πώληση τεχνολογίας"
print(f"Μετά επιτυχή έρευνα: max(${ev_development:,.0f}, ${ev_no_develop:,.0f}) → {decision_research} = ${ev_after_research_success:,.0f}")

# EV(Research)
ev_research_fail = sell_current_dss  # $2M
ev_research = p_research_success * ev_after_research_success + (1 - p_research_success) * ev_research_fail - research_cost
print(f"EV(Έρευνα) = 0.80×${ev_after_research_success:,.0f} + 0.20×${ev_research_fail:,.0f} - ${research_cost:,.0f} = ${ev_research:,.0f}")

# Final decision
print(f"\n--- ΤΕΛΙΚΗ ΑΠΟΦΑΣΗ BAAG ---")
print(f"EV(Χωρίς Έρευνα) = ${ev_no_research:,.0f}")
print(f"EV(Με Έρευνα) = ${ev_research:,.0f}")
best_baag = max(ev_no_research, ev_research)
if ev_research >= ev_no_research:
    print(f"ΒΕΛΤΙΣΤΗ ΠΟΛΙΤΙΚΗ: Επένδυση στην Έρευνα → Ανάπτυξη → Κυκλοφορία")
    print(f"Αναμενόμενη αξία: ${ev_research:,.0f}")
else:
    print(f"ΒΕΛΤΙΣΤΗ ΠΟΛΙΤΙΚΗ: Πώληση DSS ως έχει")

# tree for BAAG
fig, ax = plt.subplots(figsize=(18, 11))
ax.set_xlim(-0.5, 16)
ax.set_ylim(-1, 12)
ax.axis('off')
ax.set_title('Δέντρο Απόφασης - BAAG/DSS (Άσκηση 3)', fontsize=14, fontweight='bold')

# D1: Research or not
draw_decision_node(ax, 0.5, 5.5, 0.35)
ax.text(0.5, 5.5, 'D1', ha='center', va='center', fontsize=9, fontweight='bold', color='white', zorder=6)
ax.text(0.5, 4.9, f'EV=${best_baag/1e6:.2f}M', fontsize=7, ha='center', color='purple',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow'))

# No research
ax.annotate('', xy=(3, 2), xytext=(0.7, 5.3),
            arrowprops=dict(arrowstyle='->', lw=1.5))
ax.text(1.5, 3.2, 'Χωρίς Έρευνα', fontsize=8, color='#1565C0', fontweight='bold')
ax.text(3.1, 2, f'${sell_current_dss/1e6:.1f}M', fontsize=8, fontweight='bold', color='green')

# With research
ax.annotate('', xy=(3, 8), xytext=(0.7, 5.7),
            arrowprops=dict(arrowstyle='->', lw=1.5))
ax.text(1.3, 7.3, f'Έρευνα\n(-$300K)', fontsize=8, color='#1565C0', fontweight='bold')

# C1: Research result
draw_chance_node(ax, 3.2, 8, 0.15)
ax.text(3.2, 8, 'C1', ha='center', va='center', fontsize=7, fontweight='bold', zorder=6)

# Research fails
ax.annotate('', xy=(5, 6), xytext=(3.35, 7.9),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.2))
ax.text(3.9, 6.6, 'Αποτυχία 20%', fontsize=7, color='red')
ax.text(5.1, 6, f'${sell_current_dss/1e6:.1f}M', fontsize=7, fontweight='bold', color='red')

# Research succeeds → D2
ax.annotate('', xy=(5, 9.5), xytext=(3.35, 8.1),
            arrowprops=dict(arrowstyle='->', color='green', lw=1.2))
ax.text(3.9, 9.1, 'Επιτυχία 80%', fontsize=7, color='green')

draw_decision_node(ax, 5.2, 9.5, 0.3)
ax.text(5.2, 9.5, 'D2', ha='center', va='center', fontsize=8, fontweight='bold', color='white', zorder=6)

# Sell tech
ax.annotate('', xy=(7, 7.5), xytext=(5.35, 9.35),
            arrowprops=dict(arrowstyle='->', lw=1.2))
ax.text(5.9, 8.1, 'Πώληση τεχν.\n(DSS+GM)', fontsize=7)
ax.text(7.1, 7.5, f'${ev_no_develop/1e6:.1f}M', fontsize=7, fontweight='bold')

# Develop → C2
ax.annotate('', xy=(7, 10.5), xytext=(5.35, 9.65),
            arrowprops=dict(arrowstyle='->', lw=1.2))
ax.text(5.8, 10.4, f'Ανάπτυξη\n(-$800K)', fontsize=7)

draw_chance_node(ax, 7.2, 10.5, 0.15)
ax.text(7.2, 10.5, 'C2', ha='center', va='center', fontsize=7, fontweight='bold', zorder=6)

# Dev fails
ax.annotate('', xy=(9, 9.5), xytext=(7.35, 10.4),
            arrowprops=dict(arrowstyle='->', color='red', lw=1))
ax.text(7.9, 9.6, 'Αποτ. 35%', fontsize=6, color='red')
ax.text(9.1, 9.5, f'${ev_dev_fail/1e6:.1f}M', fontsize=7, fontweight='bold', color='red')

# Dev succeeds → D3
ax.annotate('', xy=(9, 11.2), xytext=(7.35, 10.6),
            arrowprops=dict(arrowstyle='->', color='green', lw=1))
ax.text(7.9, 11.0, 'Επιτ. 65%', fontsize=6, color='green')

draw_decision_node(ax, 9.2, 11.2, 0.25)
ax.text(9.2, 11.2, 'D3', ha='center', va='center', fontsize=7, fontweight='bold', color='white', zorder=6)

# Sell concept
ax.annotate('', xy=(11, 10.3), xytext=(9.35, 11.1),
            arrowprops=dict(arrowstyle='->', lw=1))
ax.text(9.9, 10.4, f'Πωλ.concept', fontsize=6)
ax.text(11.1, 10.3, f'${gm_concept_sale/1e6:.1f}M', fontsize=7, fontweight='bold')

# Launch → C3
ax.annotate('', xy=(11, 11.5), xytext=(9.35, 11.3),
            arrowprops=dict(arrowstyle='->', lw=1))
ax.text(9.9, 11.7, f'Κυκλοφορία\n(-$200K)', fontsize=6)

draw_chance_node(ax, 11.2, 11.5, 0.12)

ax.annotate('', xy=(13, 11.8), xytext=(11.32, 11.58),
            arrowprops=dict(arrowstyle='->', color='green', lw=1))
ax.text(11.8, 11.9, 'Υψηλή 30%', fontsize=5, color='green')
ax.text(13.1, 11.8, f'$8.0M', fontsize=6, fontweight='bold', color='green')

ax.annotate('', xy=(13, 11.4), xytext=(11.32, 11.5),
            arrowprops=dict(arrowstyle='->', lw=1))
ax.text(11.8, 11.45, 'Μεσαία 50%', fontsize=5)
ax.text(13.1, 11.4, f'$4.0M', fontsize=6, fontweight='bold')

ax.annotate('', xy=(13, 11.0), xytext=(11.32, 11.42),
            arrowprops=dict(arrowstyle='->', color='red', lw=1))
ax.text(11.8, 11.0, 'Χαμηλή 20%', fontsize=5, color='red')
ax.text(13.1, 11.0, f'$2.2M', fontsize=6, fontweight='bold', color='red')

legend_elements = [
    mpatches.Patch(facecolor='#4CAF50', edgecolor='black', label='Κόμβος Απόφασης'),
    mpatches.Patch(facecolor='#FF9800', edgecolor='black', label='Κόμβος Τύχης'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
plt.savefig(FIGDIR + 'ex3_baag_tree.png', dpi=150, bbox_inches='tight')
plt.close()
print("Αποθηκεύτηκε: ex3_baag_tree.png")

# ====
print("\n" + "=" * 70)
print("ΑΣΚΗΣΗ 4: Carolina Cougars")
print("=" * 70)
# ====

# Strategy 1: No investment ($0)
cost_s1 = 0
p_contend_s1 = 0.25
p_out_s1 = 0.75

# If contender S1
p_high_c_s1 = 0.70; profit_high_c_s1 = 170
p_med_c_s1 = 0.25; profit_med_c_s1 = 115
p_low_c_s1 = 0.05; profit_low_c_s1 = 90

# If out S1
p_high_o_s1 = 0.05; profit_high_o_s1 = 95
p_med_o_s1 = 0.20; profit_med_o_s1 = 55
p_low_o_s1 = 0.75; profit_low_o_s1 = 30

ev_contend_s1 = p_high_c_s1*profit_high_c_s1 + p_med_c_s1*profit_med_c_s1 + p_low_c_s1*profit_low_c_s1
ev_out_s1 = p_high_o_s1*profit_high_o_s1 + p_med_o_s1*profit_med_o_s1 + p_low_o_s1*profit_low_o_s1
ev_s1 = p_contend_s1 * ev_contend_s1 + p_out_s1 * ev_out_s1 - cost_s1

print(f"\n--- Στρατηγική 1: Καμία Επένδυση ---")
print(f"EV(Contender) = {ev_contend_s1:.2f}M")
print(f"EV(Out) = {ev_out_s1:.2f}M")
print(f"EV(S1) = 0.25×{ev_contend_s1:.2f} + 0.75×{ev_out_s1:.2f} = {ev_s1:.2f}M")

# Strategy 2: Moderate investment ($20M)
cost_s2 = 20
p_contend_s2 = 0.50

# Contender → Stand pat or Buy
# Stand pat
ev_sp_c_s2 = 0.75*195 + 0.20*160 + 0.05*120
# Buy ($8M cost)
ev_buy_c_s2 = 0.80*200 + 0.15*170 + 0.05*125 - 8

ev_contend_s2 = max(ev_sp_c_s2, ev_buy_c_s2)
decision_c_s2 = "Buy" if ev_buy_c_s2 >= ev_sp_c_s2 else "Stand Pat"

print(f"\n--- Στρατηγική 2: Μέτρια Επένδυση ($20M) ---")
print(f"Contender - Stand Pat: EV = {ev_sp_c_s2:.2f}M")
print(f"Contender - Buy Players (-$8M): EV = {ev_buy_c_s2:.2f}M")
print(f"Βέλτιστη: {decision_c_s2} = {ev_contend_s2:.2f}M")

# Out → Stand pat or Sell
ev_sp_o_s2 = 0.12*110 + 0.28*65 + 0.60*40
ev_sell_o_s2 = 0.08*100 + 0.22*60 + 0.70*35 + 8  # +$8M from selling

ev_out_s2 = max(ev_sp_o_s2, ev_sell_o_s2)
decision_o_s2 = "Sell" if ev_sell_o_s2 >= ev_sp_o_s2 else "Stand Pat"

print(f"Out - Stand Pat: EV = {ev_sp_o_s2:.2f}M")
print(f"Out - Sell Players (+$8M): EV = {ev_sell_o_s2:.2f}M")
print(f"Βέλτιστη: {decision_o_s2} = {ev_out_s2:.2f}M")

ev_s2 = p_contend_s2 * ev_contend_s2 + (1 - p_contend_s2) * ev_out_s2 - cost_s2
print(f"EV(S2) = 0.50×{ev_contend_s2:.2f} + 0.50×{ev_out_s2:.2f} - {cost_s2} = {ev_s2:.2f}M")

# Strategy 3: Large investment ($52M)
cost_s3 = 52
p_contend_s3 = 0.65

# Contender → Stand pat or Buy ($10M)
ev_sp_c_s3 = 0.80*210 + 0.15*170 + 0.05*125
ev_buy_c_s3 = 0.83*220 + 0.12*175 + 0.05*130 - 10

ev_contend_s3 = max(ev_sp_c_s3, ev_buy_c_s3)
decision_c_s3 = "Buy" if ev_buy_c_s3 >= ev_sp_c_s3 else "Stand Pat"

print(f"\n--- Στρατηγική 3: Μεγάλη Επένδυση ($52M) ---")
print(f"Contender - Stand Pat: EV = {ev_sp_c_s3:.2f}M")
print(f"Contender - Buy Players (-$10M): EV = {ev_buy_c_s3:.2f}M")
print(f"Βέλτιστη: {decision_c_s3} = {ev_contend_s3:.2f}M")

# Out → Stand pat or Sell ($12M)
ev_sp_o_s3 = 0.15*110 + 0.30*70 + 0.55*50
ev_sell_o_s3 = 0.10*105 + 0.30*65 + 0.60*45 + 12

ev_out_s3 = max(ev_sp_o_s3, ev_sell_o_s3)
decision_o_s3 = "Sell" if ev_sell_o_s3 >= ev_sp_o_s3 else "Stand Pat"

print(f"Out - Stand Pat: EV = {ev_sp_o_s3:.2f}M")
print(f"Out - Sell Players (+$12M): EV = {ev_sell_o_s3:.2f}M")
print(f"Βέλτιστη: {decision_o_s3} = {ev_out_s3:.2f}M")

ev_s3 = p_contend_s3 * ev_contend_s3 + (1 - p_contend_s3) * ev_out_s3 - cost_s3
print(f"EV(S3) = 0.65×{ev_contend_s3:.2f} + 0.35×{ev_out_s3:.2f} - {cost_s3} = {ev_s3:.2f}M")

print(f"\n--- ΤΕΛΙΚΗ ΑΠΟΦΑΣΗ Carolina Cougars ---")
print(f"EV(Καμία Επένδυση) = {ev_s1:.2f}M")
print(f"EV(Μέτρια Επένδυση $20M) = {ev_s2:.2f}M")
print(f"EV(Μεγάλη Επένδυση $52M) = {ev_s3:.2f}M")
best_cougars = max(ev_s1, ev_s2, ev_s3)
if best_cougars == ev_s1:
    print("ΒΕΛΤΙΣΤΗ: Καμία Επένδυση")
elif best_cougars == ev_s2:
    print("ΒΕΛΤΙΣΤΗ: Μέτρια Επένδυση ($20M)")
else:
    print("ΒΕΛΤΙΣΤΗ: Μεγάλη Επένδυση ($52M)")
print(f"Αναμενόμενη αξία: {best_cougars:.2f}M")

# comparison chart
fig, ax = plt.subplots(figsize=(10, 6))
strategies = ['Καμία Επένδυση\n($0)', 'Μέτρια Επένδυση\n($20M)', 'Μεγάλη Επένδυση\n($52M)']
evs = [ev_s1, ev_s2, ev_s3]
colors = ['#4CAF50' if ev == best_cougars else '#2196F3' for ev in evs]
bars = ax.bar(strategies, evs, color=colors, edgecolor='black', linewidth=1.2)
for bar, ev in zip(bars, evs):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'${ev:.2f}M', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_ylabel('Αναμενόμενη Αξία (εκατ. $)', fontsize=12)
ax.set_title('Carolina Cougars - Σύγκριση Στρατηγικών (Άσκηση 4)', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(FIGDIR + 'ex4_cougars_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Αποθηκεύτηκε: ex4_cougars_comparison.png")

# Cougars decision tree
fig, ax = plt.subplots(figsize=(20, 14))
ax.set_xlim(-1, 18)
ax.set_ylim(-1, 16)
ax.axis('off')
ax.set_title('Δέντρο Απόφασης - Carolina Cougars (Άσκηση 4)', fontsize=14, fontweight='bold')

# root node
draw_decision_node(ax, 0.5, 8, 0.4)
ax.text(0.5, 8, 'D1', ha='center', va='center', fontsize=9, fontweight='bold', color='white', zorder=6)
ax.text(0.5, 7.3, f'EV=${best_cougars:.1f}M', fontsize=7, ha='center', color='purple',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow'))

# S1: No investment
ax.annotate('', xy=(2.5, 2), xytext=(0.7, 7.8),
            arrowprops=dict(arrowstyle='->', lw=1.5))
ax.text(1, 4.5, 'Καμία\nΕπένδυση', fontsize=7, color='#1565C0', fontweight='bold')

draw_chance_node(ax, 2.7, 2, 0.15)
ax.annotate('', xy=(4.5, 3), xytext=(2.85, 2.1), arrowprops=dict(arrowstyle='->', color='green', lw=1))
ax.text(3.4, 2.9, f'Contender 25%', fontsize=6, color='green')
ax.text(4.6, 3, f'EV={ev_contend_s1:.1f}M', fontsize=6)

ax.annotate('', xy=(4.5, 1), xytext=(2.85, 1.9), arrowprops=dict(arrowstyle='->', color='red', lw=1))
ax.text(3.4, 1.1, f'Out 75%', fontsize=6, color='red')
ax.text(4.6, 1, f'EV={ev_out_s1:.1f}M', fontsize=6)
ax.text(2.7, 1.4, f'EV={ev_s1:.1f}M', fontsize=6, ha='center', color='purple',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='lightyellow'))

# S2: Moderate
ax.annotate('', xy=(2.5, 8), xytext=(0.7, 8),
            arrowprops=dict(arrowstyle='->', lw=1.5))
ax.text(1.3, 8.3, f'Μέτρια\n(-$20M)', fontsize=7, color='#1565C0', fontweight='bold')

draw_chance_node(ax, 2.7, 8, 0.15)

# S2 Contender
ax.annotate('', xy=(4.5, 10), xytext=(2.85, 8.1), arrowprops=dict(arrowstyle='->', color='green', lw=1))
ax.text(3.4, 9.5, f'Contender 50%', fontsize=6, color='green')

draw_decision_node(ax, 4.7, 10, 0.2)
ax.text(4.7, 10, 'D', fontsize=6, ha='center', va='center', fontweight='bold', color='white', zorder=6)

ax.annotate('', xy=(6.5, 10.5), xytext=(4.8, 10.1), arrowprops=dict(arrowstyle='->', lw=1))
ax.text(5.4, 10.6, f'Stand Pat', fontsize=6)
ax.text(6.6, 10.5, f'EV={ev_sp_c_s2:.1f}M', fontsize=6)

ax.annotate('', xy=(6.5, 9.5), xytext=(4.8, 9.9), arrowprops=dict(arrowstyle='->', lw=1))
ax.text(5.4, 9.4, f'Buy(-$8M)', fontsize=6)
ax.text(6.6, 9.5, f'EV={ev_buy_c_s2:.1f}M', fontsize=6)

# S2 Out
ax.annotate('', xy=(4.5, 6), xytext=(2.85, 7.9), arrowprops=dict(arrowstyle='->', color='red', lw=1))
ax.text(3.4, 6.5, f'Out 50%', fontsize=6, color='red')

draw_decision_node(ax, 4.7, 6, 0.2)
ax.text(4.7, 6, 'D', fontsize=6, ha='center', va='center', fontweight='bold', color='white', zorder=6)

ax.annotate('', xy=(6.5, 6.5), xytext=(4.8, 6.1), arrowprops=dict(arrowstyle='->', lw=1))
ax.text(5.4, 6.6, f'Stand Pat', fontsize=6)
ax.text(6.6, 6.5, f'EV={ev_sp_o_s2:.1f}M', fontsize=6)

ax.annotate('', xy=(6.5, 5.5), xytext=(4.8, 5.9), arrowprops=dict(arrowstyle='->', lw=1))
ax.text(5.4, 5.4, f'Sell(+$8M)', fontsize=6)
ax.text(6.6, 5.5, f'EV={ev_sell_o_s2:.1f}M', fontsize=6)

ax.text(2.7, 7.4, f'EV={ev_s2:.1f}M', fontsize=6, ha='center', color='purple',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='lightyellow'))

# S3: Large
ax.annotate('', xy=(2.5, 14), xytext=(0.7, 8.2),
            arrowprops=dict(arrowstyle='->', lw=1.5))
ax.text(1, 11.5, f'Μεγάλη\n(-$52M)', fontsize=7, color='#1565C0', fontweight='bold')

draw_chance_node(ax, 2.7, 14, 0.15)

ax.annotate('', xy=(4.5, 15), xytext=(2.85, 14.1), arrowprops=dict(arrowstyle='->', color='green', lw=1))
ax.text(3.4, 15, f'Contender 65%', fontsize=6, color='green')

draw_decision_node(ax, 4.7, 15, 0.2)
ax.text(4.7, 15, 'D', fontsize=6, ha='center', va='center', fontweight='bold', color='white', zorder=6)

ax.annotate('', xy=(6.5, 15.5), xytext=(4.8, 15.1), arrowprops=dict(arrowstyle='->', lw=1))
ax.text(5.4, 15.6, f'Stand Pat', fontsize=6)
ax.text(6.6, 15.5, f'EV={ev_sp_c_s3:.1f}M', fontsize=6)

ax.annotate('', xy=(6.5, 14.5), xytext=(4.8, 14.9), arrowprops=dict(arrowstyle='->', lw=1))
ax.text(5.4, 14.4, f'Buy(-$10M)', fontsize=6)
ax.text(6.6, 14.5, f'EV={ev_buy_c_s3:.1f}M', fontsize=6)

ax.annotate('', xy=(4.5, 13), xytext=(2.85, 13.9), arrowprops=dict(arrowstyle='->', color='red', lw=1))
ax.text(3.4, 13, f'Out 35%', fontsize=6, color='red')

draw_decision_node(ax, 4.7, 13, 0.2)
ax.text(4.7, 13, 'D', fontsize=6, ha='center', va='center', fontweight='bold', color='white', zorder=6)

ax.annotate('', xy=(6.5, 13.5), xytext=(4.8, 13.1), arrowprops=dict(arrowstyle='->', lw=1))
ax.text(5.4, 13.6, f'Stand Pat', fontsize=6)
ax.text(6.6, 13.5, f'EV={ev_sp_o_s3:.1f}M', fontsize=6)

ax.annotate('', xy=(6.5, 12.5), xytext=(4.8, 12.9), arrowprops=dict(arrowstyle='->', lw=1))
ax.text(5.4, 12.4, f'Sell(+$12M)', fontsize=6)
ax.text(6.6, 12.5, f'EV={ev_sell_o_s3:.1f}M', fontsize=6)

ax.text(2.7, 13.4, f'EV={ev_s3:.1f}M', fontsize=6, ha='center', color='purple',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='lightyellow'))

legend_elements = [
    mpatches.Patch(facecolor='#4CAF50', edgecolor='black', label='Κόμβος Απόφασης'),
    mpatches.Patch(facecolor='#FF9800', edgecolor='black', label='Κόμβος Τύχης'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
plt.savefig(FIGDIR + 'ex4_cougars_tree.png', dpi=150, bbox_inches='tight')
plt.close()
print("Αποθηκεύτηκε: ex4_cougars_tree.png")

# ====
print("\n" + "=" * 70)
print("ΑΣΚΗΣΗ 5: Brainy Business (Cerebrosoft)")
print("=" * 70)
# ====

# Problem data
dev_cost_brain = 850000   # 800K dev + 50K support
annual_support = 0  # already included above
market_research_cost = 10000

# Competition probabilities (prior)
p_severe = 0.20
p_moderate = 0.70
p_weak = 0.10

# Prices
prices = [50, 40, 30]

# Sales probability tables
# Table 1: High price ($50)
prob_50 = {
    'severe':   {'50K': 0.20, '30K': 0.25, '20K': 0.55},
    'moderate': {'50K': 0.25, '30K': 0.30, '20K': 0.45},
    'weak':     {'50K': 0.30, '30K': 0.35, '20K': 0.35}
}

# Table 2: Medium price ($40)
prob_40 = {
    'severe':   {'50K': 0.25, '30K': 0.35, '20K': 0.40},
    'moderate': {'50K': 0.30, '30K': 0.40, '20K': 0.30},
    'weak':     {'50K': 0.40, '30K': 0.50, '20K': 0.10}
}

# Table 3: Low price ($30)
prob_30 = {
    'severe':   {'50K': 0.35, '30K': 0.40, '20K': 0.25},
    'moderate': {'50K': 0.40, '30K': 0.50, '20K': 0.10},
    'weak':     {'50K': 0.50, '30K': 0.45, '20K': 0.05}
}

prob_tables = {50: prob_50, 40: prob_40, 30: prob_30}
units = {'50K': 50000, '30K': 30000, '20K': 20000}
comp_probs = {'severe': p_severe, 'moderate': p_moderate, 'weak': p_weak}

print("\n--- Μέρος 1: Πίνακας Αποδοχών (χωρίς έρευνα αγοράς) ---")

# Payoff = revenue - dev_cost_brain
# Revenue = price × units
def calc_payoff(price, unit_key):
    """Υπολογισμός αποδοχής (κέρδος/ζημία) για δεδομένη τιμή και πωλήσεις"""
    revenue = price * units[unit_key]
    return revenue - dev_cost_brain

print("\nΠίνακας Αποδοχών (Payoff Table):")
print(f"{'Τιμή':>8} {'Πωλήσεις':>10} {'Έσοδα':>12} {'Κέρδος':>12}")
print("-" * 50)

payoff_table = {}
for price in prices:
    payoff_table[price] = {}
    for uk in ['50K', '30K', '20K']:
        pf = calc_payoff(price, uk)
        payoff_table[price][uk] = pf
        print(f"${price:>6} {uk:>10} ${price*units[uk]:>10,} ${pf:>10,}")

# EV for each price (integrated probability per competition)
print(f"\n--- Αναμενόμενες Αποδοχές (Bayes' Decision Rule) ---")

ev_prices = {}
for price in prices:
    ev = 0
    prob_t = prob_tables[price]
    for comp in ['severe', 'moderate', 'weak']:
        p_comp = comp_probs[comp]
        for uk in ['50K', '30K', '20K']:
            p_units = prob_t[comp][uk]
            pf = payoff_table[price][uk]
            ev += p_comp * p_units * pf
    ev_prices[price] = ev
    print(f"EV(Τιμή ${price}) = ${ev:,.0f}")

# No launch
ev_abandon = 0
print(f"EV(Εγκατάλειψη) = $0")

best_price_no_research = max(ev_prices, key=ev_prices.get)
best_ev_no_research = ev_prices[best_price_no_research]
decision_bayes = "Εγκατάλειψη" if ev_abandon > best_ev_no_research else f"Κυκλοφορία σε ${best_price_no_research}"

print(f"\n--- Απόφαση Bayes ---")
print(f"Βέλτιστη τιμή: ${best_price_no_research} με EV = ${best_ev_no_research:,.0f}")
if best_ev_no_research > 0:
    print(f"Αφού EV > 0, η Charlotte πρέπει να κυκλοφορήσει το Brainet στα ${best_price_no_research}")
else:
    print(f"Αφού EV < 0, η Charlotte πρέπει να εγκαταλείψει το προϊόν")

# --- Part 3: With Market Research ---
print(f"\n--- Μέρος 3: Με Έρευνα Αγοράς ($10,000) ---")

# Market research reliability (likelihood)
likelihood = {
    'severe':   {'pred_severe': 0.80, 'pred_moderate': 0.15, 'pred_weak': 0.05},
    'moderate': {'pred_severe': 0.15, 'pred_moderate': 0.80, 'pred_weak': 0.05},
    'weak':     {'pred_severe': 0.03, 'pred_moderate': 0.07, 'pred_weak': 0.90}
}

# Calculate P(prediction)
predictions = ['pred_severe', 'pred_moderate', 'pred_weak']
p_prediction = {}
for pred in predictions:
    p = 0
    for comp in ['severe', 'moderate', 'weak']:
        p += comp_probs[comp] * likelihood[comp][pred]
    p_prediction[pred] = p
    print(f"P({pred}) = {p:.4f}")

# Posterior probabilities: P(comp | prediction)
posterior = {}
for pred in predictions:
    posterior[pred] = {}
    for comp in ['severe', 'moderate', 'weak']:
        post = comp_probs[comp] * likelihood[comp][pred] / p_prediction[pred]
        posterior[pred][comp] = post
    print(f"\nΔεδομένου {pred}:")
    for comp in ['severe', 'moderate', 'weak']:
        print(f"  P({comp} | {pred}) = {posterior[pred][comp]:.4f}")

# EV for each price given prediction
print(f"\n--- EV ανά πρόβλεψη ανά τιμή ---")
ev_given_pred = {}
for pred in predictions:
    ev_given_pred[pred] = {}
    for price in prices:
        ev = 0
        prob_t = prob_tables[price]
        for comp in ['severe', 'moderate', 'weak']:
            p_comp = posterior[pred][comp]
            for uk in ['50K', '30K', '20K']:
                p_units = prob_t[comp][uk]
                pf = payoff_table[price][uk]
                ev += p_comp * p_units * pf
        ev_given_pred[pred][price] = ev
    # Best price
    best_p = max(ev_given_pred[pred], key=ev_given_pred[pred].get)
    best_ev = ev_given_pred[pred][best_p]
    print(f"\n{pred}:")
    for price in prices:
        print(f"  EV(${price}) = ${ev_given_pred[pred][price]:,.0f}")
    print(f"  Βέλτιστη: ${best_p} (EV=${best_ev:,.0f})")
    if best_ev <= 0:
        print(f"  → Εγκατάλειψη (EV < 0)")

# EV(with research)
ev_with_research = 0
for pred in predictions:
    best_ev = max(max(ev_given_pred[pred].values()), 0)  # max(best price, abandon=0)
    ev_with_research += p_prediction[pred] * best_ev
ev_with_research -= market_research_cost

print(f"\n--- Σύγκριση ---")
print(f"EV(Χωρίς Έρευνα) = ${max(best_ev_no_research, 0):,.0f}")
print(f"EV(Με Έρευνα) = ${ev_with_research:,.0f}")
print(f"Αξία Έρευνας = ${ev_with_research - max(best_ev_no_research, 0):,.0f}")

if ev_with_research > max(best_ev_no_research, 0):
    print("ΑΠΟΦΑΣΗ: Πληρωμή $10,000 για έρευνα αγοράς")
else:
    print("ΑΠΟΦΑΣΗ: Δεν αξίζει η έρευνα αγοράς")

# Best overall policy
print(f"\n--- ΒΕΛΤΙΣΤΗ ΠΟΛΙΤΙΚΗ Cerebrosoft ---")
overall_best = max(ev_with_research, max(best_ev_no_research, 0))
if overall_best == ev_with_research and ev_with_research > max(best_ev_no_research, 0):
    print("1. Πληρωμή $10,000 για έρευνα αγοράς")
    for pred in predictions:
        best_p = max(ev_given_pred[pred], key=ev_given_pred[pred].get)
        best_ev = ev_given_pred[pred][best_p]
        if best_ev > 0:
            print(f"  Αν πρόβλεψη {pred}: Κυκλοφορία σε ${best_p}")
        else:
            print(f"  Αν πρόβλεψη {pred}: Εγκατάλειψη")
else:
    print(f"Κυκλοφορία χωρίς έρευνα στα ${best_price_no_research}")

# Cerebrosoft charts
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart: EV per price (without research)
ax1 = axes[0]
price_labels = [f'${p}' for p in prices]
ev_vals = [ev_prices[p] for p in prices]
colors_bar = ['#FF5722' if v < 0 else '#4CAF50' for v in ev_vals]
bars = ax1.bar(price_labels + ['Εγκατάλειψη'], ev_vals + [0],
               color=colors_bar + ['#9E9E9E'], edgecolor='black')
for bar, ev in zip(bars, ev_vals + [0]):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5000,
             f'${ev:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
ax1.set_ylabel('Αναμενόμενη Αποδοχή ($)', fontsize=11)
ax1.set_title('Χωρίς Έρευνα Αγοράς', fontsize=12, fontweight='bold')
ax1.axhline(y=0, color='black', linewidth=0.5)
ax1.grid(axis='y', alpha=0.3)

# Heatmap: EV with research
ax2 = axes[1]
data_matrix = []
for pred in predictions:
    row = [ev_given_pred[pred][p] for p in prices]
    data_matrix.append(row)
data_matrix = np.array(data_matrix)

im = ax2.imshow(data_matrix, cmap='RdYlGn', aspect='auto')
ax2.set_xticks(range(3))
ax2.set_xticklabels([f'${p}' for p in prices])
ax2.set_yticks(range(3))
ax2.set_yticklabels(['Πρόβλ.\nΣκληρός', 'Πρόβλ.\nΜέτριος', 'Πρόβλ.\nΑσθενής'])
for i in range(3):
    for j in range(3):
        ax2.text(j, i, f'${data_matrix[i,j]:,.0f}', ha='center', va='center', fontsize=8,
                fontweight='bold', color='black')
ax2.set_title('EV ανά Πρόβλεψη & Τιμή\n(Με Έρευνα)', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax2, label='EV ($)')

plt.tight_layout()
plt.savefig(FIGDIR + 'ex5_cerebrosoft.png', dpi=150, bbox_inches='tight')
plt.close()
print("Αποθηκεύτηκε: ex5_cerebrosoft.png")

# Cerebrosoft tree (simplified)
fig, ax = plt.subplots(figsize=(18, 12))
ax.set_xlim(-1, 16)
ax.set_ylim(-1, 14)
ax.axis('off')
ax.set_title('Δέντρο Απόφασης - Cerebrosoft/Brainet (Άσκηση 5)', fontsize=14, fontweight='bold')

# D1: Research or not
draw_decision_node(ax, 0.5, 7, 0.35)
ax.text(0.5, 7, 'D1', ha='center', va='center', fontsize=9, fontweight='bold', color='white', zorder=6)

# No research → D2
ax.annotate('', xy=(3, 3), xytext=(0.7, 6.8),
            arrowprops=dict(arrowstyle='->', lw=1.5))
ax.text(1.3, 4.5, 'Χωρίς\nΈρευνα', fontsize=8, color='#1565C0', fontweight='bold')

draw_decision_node(ax, 3.2, 3, 0.3)
ax.text(3.2, 3, 'D2', ha='center', va='center', fontsize=8, fontweight='bold', color='white', zorder=6)

# Prices without research
y_prices = [4.5, 3, 1.5, 0]
labels_d2 = ['$50', '$40', '$30', 'Εγκαταλ.']
evs_d2 = [ev_prices[50], ev_prices[40], ev_prices[30], 0]
for i, (yp, lbl, ev) in enumerate(zip(y_prices, labels_d2, evs_d2)):
    ax.annotate('', xy=(5.5, yp), xytext=(3.35, 3 + (yp-3)*0.3),
                arrowprops=dict(arrowstyle='->', lw=1))
    ax.text(4.2, yp+0.1, lbl, fontsize=7)
    if lbl != 'Εγκαταλ.':
        draw_chance_node(ax, 5.7, yp, 0.12)
        ax.text(6, yp, f'EV=${ev:,.0f}', fontsize=6, va='center')
    else:
        ax.text(5.6, yp, '$0', fontsize=7, fontweight='bold')

# With research → C1
ax.annotate('', xy=(3, 11), xytext=(0.7, 7.2),
            arrowprops=dict(arrowstyle='->', lw=1.5))
ax.text(1.3, 9.5, f'Έρευνα\n(-$10K)', fontsize=8, color='#1565C0', fontweight='bold')

draw_chance_node(ax, 3.2, 11, 0.15)
ax.text(3.2, 11, 'C1', ha='center', va='center', fontsize=7, fontweight='bold', zorder=6)

# Predictions
pred_labels = ['Πρόβλ. Σκληρός', 'Πρόβλ. Μέτριος', 'Πρόβλ. Ασθενής']
pred_ys = [13, 11, 9]
for i, (pred, pred_lbl, py) in enumerate(zip(predictions, pred_labels, pred_ys)):
    ax.annotate('', xy=(5.5, py), xytext=(3.35, 11 + (py-11)*0.3),
                arrowprops=dict(arrowstyle='->', lw=1))
    ax.text(4, py+0.2, f'{pred_lbl}\nP={p_prediction[pred]:.3f}', fontsize=6)

    draw_decision_node(ax, 5.7, py, 0.2)
    ax.text(5.7, py, 'D', fontsize=6, ha='center', va='center', fontweight='bold', color='white', zorder=6)

    # Prices
    best_p = max(ev_given_pred[pred], key=ev_given_pred[pred].get)
    best_ev = max(ev_given_pred[pred][best_p], 0)
    ax.text(6.2, py-0.3, f'Βέλτ: ${best_p if best_ev>0 else "Εγκ."}', fontsize=6, color='purple',
            bbox=dict(boxstyle='round,pad=0.1', facecolor='lightyellow'))

legend_elements = [
    mpatches.Patch(facecolor='#4CAF50', edgecolor='black', label='Κόμβος Απόφασης'),
    mpatches.Patch(facecolor='#FF9800', edgecolor='black', label='Κόμβος Τύχης'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
plt.savefig(FIGDIR + 'ex5_cerebrosoft_tree.png', dpi=150, bbox_inches='tight')
plt.close()
print("Αποθηκεύτηκε: ex5_cerebrosoft_tree.png")

print("\n" + "=" * 70)
print("ΟΛΟΙ ΟΙ ΥΠΟΛΟΓΙΣΜΟΙ ΟΛΟΚΛΗΡΩΘΗΚΑΝ ΕΠΙΤΥΧΩΣ")
print("=" * 70)
