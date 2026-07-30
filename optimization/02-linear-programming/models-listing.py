ΑΣΚΗΣΗ 1

import pulp


 prob = pulp . LpProblem ( " Aircraft_Fuel_Optimization " , pulp . LpMinimize )
 airports = [ ’ ATH ’ , ’ LON ’ , ’ PAR ’ , ’ ROM ’]
 prices = { ’ ATH ’: 1.15 , ’ LON ’: 1.60 , ’ PAR ’: 1.70 , ’ ROM ’: 1.65}
 min_fuel = { ’ ATH ’: 5000 , ’ LON ’: 5000 , ’ PAR ’: 5000 , ’ ROM ’: 5000}
 max_fuel = { ’ ATH ’: 30000 , ’ LON ’: 30000 , ’ PAR ’: 30000 , ’ ROM ’: 30000}
 buy = pulp . LpVariable . dicts ( " Buy " , airports , lowBound =0)
 depart = pulp . LpVariable . dicts ( " Depart " , airports , lowBound =0)
 arrive = pulp . LpVariable . dicts ( " Arrive " , airports , lowBound =0)


 prob += pulp . lpSum ([ prices [ index ] * buy [ index ] for index in airports ])

 for index in airports :
 prob += depart [ index ] ==arrive [ index ] + buy [ index ]


 for index in airports :
 prob += depart [ index ] >= min_fuel [ index ]
 prob += depart [ index ] <= max_fuel [ index ]


 prob . solve ()

 print ( " Status : " , pulp . LpStatus [ prob . status ])
 for index in airports :
 print ( f " { index }: Buy { buy [ index ]. varValue } , Depart { depart [ index ]. varValue } " )
 print ( f " Total Cost : { pulp . value ( prob . objective ) } " )


ΑΣΚΗΣΗ 2
import pulp

vineyards = ['A', 'B', 'C']
factories = ['Crete', 'Ilia', 'Attica', 'Macedonia']
products = ['Bottled', 'Frozen', 'Jelly']

supply = {'A': 1400, 'B': 1100, 'C': 1700}
capacity = {'Crete': 1200, 'Ilia': 1100, 'Attica': 1400, 'Macedonia': 1400}
demand = {'Bottled': 1200, 'Frozen': 900, 'Jelly': 700}

trans_cost = {
    'A': {'Crete': 850, 'Ilia': 720, 'Attica': 910, 'Macedonia': 750},
    'B': {'Crete': 970, 'Ilia': 790, 'Attica': 1050, 'Macedonia': 880},
    'C': {'Crete': 900, 'Ilia': 830, 'Attica': 780, 'Macedonia': 820}
}

proc_cost = {
    'Bottled': {'Crete': 2100, 'Ilia': 2350, 'Attica': 2200, 'Macedonia': 1900},
    'Frozen': {'Crete': 4100, 'Ilia': 4300, 'Attica': 3950, 'Macedonia': 3900},
    'Jelly': {'Crete': 2600, 'Ilia': 2300, 'Attica': 2500, 'Macedonia': 2800}
}

rates = {'Bottled': 1.0, 'Frozen': 2.0, 'Jelly': 1.5}

prob = pulp.LpProblem("Wallasidis_Juice_Optimization", pulp.LpMinimize)

X = pulp.LpVariable.dicts("Transport", (vineyards, factories), lowBound=0)
Y = pulp.LpVariable.dicts("Production", (factories, products), lowBound=0)

prob += pulp.lpSum([trans_cost[idx][step] * X[idx][step] for idx in vineyards for step in factories]) + \
        pulp.lpSum([proc_cost[tertiary][step] * Y[step][tertiary] for step in factories for tertiary in products])

for idx in vineyards:
    prob += pulp.lpSum([X[idx][step] for step in factories]) <= supply[idx], f"Supply_{idx}"

for step in factories:
    prob += pulp.lpSum([X[idx][step] for idx in vineyards]) <= capacity[step], f"Capacity_{step}"
    prob += pulp.lpSum([X[idx][step] for idx in vineyards]) == pulp.lpSum([rates[tertiary] * Y[step][tertiary] for tertiary in products]), f"Balance_{step}"

for tertiary in products:
    prob += pulp.lpSum([Y[step][tertiary] for step in factories]) == demand[tertiary], f"Demand_{tertiary}"

prob.solve()

print(f"Total Cost: {pulp.value(prob.objective)}")
for idx in vineyards:
    for step in factories:
        if X[idx][step].varValue > 0: print(f"Trans {idx}->{step}: {X[idx][step].varValue}")
for step in factories:
    for tertiary in products:
        if Y[step][tertiary].varValue > 0: print(f"Prod {step}->{tertiary}: {Y[step][tertiary].varValue}")
\end{lstlisting}

\section{Υλοποίηση Κώδικα Άσκηση 3 C}
\begin{lstlisting}[language=Python, caption=Σενάριο Βελτιστοποίησης με χρήση PuLP]
import pulp

prob = pulp.LpProblem("Eva_Diet_Optimization", pulp.LpMaximize)

x1 = pulp.LpVariable("Meal_A", lowBound=0)
x2 = pulp.LpVariable("Meal_B", lowBound=0)

# Objective Function (Tastiness per portion)
# Meal A: 37g * 85 = 3145, Meal B: 65g * 95 = 6175
prob += 3145 * x1 + 6175 * x2

# Constraints
prob += 120 * x1 + 160 * x2 <= 450, "Calories"
prob += 5 * x1 + 10 * x2 <= 25, "Fat"
prob += 37 * x1 + 65 * x2 >= 120, "Min_Weight"

prob.solve()

print(f"Optimal Solution: Meal A = {x1.varValue}, Meal B = {x2.varValue}")
print(f"Max Tastiness: {pulp.value(prob.objective)}")

print("\n--- Shadow Prices ---")
for name, c in prob.constraints.items():
    print(f"{name}: {c.pi}")
\end{lstlisting}

ΑΣΚΗΣΗ 3
import numpy as np
import matplotlib.pyplot as plt

item  =  np.linspace(0, 5, 400)

y_calories  =  (450- 120 * item) /160

y_fat  =  (25- 5 * item) / 10

y_weight  =  (120 - 37 * item) / 65

plt.figure(figsize = (10, 8))

plt.plot(item, y_calories, label = r"Calories ($120x_1 +160x_2 \leq 450$)", color = "blue")
plt.plot(item, y_fat, label = r"Fat ($5x_1 +10x_2 \leq 25$)", color='red')
plt.plot(item, y_weight, label=r"Weight ($37x_1 + 65x_2 \geq 120$)", color="green")

y_upper = np.minimum(y_calories, y_fat)

y_lower = np.maximum(y_weight, 0)

plt.fill_between(item, y_lower, y_upper, where=(y_upper  >=  y_lower),
                 color='gray', alpha=0.3, label="Feasible Region")

opt_x = 1.25
opt_y = 1.875
plt.plot(opt_x, opt_y, "ko", markersize=8, label='Optimal Solution (1.25, 1.875)')
plt.annotate(f"Optimal\total({opt_x}, {opt_y})", xy=(opt_x, opt_y),
             xytext=(opt_x + 0.5, opt_y + 0.5),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.xlim(0, 4)
plt.ylim(0, 3.5)
plt.xlabel("Meal A Portions ($x_1$)")
plt.ylabel("Meal B Portions ($x_2$)")
plt.title('Graphical Solution - Diet Optimization')
plt.legend()
plt.grid(True)

plt.show()



ΑΣΚΗΣΗ 4
import pulp

def solve_scenario(name, weights, doctor_cost_per_unit, hard_budget=False, preemptive=False):
    prob = pulp.LpProblem(name, pulp.LpMinimize)

    x1 = pulp.LpVariable("Basic", lowBound=0, cat='Integer')
    x2 = pulp.LpVariable("Advanced", lowBound=0, cat='Integer')
    x3 = pulp.LpVariable("Supreme", lowBound=0, cat='Integer')

    d1_minus = pulp.LpVariable("d1_under", lowBound=0)
    d1_plus = pulp.LpVariable("d1_over", lowBound=0)
    d2_minus = pulp.LpVariable("d2_under", lowBound=0)
    d2_plus = pulp.LpVariable("d2_over", lowBound=0)
    d3_minus = pulp.LpVariable("d3_under", lowBound=0)
    d3_plus = pulp.LpVariable("d3_over", lowBound=0)

    cost_sup = 720 + doctor_cost_per_unit

    prob += x1 + x2 + x3 <= 40000
    prob += 120*x1 + 180*x2 + 220*x3 <= 6000000

    prob += 30*x1 + 35*x2 + 54*x3 + d1_minus - d1_plus   ==2200000
    prob += x3 + d2_minus - d2_plus   ==3000
    prob += 300*x1 + 350*x2 + cost_sup*x3 + d3_minus - d3_plus   ==20000000

    if hard_budget:
        prob += d3_plus   ==0

    if preemptive:

        prob += d3_plus   ==0
        prob += weights['w1'] * d1_minus + weights['w2'] * d2_minus
    else:
        prob += weights['w1'] * d1_minus + weights['w2'] * d2_minus + weights['w3'] * d3_plus

    prob.solve()
    return x1.varValue, x2.varValue, x3.varValue, pulp.value(prob.objective)

w_a = {'w1': 7/100000, 'w2': 1/1000, 'w3': 1/1000000}

w_b = {'w1': 10/55000, 'w2': 1/1000, 'w3': 1/1000000}

sol_a = solve_scenario("A", w_a, 330)
sol_b = solve_scenario("B", w_b, 330)
sol_c = solve_scenario("C", w_b, 440)
sol_d = solve_scenario("D", w_a, 330, hard_budget=True)
sol_e = solve_scenario("E", w_a, 330, preemptive=True)

print(f"Scenario A: {sol_a}")
print(f"Scenario B: {sol_b}")
print(f"Scenario C: {sol_c}")
print(f"Scenario D: {sol_d}")
print(f"Scenario E: {sol_e}")





ΑΣΚΗΣΗ 5
import pulp

def solve_school_busing():
    # --- Δεδομένα Προβλήματος ---
    districts = ['North', 'South', 'East', 'West']
    schools = ['North', 'South', 'East', 'West']
    races = ['White', 'Black']

    # Προσφορά Μαθητών ανά Περιοχή και Φυλή
    supply = {
        'North': {'White': 1000, 'Black': 300},
        'South': {'White': 450, 'Black': 800},
        'East': {'White': 1050, 'Black': 400},
        'West': {'White': 500, 'Black': 500}
    }

    # Χωρητικότητες Σχολείων
    capacities = {'North': 1200, 'South': 1000, 'East': 1000, 'West': 1200}
    
    # Υπολογισμός Στόχων Πληθυσμού (Αναλογικός Υπερπληθυσμός)
    total_students = 5000
    total_capacity = 4400
    prop_factor = total_students / total_capacity # ~1.136
    targets_pop = {s: capacities[s] * prop_factor for s in schools}

    # Αποστάσεις (Μίλια)
    distances = {
        'North': {'North': 0, 'South': 30, 'East': 12, 'West': 20},
        'South': {'North': 30, 'South': 0, 'East': 18, 'West': 26},
        'East': {'North': 12, 'South': 18, 'East': 0, 'West': 24},
        'West': {'North': 20, 'South': 26, 'East': 24, 'West': 0}
    }

    target_miles = 30000

    # --- Μοντελοποίηση ---
    prob = pulp.LpProblem("School_Busing", pulp.LpMinimize)
    
    # Μεταβλητές Απόφασης: x[district][school][race]
    # Χρησιμοποιούμε συνεχείς μεταβλητές για ταχύτητα και ευελιξία σε μεγάλους αριθμούς
    x = pulp.LpVariable.dicts("Assign", (districts, schools, races), lowBound=0)
    
    # Μεταβλητές Απόκλισης (Deviational Variables)
    d1_m = pulp.LpVariable.dicts("d1_minus", schools, lowBound=0)
    d1_p = pulp.LpVariable.dicts("d1_plus", schools, lowBound=0)
    d2_m = pulp.LpVariable("d2_minus", lowBound=0)
    d2_p = pulp.LpVariable("d2_plus", lowBound=0)
    d3_m = pulp.LpVariable.dicts("d3_minus", schools, lowBound=0)
    d3_p = pulp.LpVariable.dicts("d3_plus", schools, lowBound=0)

    # Σκληροί Περιορισμοί: Όλοι οι μαθητές πρέπει να πάνε σχολείο
    for d in districts:
        for r in races:
            prob += pulp.lpSum([x[d][s][r] for s in schools]) == supply[d][r]

    # Στόχος 1: Φυλετική Ισορροπία (60% White - 40% Black)
    # Μαθηματικά: 2*White - 3*Black = 0
    for s in schools:
        white_in_s = pulp.lpSum([x[d][s]['White'] for d in districts])
        black_in_s = pulp.lpSum([x[d][s]['Black'] for d in districts])
        prob += 2 * white_in_s - 3 * black_in_s + d1_m[s] - d1_p[s] == 0

    # Στόχος 2: Μίλια <= 30,000
    total_miles = pulp.lpSum([x[d][s][r] * distances[d][s] 
                              for d in districts for s in schools for r in races])
    prob += total_miles + d2_m - d2_p == target_miles

    # Στόχος 3: Αναλογικός Υπερπληθυσμός
    for s in schools:
        total_in_s = pulp.lpSum([x[d][s][r] for d in districts for r in races])
        prob += total_in_s + d3_m[s] - d3_p[s] == targets_pop[s]

    # --- Διαδοχική Επίλυση (Preemptive Goal Programming) ---
    
    # Priority 1: Ελαχιστοποίηση Φυλετικής Ανισορροπίας
    prob.setObjective(pulp.lpSum([d1_m[s] + d1_p[s] for s in schools]))
    prob.solve()
    # Κλείδωμα τιμής P1
    val_p1 = pulp.value(prob.objective)
    prob += pulp.lpSum([d1_m[s] + d1_p[s] for s in schools]) == val_p1

    # Priority 2: Ελαχιστοποίηση Υπέρβασης Μιλίων
    prob.setObjective(d2_p)
    prob.solve()
    # Κλείδωμα τιμής P2
    val_p2 = pulp.value(prob.objective)
    prob += d2_p == val_p2

    # Priority 3: Ελαχιστοποίηση Απόκλισης από Στόχο Πληθυσμού
    prob.setObjective(pulp.lpSum([d3_m[s] + d3_p[s] for s in schools]))
    prob.solve()
    
    return x, pulp.value(total_miles)

# Εκτέλεση
assignments, final_miles = solve_school_busing()
print(f"Total Miles: {final_miles}")



ΑΣΚΗΣΗ 6

import pulp

segments =  [1, 2, 3, 4, 5, 6]
costs =  {1: 20, 2: 18, 3: 22, 4: 24, 5: 17, 6: 19}
accidents =  {1: 0.27, 2: 0.21, 3: 0.28, 4: 0.19, 5: 0.23, 6: 0.33}
physical =  {1: 18, 2: 26, 3: 10, 4: 34, 5: 25, 6: 17}
sight =  {1: 1700, 2: 900, 3: 650, 4: 230, 5: 1600, 6: 520}
response =  {1: 0.32, 2: 0.65, 3: 0.43, 4: 0.87, 5: 0.55, 6: 0.49}

targets =  {"cost": 450, 'acc': 5, "phys": 350, "sight": 30000, 'resp': 13}

prob =  pulp.LpProblem('Patrol_Allocation', pulp.LpMinimize)
item =  pulp.LpVariable.dicts("Cars", segments, lowBound= 2, upBound= 5, cat= "Integer")

d1_m =  pulp.LpVariable(
    "d1_m", lowBound= 0); d1_p =  pulp.LpVariable("d1_p", lowBound= 0
)
d2_m =  pulp.LpVariable('d2_m', lowBound= 0); d2_p = pulp.LpVariable("d2_p", lowBound=0)
d3_m = pulp.LpVariable('d3_m', lowBound=0); d3_p = pulp.LpVariable("d3_p", lowBound=0)
d4_m = pulp.LpVariable(
    'd4_m', lowBound=0); d4_p = pulp.LpVariable('d4_p', lowBound=0
)
d5_m = pulp.LpVariable(
    "d5_m", lowBound=0); d5_p = pulp.LpVariable('d5_p', lowBound=0
)

prob + = pulp.lpSum([item[iter] for iter in segments])  == 23
prob + = pulp.lpSum([costs[iter] * item[iter] for iter in segments]) + d1_m- d1_p  == targets['cost']
prob + = pulp.lpSum([accidents[iter]*item[iter] for iter in segments]) + d2_m- d2_p  == targets['acc']
prob += pulp.lpSum([physical[iter]*item[iter] for iter in segments]) + d3_m - d3_p  == targets["phys"]
prob += pulp.lpSum([sight[iter]*item[iter] for iter in segments]) + d4_m - d4_p == targets['sight']
prob += pulp.lpSum([response[iter]*item[iter] for iter in segments]) + d5_m - d5_p == targets['resp']

prob.setObjective(d1_p)
prob.solve()
prob += d1_p == d1_p.varValue

prob.setObjective(d2_m)
prob.solve()
prob += d2_m == d2_m.varValue

prob.setObjective(d3_m)
prob.solve()
prob += d3_m == d3_m.varValue

prob.setObjective(d4_m)
prob.solve()
prob += d4_m == d4_m.varValue

prob.setObjective(d5_m)
prob.solve()

print('Assignments:', {iter: item[iter].varValue for iter in segments})
print(f'Final Cost: {combined(costs[iter]*item[iter].varValue for iter in segments)}')
print(f'Final Response Time: {28 - combined(response[iter]*item[iter].varValue for iter in segments)}')







ΑΣΚΗΣΗ 7

import pulp

def solve_exercise_7():

    c1_base, m1_base, c1_step, m1_step = 90000, 15000, 5000, 1500

    c2_base, m2_base, c2_step, m2_step = 60000, 9000, 30000, 1200

    target_fa_total = 10
    target_budget = 250000
    target_maint = 30000

    prob = pulp.LpProblem("Airport_Security", pulp.LpMinimize)

    x1 = pulp.LpVariable("Portal_Upgrade", lowBound=0, upBound=8, cat='Integer')
    x2 = pulp.LpVariable("Screening_Upgrade", lowBound=0, upBound=3, cat='Integer')

    d2_m = pulp.LpVariable("d2_minus", lowBound=0); d2_p = pulp.LpVariable("d2_plus", lowBound=0)
    d3_m = pulp.LpVariable("d3_minus", lowBound=0); d3_p = pulp.LpVariable("d3_plus", lowBound=0)
    d4_m = pulp.LpVariable("d4_minus", lowBound=0); d4_p = pulp.LpVariable("d4_plus", lowBound=0)

    current_fa = (10 - 1*x1) + (6 - 1*x2)
    current_cost = (c1_base + c1_step*x1) + (c2_base + c2_step*x2)
    current_maint = (m1_base + m1_step*x1) + (m2_base + m2_step*x2)

    prob += current_fa + d2_m - d2_p  == target_fa_total

    prob += current_cost + d3_m - d3_p  == target_budget

    prob += current_maint + d4_m - d4_p  == target_maint

    prob.setObjective(d2_p)
    prob.solve()
    prob += d2_p  == d2_p.varValue

    prob.setObjective(d3_p)
    prob.solve()
    prob += d3_p  == d3_p.varValue

    prob.setObjective(d4_p)
    prob.solve()

    return x1.varValue, x2.varValue, pulp.value(prob.objective)

x1_sol, x2_sol, maint_overage = solve_exercise_7()
print(f"Optimal Upgrades: Portal={x1_sol}, Screening={x2_sol}")
print(f"Maintenance Overage: {maint_overage}")
