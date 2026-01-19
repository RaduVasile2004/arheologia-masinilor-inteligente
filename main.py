import json
import pulp
import numpy as np
import time
from scipy.optimize import minimize

class KantorovichProjectSolver:
    def __init__(self):
        self.raw_inputs = {}
        self.raw_outputs = {}
        self.expected_results_map = {}
        self.results_log = []

    def load_files(self):
        """Incarca fisierele JSON utilizand codare UTF-8."""
        files_map = {
            'k_in': 'kantorovich_input_data.json',
            'k_out': 'kantorovich_output_data.json',
            'g_in': 'generated_input_data.json',
            'g_out': 'generated_output_data.json'
        }

        print("Incarcare fisiere date...")
        try:
            with open(files_map['k_in'], 'r', encoding='utf-8') as f:
                self.raw_inputs['kantorovich'] = json.load(f)
            with open(files_map['g_in'], 'r', encoding='utf-8') as f:
                self.raw_inputs['generated'] = json.load(f)

            with open(files_map['k_out'], 'r', encoding='utf-8') as f:
                self.raw_outputs['kantorovich'] = json.load(f)
            with open(files_map['g_out'], 'r', encoding='utf-8') as f:
                self.raw_outputs['generated'] = json.load(f)

            self._index_expected_results()
            print("Fisiere incarcate cu succes.")

        except FileNotFoundError as e:
            print(f"Eroare: Fisier lipsa: {e.filename}")
            exit()
        except json.JSONDecodeError as e:
            print(f"Eroare: JSON Invalid: {e}")
            exit()

    def _index_expected_results(self):
        """Indexeaza rezultatele asteptate pentru validare rapida."""
        # Rezultate generate
        gen_out = self.raw_outputs.get('generated', {})
        for key, value in gen_out.items():
            if isinstance(value, list):
                for item in value:
                    if 'id' in item:
                        val = (item.get('total_output') or item.get('total_cost') or
                               item.get('min_variance') or item.get('optimal_output'))
                        if val is not None:
                            self.expected_results_map[item['id']] = val

        # Rezultate Kantorovich (mapare manuala)
        k_out = self.raw_outputs.get('kantorovich', {})
        if 'problem_A_solutions' in k_out:
            sol = k_out['problem_A_solutions'].get('optimal_solution', {})
            val = sol.get('total_complete_sets') or sol.get('max_complete_sets')
            if val: self.expected_results_map['problem_A'] = val

    # --- SOLVER: ALOCARE (A & B) ---
    def solve_allocation(self, data, problem_type='A'):
        prob_id = data.get('id', data.get('name', 'Unknown'))
        if prob_id == "Machine Allocation Problem (Problem A)": prob_id = "problem_A"

        # 1. Matrice Productivitate
        raw_prod = data.get('productivity') or data.get('productivity_matrix')
        if not raw_prod: return None
        prod = np.array(raw_prod)
        rows, cols = prod.shape

        # 2. Capacitati Masini (Count)
        machine_capacities = [1.0] * rows
        if 'machines' in data and isinstance(data['machines'], list):
            for idx, m in enumerate(data['machines']):
                if isinstance(m, dict) and 'count' in m and idx < rows:
                    machine_capacities[idx] = float(m['count'])

        # 3. Modelare PuLP
        model = pulp.LpProblem(f"Alloc_{prob_id}", pulp.LpMaximize)
        Z = pulp.LpVariable("Z", lowBound=0)
        h = pulp.LpVariable.dicts("h", ((i, j) for i in range(rows) for j in range(cols)), lowBound=0)

        model += Z

        # Constrangere: Capacitate Masini
        for i in range(rows):
            model += pulp.lpSum([h[(i, j)] for j in range(cols)]) == machine_capacities[i]

        # Constrangere: Balanta Productie
        for j in range(cols):
            model += pulp.lpSum([prod[i][j] * h[(i, j)] for i in range(rows)]) == Z

        # Constrangere: Resurse (doar Tip B)
        if problem_type == 'B':
            cons = (data.get('resource_consumption') or
                    data.get('energy_consumption') or
                    data.get('labor_hours'))

            limit = (data.get('max_resource') or
                     data.get('max_energy') or
                     data.get('max_labor_hours'))

            if cons and limit:
                try:
                    if isinstance(cons, list) and len(cons) == rows:
                        model += pulp.lpSum([cons[i][j] * h[(i, j)] for i in range(rows) for j in range(cols)]) <= limit
                except Exception:
                    pass

        # 4. Rezolvare si masurare timp
        start = time.time()
        model.solve(pulp.PULP_CBC_CMD(msg=False))
        duration = (time.time() - start) * 1000

        return {
            "id": prob_id,
            "val": pulp.value(Z),
            "time": duration,
            "status": pulp.LpStatus[model.status]
        }

    # --- SOLVER: TRANSPORT ---
    def solve_transportation(self, data):
        prob_id = data.get('id', 'Transp')

        supply = data.get('supply') or [x['supply'] for x in data.get('supply_points', [])]
        demand = data.get('demand') or [x['demand'] for x in data.get('demand_points', [])]
        costs = data.get('transport_costs') or data.get('cost_matrix') or data.get('costs')

        if not (supply and demand and costs): return None

        model = pulp.LpProblem(f"Trans_{prob_id}", pulp.LpMinimize)
        x = pulp.LpVariable.dicts("Route", (range(len(supply)), range(len(demand))), lowBound=0)

        model += pulp.lpSum([x[i][j] * costs[i][j] for i in range(len(supply)) for j in range(len(demand))])

        for i in range(len(supply)):
            model += pulp.lpSum([x[i][j] for j in range(len(demand))]) <= supply[i]
        for j in range(len(demand)):
            model += pulp.lpSum([x[i][j] for i in range(len(supply))]) >= demand[j]

        start = time.time()
        model.solve(pulp.PULP_CBC_CMD(msg=False))

        return {
            "id": prob_id,
            "val": pulp.value(model.objective),
            "time": (time.time() - start) * 1000,
            "status": pulp.LpStatus[model.status]
        }

    # --- SOLVER: PORTOFOLIU (SciPy) ---
    def solve_portfolio(self, data):
        prob_id = data.get('id', 'Portfol')
        returns = np.array(data['expected_returns'])
        cov_matrix = np.array(data['covariance_matrix'])
        target = data.get('target_return', 0.10)
        n = len(returns)

        def objective(w): return w @ cov_matrix @ w

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: w @ returns - target}
        ]
        bounds = tuple((0, 1) for _ in range(n))
        x0 = np.array([1 / n] * n)

        start = time.time()
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        return {
            "id": prob_id,
            "val": res.fun,
            "time": (time.time() - start) * 1000,
            "status": "Optimal" if res.success else "Failed"
        }

    # --- PROCESARE SI VALIDARE ---
    def process_all(self):
        print("\nRulare benchmark-uri si validare...")
        gen_in = self.raw_inputs['generated']
        k_in = self.raw_inputs['kantorovich']

        # Procesare seturi de date generate
        for p in gen_in.get('problem_A_variants', []):
            self._log(self.solve_allocation(p, 'A'), "Problem A (Gen)")

        for p in gen_in.get('problem_B_variants', []):
            self._log(self.solve_allocation(p, 'B'), "Problem B (Gen)")

        for p in gen_in.get('transportation_problems', []):
            self._log(self.solve_transportation(p), "Transport")

        for p in gen_in.get('portfolio_optimization', []):
            self._log(self.solve_portfolio(p), "Portofoliu")

        # Procesare seturi originale
        if 'problem_A' in k_in:
            p = k_in['problem_A']
            p['id'] = 'problem_A'
            self._log(self.solve_allocation(p, 'A'), "Problem A (Orig)")

    def _log(self, res, cat):
        if not res: return
        expected = self.expected_results_map.get(res['id'])

        status = "N/A"
        if expected is not None:
            # Toleranta adaptiva
            tol = 0.5 if res['val'] > 10 else 0.05
            if abs(res['val'] - expected) <= tol:
                status = "PASS"
            else:
                status = f"FAIL (Diff: {abs(res['val'] - expected):.4f})"

        self.results_log.append({
            "Category": cat, "ID": res['id'], "Result": res['val'],
            "Expected": expected if expected else "-", "Time": res['time'], "Status": status
        })

    def print_report(self):
        print("\n" + "=" * 90)
        print(f"{'CATEGORIE':<20} | {'ID':<10} | {'CALCULAT':<12} | {'ASTEPTAT':<12} | {'TIMP(ms)':<8} | {'VALIDARE'}")
        print("=" * 90)
        for r in self.results_log:
            calc = f"{r['Result']:.4f}"
            exp = f"{r['Expected']:.4f}" if isinstance(r['Expected'], (int, float)) else str(r['Expected'])
            print(f"{r['Category']:<20} | {r['ID']:<10} | {calc:<12} | {exp:<12} | {r['Time']:<8.2f} | {r['Status']}")
        print("=" * 90)

if __name__ == "__main__":
    solver = KantorovichProjectSolver()
    solver.load_files()
    solver.process_all()
    solver.print_report()