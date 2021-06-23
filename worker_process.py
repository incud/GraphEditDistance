from GraphEditDistanceCalculator import GraphEditDistanceCalculator
from datetime import datetime
import networkx as nx


def worker(vertices, g1, g2, n1, n2, a, b, queue):
    rows = []
    calculator = GraphEditDistanceCalculator(g1=g1, g2=g2, a=a, b=b)

    start = datetime.now()
    sample, energy, info = calculator.run_simulated()
    end = datetime.now()

    row = {'vertices': vertices, 'g1_name': n1, 'g2_name': n2, 'start': start, 'end': end, 'a': a, 'b': b}
    row = {**row, **info}
    rows.append(row)
    print(".", end="", flush=True)

    for is_adv, is_leap in [(False, False), (True, False), (False, True)]:
        calculator = GraphEditDistanceCalculator(g1=g1, g2=g2, a=a, b=b)

        start = datetime.now()
        sample, energy, info = calculator.run_dwave(is_advantage=is_adv, is_leap=is_leap)
        end = datetime.now()

        row = {'vertices': vertices, 'g1_name': n1, 'g2_name': n2, 'start': start, 'end': end, 'a': a, 'b': b}
        row = {**row, **info}
        rows.append(row)
        print(".", end="", flush=True)

    print("END", flush=True)
    queue.put(rows)
