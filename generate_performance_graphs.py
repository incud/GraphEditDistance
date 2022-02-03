import matplotlib.pyplot as plt

# ========================== CLASSICAL HEURISTICS COMPARISON ==========================

x = range(3, 9+1)
R1 = [0, .08, .16, .17, .19, .17, .17]
B7 = [.32, .19, .34, .49, .42, .46, .54]
R3 = [0, .06, 0, .15, .29, .21, .25]
I1 = [.22, .10, .28, .19, .27, .35, .18]

plt.plot(x, R1, label="R1")
plt.plot(x, B7, label="B7")
plt.plot(x, R3, label="R3")
plt.plot(x, I1, label="I1")
# plt.title("Relative difference comparison of classical heuristics (lower is better)")
plt.xlabel("Number of vertices", fontsize=20)
plt.ylabel("Mean relative difference", fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(bbox_to_anchor=(0.99, -1.3), fontsize=14)
plt.xlim((3, 10))
plt.savefig("plot_performances/classical_heuristics.png", dpi=300, bbox_inches='tight')

# ========================== CLASSICAL HEURISTICS COMPARISON ==========================

SA = [0.00, 0.00, 0.06, 0.00, 0.05, 0.00, 0.10]
D1 = [0.00, 0.00, 0.06, 0.35, 0.50, 0.67, 0.87]
D2 = [0.00, 0.00, 0.06, 0.24, 0.43, 0.41, 0.87]
D3 = [0.00, 0.00, 0.09, 0.23, 0.47, 0.50, 0.87]
D4 = [0.00, 0.00, 0.06, 0.29, 0.61, 0.69, 0.81]
D5 = [0.00, 0.00, 0.06, 0.11, 0.51, 0.60, 0.70]
D6 = [0.00, 0.00, 0.06, 0.25, 0.44, 0.70, 0.75]
D7 = [0.00, 0.00, 0.06, 0.00, 0.05, 0.06, 0.13]
V1 = [0.64, 0.86, 0.89, 0.90, 0.91, 0.85, 0.87]
V3 = [0.54, 0.76, 0.84, 0.90, 0.91, 0.85, 0.87]
Q1 = [0.44, 0.88, 0.89, 0.90, 0.91, 0.85, 0.87]
Q3 = [0.65, 0.91, 0.89, 0.90, 0.91, 0.85, 0.87]

plt.figure()
plt.plot(x, SA, label="SA")  # Simulated annealing
plt.plot(x, D1, label="D1")  # D-Wave 2006 Q (default configuration)
plt.plot(x, D2, label="D2")  # D-Wave 2006 Q (long annealing time)
plt.plot(x, D3, label="D3")  # D-Wave 2006 Q (pause in annealing)
plt.plot(x, D4, label="D4")  # D-Wave Advantage (default configuration)
plt.plot(x, D5, label="D5")  # D-Wave Advantage (long annealing time)
plt.plot(x, D6, label="D6")  # D-Wave Advantage (pause in annealing)
plt.plot(x, D7, label="D7")  # D-Wave Leap
# plt.title("Relative difference comparison of SA and quantum annealers (formulation having b=0.1, lower is better)")
plt.xlabel("N", fontsize=20)
plt.ylabel("Mean relative difference", fontsize=20)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim((3, 10))
plt.savefig("plot_performances/quantum_annealers.png", dpi=300, bbox_inches='tight')


plt.figure()
plt.plot(x, SA, label="SA")  # Simulated annealing
plt.plot(x, D1, label="D1")  # D-Wave 2006 Q (default configuration)
plt.plot(x, D2, label="D2")  # D-Wave 2006 Q (long annealing time)
plt.plot(x, D3, label="D3")  # D-Wave 2006 Q (pause in annealing)
# plt.title("Relative difference comparison of SA and D-Wave 2006 (formulation having b=0.1, lower is better)")
plt.xlabel("Number of vertices", fontsize=20)
plt.ylabel("Mean relative difference", fontsize=20)
plt.legend(bbox_to_anchor=(0.99, -1.3), fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim((3, 10))
plt.savefig("plot_performances/quantum_annealers_2000.png", dpi=300, bbox_inches='tight')

plt.figure()
plt.plot(x, SA, label="SA")  # Simulated annealing
plt.plot(x, D4, label="D4")  # D-Wave Advantage (default configuration)
plt.plot(x, D5, label="D5")  # D-Wave Advantage (long annealing time)
plt.plot(x, D6, label="D6")  # D-Wave Advantage (pause in annealing)
# plt.title("Relative difference comparison of SA and D-Wave Advantage (formulation having b=0.1, lower is better)")
plt.xlabel("Number of vertices", fontsize=20)
plt.ylabel("Mean relative difference", fontsize=20)
plt.legend(bbox_to_anchor=(0.99, -1.3), fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim((3, 10))
plt.savefig("plot_performances/quantum_annealers_advantage.png", dpi=300, bbox_inches='tight')

plt.figure()
plt.plot(x, SA, label="SA")  # Simulated annealing
plt.plot(x, D2, label="D2")  # D-Wave 2006 Q (long annealing time)
plt.plot(x, D5, label="D5")  # D-Wave Advantage (long annealing time)
plt.plot(x, D7, label="D7")  # D-Wave Leap
# plt.title("Relative difference comparison of SA and quantum annealers (formulation having b=0.1, lower is better)")
plt.xlabel("Number of vertices", fontsize=20)
plt.ylabel("Mean relative difference", fontsize=20)
plt.legend(bbox_to_anchor=(0.99, -1.3), fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim((3, 10))
plt.savefig("plot_performances/quantum_annealers_best.png", dpi=300, bbox_inches='tight')





plt.figure()
plt.plot(x, SA, label="SA")  # Simulated annealing
plt.plot(x, V1, label="V1")  # VQE (p=1)
plt.plot(x, V3, label="V3")  # VQE (p=3)
plt.plot(x, Q1, label="Q1")  # QAOA (p=1)
plt.plot(x, Q3, label="Q3")  # QAOA (p=3)
# plt.title("Relative difference comparison of SA and variational algorithms (formulation having b=0.1, lower is better)")
plt.xlabel("Number of vertices", fontsize=20)
plt.ylabel("Mean relative difference", fontsize=20)
plt.legend(bbox_to_anchor=(0.99, -1.3))
plt.xlim((3, 10))
plt.savefig("plot_performances/quantum_variational.png", dpi=300, bbox_inches='tight')


