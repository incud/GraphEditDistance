import matplotlib.pyplot as plt

# ========================== CLASSICAL HEURISTICS COMPARISON ==========================

x = range(3, 9+1)
R1 = [1.00, 0.88, 0.75, 0.56, 0.56, 0.56, 0.50]
B7 = [0.67, 0.75, 0.56, 0.25, 0.19, 0.06, 0.00]
R3 = [1.00, 1.00, 0.81, 0.69, 0.56, 0.75, 0.50]
I1 = [0.78, 0.88, 0.62, 0.81, 0.62, 0.56, 0.75]

plt.plot(x, R1, label="R1")  # RING
plt.plot(x, B7, label="B7")  # BIPARTITE ML
plt.plot(x, R3, label="R3")  # REFINE
plt.plot(x, I1, label="I1")  # IPFP
# plt.title("Relative difference comparison of classical heuristics (lower is better)")
plt.xlabel("Number of vertices", fontsize=20)
plt.ylabel("Success solution probability", fontsize=20)
plt.legend(bbox_to_anchor=(0.99, -1.3), fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim((3, 10))
plt.savefig("plot_performances/SP_classical_heuristics.png", dpi=300, bbox_inches='tight')

# ========================== CLASSICAL HEURISTICS COMPARISON ==========================

SA = [1.00, 1.00, 0.88, 1.00, 0.88, 1.00, 0.75]
D1 = [1.00, 1.00, 0.88, 0.38, 0.12, 0.00, 0.00]
D2 = [1.00, 1.00, 0.88, 0.69, 0.25, 0.25, 0.00]
D3 = [1.00, 1.00, 0.81, 0.56, 0.19, 0.19, 0.00]
D4 = [1.00, 1.00, 0.88, 0.56, 0.06, 0.00, 0.00]
D5 = [1.00, 1.00, 0.88, 0.75, 0.19, 0.06, 0.00]
D6 = [1.00, 1.00, 0.88, 0.56, 0.12, 0.00, 0.00]
D7 = [1.00, 1.00, 0.88, 1.00, 0.88, 0.88, 0.69]
V1 = [0.33, 0.06, 0.00, 0.00, 0.00, 0.00, 0.00]
V3 = [0.44, 0.12, 0.06, 0.00, 0.00, 0.00, 0.00]
Q1 = [0.56, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
Q3 = [0.33, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]

plt.figure()
plt.plot(x, SA, label="SA")  # Simulated annealing
plt.plot(x, D4, label="D4")  # D-Wave Advantage (default configuration)
plt.plot(x, D5, label="D5")  # D-Wave Advantage (long annealing time)
plt.plot(x, D6, label="D6")  # D-Wave Advantage (pause in annealing)
# plt.title("Relative difference comparison of SA and D-Wave Advantage (formulation having b=0.1, lower is better)")
plt.xlabel("Number of vertices", fontsize=20)
plt.ylabel("Success solution probability", fontsize=20)
plt.legend(bbox_to_anchor=(0.99, -1.3), fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim((3, 10))
plt.savefig("plot_performances/SP_quantum_annealers_advantage.png", dpi=300, bbox_inches='tight')

plt.figure()
plt.plot(x, SA, label="SA")  # Simulated annealing
plt.plot(x, D2, label="D2")  # D-Wave 2006 Q (long annealing time)
plt.plot(x, D5, label="D5")  # D-Wave Advantage (long annealing time)
plt.plot(x, D7, label="D7")  # D-Wave Leap
# plt.title("Relative difference comparison of SA and quantum annealers (formulation having b=0.1, lower is better)")
plt.xlabel("Number of vertices", fontsize=20)
plt.ylabel("Success solution probability", fontsize=20)
plt.legend(bbox_to_anchor=(0.99, -1.3), fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim((3, 10))
plt.savefig("plot_performances/SP_quantum_annealers_best.png", dpi=300, bbox_inches='tight')





plt.figure()
plt.plot(x, SA, label="SA")  # Simulated annealing
plt.plot(x, V1, label="V1")  # VQE (p=1)
plt.plot(x, V3, label="V3")  # VQE (p=3)
plt.plot(x, Q1, label="Q1")  # QAOA (p=1)
plt.plot(x, Q3, label="Q3")  # QAOA (p=3)
# plt.title("Relative difference comparison of SA and variational algorithms (formulation having b=0.1, lower is better)")
plt.xlabel("Number of vertices", fontsize=20)
plt.ylabel("Success solution probability", fontsize=20)
plt.legend(bbox_to_anchor=(0.99, -1.3), fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim((3, 10))
plt.savefig("plot_performances/SP_quantum_variational.png", dpi=300, bbox_inches='tight')


