##!/usr/bin/env python2
## -*- coding: utf-8 -*-
#"""
#Created on Tue Nov  3 18:49:49 2020
#
#@author: richi
#"""
#
##!/usr/bin/env python2
## -*- coding: utf-8 -*-
#"""
#Created on Wed Oct 28 10:29:43 2020
#
#@author: richi
#"""
#
#import numpy as np
#import matplotlib.pyplot as plt
#import sys
#import dimod
#sys.path.append('../')
#import qcfco.polynomial
#import qcfco.instance
#import qcfco.variable
#import instance
#import runs
#import variable
#import networkx as nx
#from networkx.algorithms.approximation import independent_set
#import itertools
#from networkx.generators.random_graphs import erdos_renyi_graph
#import time
#import dimod
#from dimod.higherorder import *
#import networkx as nx
#from dwave.system.composites import EmbeddingComposite
#from dimod.decorators import lockable_method
#from dwave.system.composites import FixedEmbeddingComposite
#from dwave.cloud import Client
#from dwave.system.samplers import DWaveSampler
#from minorminer import find_embedding
#import neal
#import time
#from tabu import TabuSampler
#import hybrid
#from hybrid.decomposers import EnergyImpactDecomposer, RandomSubproblemDecomposer
#from hybrid.core import State
#from hybrid.utils import min_sample, flip_energy_gains
#from dwave_qbsolv import QBSolv
#from dwave_sapi2.remote import RemoteConnection
#from dwave_sapi2.util import get_hardware_adjacency
#from dwave_sapi2.embedding import embed_problem
#from dwave_sapi2.embedding import find_embedding
#from dwave_sapi2.core import solve_ising
#from dwave_sapi2.embedding import unembed_answer
#from dwave_sapi2.util import qubo_to_ising
#
#def print_graph(G1, G2):
#    plt.subplot(121)
#    nx.draw(G1, with_labels=False, font_weight='bold')
#    plt.subplot(122)
#    nx.draw(G2, with_labels=False, font_weight='bold')
#    plt.show()
#
#def classic_edit_distance(G1, G2):
#    #ged = nx.graph_edit_distance(G1, G2)
#    GED_start_time = time.time()
#    for v in nx.optimize_graph_edit_distance(G1, G2):
#        print "GED= ", v
#        print "GED_time_step= ", time.time()-GED_start_time
#        ged = v
#    return ged
#
#
#number_node = 10
#
#file = open("Results10New.txt", 'a+')
#
#possible_p=[0.5, 0.7, 0.9]
#
#
#
############################### CONNECTION TO DWAVE ############################################
#
#
##connection to Dwave (USING SAPI)
#
#url = 'https://cloud.dwavesys.com/sapi'
## Token (INSERT YOUR TOKEN)
##token = 'DEV-9ecb88bf6cb1cdf05bc8b160c720d438ca94a544'
##token= 'DEV-29d20b24fd812f99604b2aa14464c76f95d4a801'
#
#
##selects the dwave computer
##solver_name = "DW_2000Q_6"
#solver_name = "Advantage_system1.1"
#
## create a remote connection using url and token
#remote_connection = RemoteConnection(url, token)
#
## get a solver
#solver = remote_connection.get_solver(solver_name)
#
##gets the adjacency matrix of  DW_2000Q_2_1 chimera graph
#Adj = get_hardware_adjacency(solver)
#
############################### END CONNECTION TO DWAVE ############################################
#
#
#
#
#
## probability of edge
#for p1_index in range(len(possible_p)):
#
#    p1=possible_p[p1_index]
#
#    G1 = erdos_renyi_graph(number_node, p1)
#
#    conn = True
#
#    # if graph are isomorphic exec again random generator
#    while conn:
#        if (nx.number_connected_components(G1)==1):
#            conn = False
#        else:
#            G1 = erdos_renyi_graph(number_node, p1)
#
#
#    print
#    print "-------------------------- p1=", p1,"-------------------------- "
#
#
#
#
#
#    for p2_index in range(p1_index,len(possible_p)):
#
#        p2=possible_p[p2_index]
#
#
#        G2 = erdos_renyi_graph(number_node, p2)
#
#        isomorphic = True
#
#        conn = True
#
#        # if graph are isomorphic exec again random generator
#        while isomorphic and conn:
#            if((not nx.is_isomorphic(G1, G2)) and (nx.number_connected_components(G2)==1)):
#                conn = False
#                isomorphic = False
#                print_graph(G1, G2)
#            else:
#                G2 = erdos_renyi_graph(number_node, p2)
#
#
#
#
#        G1=nx.convert_node_labels_to_integers(G1,first_label=1)
#        G2=nx.convert_node_labels_to_integers(G2,first_label=1)
#
#        edgesG1=[e1 for e1 in G1.edges]
#        edgesG2=[e2 for e2 in G2.edges]
#
#
#
#        ################################# END OF INPUT ####################################
#
#        #Uses Tobias' class instance, loaded with the instance parameters (edgesG1, edgesG2)
#        inst = instance.Instance(edgesG1, edgesG2)
#
#        #PRINTS THE PROPERTIES OF THE  INSTANCE
#        print
#        print "-------------------------- p2=", p2,"-------------------------- "
#        print
#        print "Properties of the input graph"
#
#        print "edges G1:",  inst.edgesG1
#
#        print "edges G2:", inst.edgesG2
#
#        print "number of vertices G1:", inst.NVerticesG1
#
#        print "number of vertices G2:", inst.NVerticesG2
#
#        print
#
#
#        file.write(" \n")
#        file.write(" \n")
#        file.write("------------------------------------------------ p1="+str(p1)+" ------------------------------------------------")
#        file.write(" \n")
#        file.write(" \n")
#        file.write("------------------------------------------------ p2="+str(p2)+" ------------------------------------------------")
#        file.write(" \n")
#        file.write(" \n")
#        file.write("Properties of the input graph")
#        file.write(" \n")
#        file.write("edges G1: "+str(inst.edgesG1))
#        file.write(" \n")
#        file.write("edges G2: "+str(inst.edgesG2))
#        file.write(" \n")
#        file.write("number of vertices G1: "+str(inst.NVerticesG1))
#        file.write(" \n")
#        file.write("number of vertices G2: "+str(inst.NVerticesG2))
#        file.write(" \n")
#        file.write(" \n")
#
#
#
#
#        if inst.NVerticesG1 != inst.NVerticesG2 :
#            raise ValueError("FATALITY: Graphs do not have the same number of vertices")
#
#
#        # modficare variable.py ogni indice di sommatoria mettere una variabile binaria controllare nella fomrula.
#
#        #creates binary variables x(v,i), y(v,i) and z(v,i) and flattens indeces
#        #Flattening: to each (v,i) associates a flat indices j
#        var= variable.Variable(inst)
#
#        #print to show how variables are flattened
#        print ("Flat index: from x(v,i) to x(j) ")
#        print (var.x)
#        print
#
#        file.write("Flat index: from x(v,i) to x(j) ")
#        file.write(" \n")
#        file.write(str(var.x))
#        file.write(" \n")
#        file.write(" \n")
#
#        ####################################################################
#        #construction of the QUBO
#
#        #first penalty term A1 (see notes for formula)
#        #define the first term as an empty polymonial
#        A1= qcfco.polynomial.Polynomial()
#        #the two for loops represent the two sums
#        for v in range(1,inst.NVerticesG2+1):
#            poly = qcfco.polynomial.Polynomial({(): -1})
#            for i in range(1,inst.NVerticesG1+1):
#                #if (v, i) in var.x.keys():
#                poly += qcfco.polynomial.Polynomial({(var.x[(v, i)],): 1})
#            temp = poly * poly
#            A1 += temp #each term inside the first sum  must be squared
#            # print(A1.toString())
#            # print()
#
#
#
#
#
#
#        #second penalty term A2 (see notes for formula)
#        #at first define the A2 term as an empty polymonial
#        A2= qcfco.polynomial.Polynomial()
#        #the two for loops represent the two sums
#        for i in range(1,inst.NVerticesG1+1):
#            poly = qcfco.polynomial.Polynomial({(): -1})
#            for v in range(1,inst.NVerticesG2+1):
#                if (v, i) in var.x.keys():
#                    poly += qcfco.polynomial.Polynomial({(var.x[(v, i)],): 1})
#            A2 += poly * poly
#
#
#
#
#        # crate independent set of edgeG1
#        IndipSetG1=[]
#        for i in range(1, inst.NVerticesG1):
#            for k in range(i+1, inst.NVerticesG1+1):
#                check = False
#                for edge in edgesG1:
#                        if edge[0]==i and k==edge[1]:
#                                check = True
#                if not check:
#                        tupla = (i, k)
#                        IndipSetG1.append(tupla)
#
#        #add all reversed edges e.g. if (1,2)  in indepset then also (2,1) in indepset
#        for val in range(len(IndipSetG1)):
#            if len(IndipSetG1) != 0:
#                FirstVal = IndipSetG1[val][0]
#                SecondVal = IndipSetG1[val][1]
#                ReverseEdge = (SecondVal, FirstVal)
#                IndipSetG1.append(ReverseEdge)
#
#
#        IndipSetG2=[]
#        for i in range(1, inst.NVerticesG2):
#            for k in range(i+1, inst.NVerticesG2+1):
#                check = False
#                for edge in edgesG2:
#                        if edge[0]==i and k==edge[1]:
#                                check = True
#                if not check:
#                        tupla = (i, k)
#                        IndipSetG2.append(tupla)
#        for val1 in range(len(IndipSetG2)):
#            if len(IndipSetG2) != 0:
#                FirstVal = IndipSetG2[val1][0]
#                SecondVal = IndipSetG2[val1][1]
#                ReverseEdge = (SecondVal, FirstVal)
#                IndipSetG2.append(ReverseEdge)
#
#
#        #first penalty term B1 start to 0
#        #check se è necessario il reverse per costruire B1 e B2 (riga 203)
#        B1= qcfco.polynomial.Polynomial()
#        if len(IndipSetG1) != 0:
#            for i in range(len(IndipSetG1)):
#                for u in range(len(edgesG2)):
#                    B1 += qcfco.polynomial.Polynomial({(var.x[(np.int64(edgesG2[u][0]), np.int64(IndipSetG1[i][0]))], var.x[(np.int64(edgesG2[u][1]), np.int64(IndipSetG1[i][1]))]): 1})
#        else:
#            # if G1 is completely connected first term of HB is 0 because there isn't edge in the indipendent set, first sum is 0
#            B1 += qcfco.polynomial.Polynomial({(): 0})
#
#        # print ("B1 term:" )
#        # print (B1.toString())
#        # print
#        # print ("B1 term in dictionary form:" )
#        # print (B1.getDWaveQUBO())
#        # print
#
#        #secondo penalty term B2 start to 0
#        B2= qcfco.polynomial.Polynomial()
#        if len(IndipSetG2) != 0:
#            for i in range(len(edgesG1)):
#                for u in range(len(IndipSetG2)):
#                    B2 += qcfco.polynomial.Polynomial({(var.x[(np.int64(IndipSetG2[u][0]), np.int64(edgesG1[i][0]))], var.x[(np.int64(IndipSetG2[u][1]), np.int64(edgesG1[i][1]))]): 1})
#        else:
#            B2 += qcfco.polynomial.Polynomial({(): 0})
#
#
#        # print ("B2 term:" )
#        # print (B2.toString())
#        # print
#        # print ("B2 term in dictionary form:" )
#        # print (B2.getDWaveQUBO())
#
#        # print
#        writeinfo=0
#
#        for A in [1, 1.5, 2, 2.5, 5]:
#
#
#
#            #Choose the parameters A and B (remember A>B*c_max) where c_max is the max weight
#            #A=15
#            # con A=1 V=3 A = 1.5 v=4 A = 2 V = 6
#
#            B=1
#
#            # sum of equation component
#            Q = A*(A1+A2)+B*(B1+B2)
#
#            quboA = A*(A1+A2)
#            quboB = B*(B1+B2)
#
#
#
#            qubo=Q.getDWaveQUBO()[0]
#            offset=Q.getDWaveQUBO()[1]
#
#            # #writes qubo in txt file
#            # file = open('qubo.txt', 'a+')
#            # file.write(str(qubo))
#            # file.close()
#
#            # #generate the S dictionary: useful for find embedding
#            # S=qubo.copy()
#
#            # for i in range(len(S)):
#            #    S[S.items()[i][0]]=1
#
#            # #print stuff
#
#
#
#            if writeinfo==0:
#                print
#                print 'number of logical qubits:', var.x[max(var.x)]+1
#                print
#                print
#                print "number of edges in Qubo"
#                print len(Q.getDWaveQUBO()[0]) - var.x[max(var.x)]+1
#
#                writeinfo=1
#
#                file.write(" \n")
#                file.write("number of logical qubits: "+str(var.x[max(var.x)]+1))
#                file.write(" \n")
#                file.write(" \n")
#                file.write("number of edges in Qubo: "+str(len(Q.getDWaveQUBO()[0]) - var.x[max(var.x)]+1))
#                file.write(" \n")
#                file.write(" \n")
#
#
#
#            print
#            print "------------------ A=", A, "---------------------"
#
#            file.write(" \n")
#            file.write("---------------------------------------------A="+str(A)+" ------------------------------------------------")
#            file.write(" \n")
#            file.write(" \n")
#
#
#
#
#            (h, J, ising_offset) = qubo_to_ising(qubo)
#
#
#            nphys_min=10000
#            for e in range(1):
#
#                #embed J into the chimera A
#                embedding = find_embedding(qubo, Adj, verbose=0)
#
#                #nphys gives the number of physical qubits
#                nchain=[None for x in range(len(embedding))]
#                for j in range(len(embedding)):
#                    nchain[j]= len(embedding[j])
#                nphys=sum(nchain)
#                if nphys<nphys_min:
#                    nphys_min=nphys
#                    best_embedding= embedding
#
#
#            print 'best_embedding=',best_embedding
#            print ('number of physical qubits:', nphys_min)
#
#
#            file.write(" \n")
#            file.write("best_embedding="+str(best_embedding))
#            file.write(" \n")
#            file.write(" \n")
#            file.write("number of physical qubits:"+str(nphys_min))
#            file.write(" \n")
#            file.write(" \n")
#
#            #################################################################
#
#            (h0, j0, jc, new_emb) = embed_problem(h, J, best_embedding, Adj)
#
#            emb_j = j0.copy()
#
#            emb_j.update(jc)
#
#            h_norm=max(abs(max(h0)),abs(min(h0)))
#
#            j_norm=max(abs(max(j0.values())),abs(min(j0.values())))
#
#            norm=max(h_norm,j_norm)
#
#            for count in range(len(h0)):
#                h0[count]=float(h0[count]/(norm))
#
#            for key in j0.keys():
#                j0[key]=float(j0[key]/norm)
#
#            j_extended=dict(jc)
#
#            for x in j_extended.keys():
#                j_extended[x]=-2.0
#
#
#            emb_j = j0.copy()
#
#            #emb_j.update(jc)
#            emb_j.update(j_extended)
#
#
#            print
#            print ('number of physical edges:', len(emb_j))
#            print
#
#
#            file.write("number of physical edges:"+str(len(emb_j)))
#            file.write(" \n")
#            file.write(" \n")
#
#            #################################################################
#
#
#            #parameters of the run
#            num_anneals=1
#
#            schedule=[[0.0, 0.0], [1.0, 1.0]]
#            #schedule=[[0.0, 0.0], [1, 0.4], [21, 0.4],[22, 1.0]]
#            #schedule=[[0.0, 0.0], [10.0, 0.3], [20.0, 1.0]]
#
#            params = {"anneal_schedule": schedule, "num_reads": num_anneals, "auto_scale": False, "flux_drift_compensation":True }
#
#
#
#
#            start_time = time.time()
#
#            result = solve_ising(solver, h0, emb_j, **params)
#
#
#            DWSolverTime = time.time() - start_time
#
#
#
#            new_answer = unembed_answer(result['solutions'], new_emb, 'vote', h, J)
#
#
#
#
#
#
#            samplerSA = neal.SimulatedAnnealingSampler()
#
#            start_time_SA = time.time()
#
#            resultSA = samplerSA.sample_qubo(qubo, solver=neal.SimulatedAnnealingSampler() , num_reads= 100  )
#
#            SASolverTime = time.time() - start_time_SA
#
#
#
#            print
#            print "DWSolverTimeDWSolverTime= ", DWSolverTime
#            print
#            print result["timing"]
#            print
#            print "SASolverTime= ", SASolverTime
#            print
#
#            #new_answer = unembed_answer(result['solutions'], sampler, 'vote' )
#            file.write(" \n")
#            file.write("DWSolverTime= "+str(DWSolverTime))
#            file.write(" \n")
#            file.write(" \n")
#            file.write("SASolverTime= "+str(SASolverTime))
#            file.write(" \n")
#            file.write(" \n")
#            file.write("result.info[timing]:"+str(result["timing"]))
#            file.write(" \n")
#            file.write(" \n")
#
#
#            print("Solution from solver")
#            print
#
#            file.write(" \n")
#            file.write("Solution from solver")
#            file.write(" \n")
#
#
#
#            binary_answer = []
#            ground_energy=[]
#            occur=[]
#            possible_solution = False
#            # set -1 value to 0, result is tuple of list of tuple
#
#            for s in range(len(new_answer)):
#                binary_answer=new_answer[s]
#                for n, i in enumerate(binary_answer):
#                    if i == -1:
#                        binary_answer[n] = 0
#
#
#                # check if solutions respect the hard costraint
#                # check soft costraint and save relativ occurence
#                if quboA.evaluate(binary_answer)==0:
#                    ground_energy.append(quboB.evaluate(binary_answer))
#                    possible_solution = True
#                    occur.append(result['num_occurrences'][s])
#                    print(str(binary_answer))
#
#
#
#
#
#
#
#            # if energy is empty no found a solution
#            if ground_energy == []:
#                ground_energy.append(Q.evaluate(binary_answer))
#                occur.append(0)
#
#
#            #Among all the valid solutions found in this round of 10k runs
#            #stored in occur and ground energy, we select only those that give minimum energy
#            #and store their occurrences as a list called total_occur
#            total_occur=[]
#            for k in range(len(ground_energy)):
#                if ground_energy[k]==min(ground_energy):
#                    total_occur.append(occur[k])
#
#            #sum all the occurrences with min energy for this round and append to a list
#            #these 2 lists contain all the best occurences found in all rounds of 10k runs
#
#            best_solution = []
#            best_solution_occurrence = []
#            if possible_solution:
#                best_solution.append(min(ground_energy))
#                best_solution_occurrence.append(sum(total_occur))
#
#                print("Best solution DW")
#                print(best_solution)
#                print("Best solution occurence DW")
#                print(best_solution_occurrence)
#
#                file.write(" \n")
#                file.write("Best solution DW")
#                file.write(" \n")
#                file.write(str(best_solution))
#                file.write(" \n")
#                file.write("Best solution occurrences DW")
#                file.write(" \n")
#                file.write(str(best_solution_occurrence))
#                file.write(" \n")
#
#
#            else:
#                print("Solution not found DW")
#                print("Hard constraint not verify")
#                print
#
#                file.write(" \n")
#                file.write("Solution not found DW")
#                file.write(" \n")
#                file.write(" \n")
#
#
#
#            sol = resultSA.record
#
#            binary_answer = []
#            ground_energy=[]
#            occur=[]
#            possible_solution = False
#            # set -1 value to 0, result is tuple of list of tuple
#
#
#
#
#            for i in range(len(sol)):
#
#                binary_answer =sol[i][0]
#
#                # check if solutions respect the hard costraint
#                # check soft costraint and save relativ occurence
#                if quboA.evaluate(binary_answer)==0:
#                    ground_energy.append(quboB.evaluate(binary_answer))
#                    possible_solution = True
#                    occur.append(sol[i][2])
#                   # print(str(binary_answer))
#
#
#
#            # if energy is empty no found a solution
#            if ground_energy == []:
#                ground_energy.append(Q.evaluate(binary_answer))
#                occur.append(0)
#
#
#            #Among all the valid solutions found in this round of 10k runs
#            #stored in occur and ground energy, we select only those that give minimum energy
#            #and store their occurrences as a list called total_occur
#            total_occur=[]
#            for k in range(len(ground_energy)):
#                if ground_energy[k]==min(ground_energy):
#                    total_occur.append(occur[k])
#
#            #sum all the occurrences with min energy for this round and append to a list
#            #these 2 lists contain all the best occurences found in all rounds of 10k runs
#
#            best_solution = []
#            best_solution_occurrence = []
#            if possible_solution:
#                best_solution.append(min(ground_energy))
#                best_solution_occurrence.append(sum(total_occur))
#
#                print
#                print("Best solution Sa")
#                print(best_solution)
#                print("Best solution occurence  SA")
#                print(best_solution_occurrence)
#
#                file.write(" \n")
#                file.write("Best solution SA")
#                file.write(" \n")
#                file.write(str(best_solution))
#                file.write(" \n")
#                file.write("Best solution occurrences SA")
#                file.write(" \n")
#                file.write(str(best_solution_occurrence))
#                file.write(" \n")
#
#            else:
#                print("Solution not found SA")
#                print("Hard constraint not verify")
#                file.write(" \n")
#                file.write("Solution not found SA")
#                file.write(" \n")
#
#
#
#            #print
#            #ged = classic_edit_distance(G1, G2)
#            #print
#            #print "classical ged= ", ged
#
#            # file.write(" \n")
#            # file.write("classical ged= ")
#            # file.write(" \n")
#            # file.write(str(ged))
#            # file.write(" \n")
#
#file.close()
            
            
            
            
