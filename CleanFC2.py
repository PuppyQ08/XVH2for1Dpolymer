import subprocess as sbp
import sys
import os
import numpy as np
import numpy.linalg as la
import pandas as pd
import time
import math
from ast import literal_eval
from pdb import set_trace as pst

'''
decfreq01: The original opitimization with negative freq as first one.
decfreq02: Move the atoms to direction of negative freq then got the normal positive freq.
decfreq03: Extract from the decfreq02's last optimized geometry.

All the indexs start with 0!

All carbon number should be even and should be odd's double !! why? otherwise need to check cellsnumber
'''
#=========================prefix setup part==============================#
#filesname_FC2fchk = 'decfreq02'
filesname_FC3fchk = 'C14H30Freq'
filesname_FC2fchk = 'C34H70Freq'
#filesname_FC2fchk = 'C34H70HFixed'
#filesname_com = 'decfreq03cart'
filesname_FC3com = 'C14H30Freq'
filesname_FC2com ='C34H70Freq'
filesname_FC3csv = 'FC3_C14AnaHess.csv'

#++++++++++++++constant setting++++++++++++++++++++++++++++++++++++++

meconstant = 1822.888486
Ang_bohr = 1.8897259886 
au_cm = 4.359743E-18/(1.660538E-27 * 0.5292E-10 * 0.5292E-10/meconstant)#hatree/(amu*bohr*bohr) it transfer to SI unit

len_a = 2.567381*Ang_bohr #The average length of cell(transfer to bohr)
massau = [12.0107*meconstant,1.00794*meconstant] #From NIST database
#XXX the K is depended on how many the cells we used. here FC only take neighbor 1 cell so in total is 3 cells
#klist= np.linspace(0,1,K//2+1)#XXX here still use 0-1 but later should times pi/len_a when using
#FC4klist = np.linspace(0,2,K4+1)[:K4//2+1]#XXX here still use 0-1 but later should times pi/len_a when using
#XXX: plus cellsnumber and endidx
#........................... 2nd 
#XXX this global variables could only be modified in local scope but not redefined.
FC2Coef_kp = {}
#............................3rd
FC2atomsname= []
#coordA = []
cal_method = '#p freq B3YLP/6-31G(d)'
#atomcharge = 0
#atommult = 1

'''
         H13 H14
           \/ 
C4-- C1 -- C2 -- C3
     /\    
   H11 H12
'''

#===============================HARMONIC PART===============================#
def harmFreq_per_k():
    for i in range(len(FC2klist)):
        getCoef_w_perk(i)
    print("The w (omg) in a.u is :\n")
    print(w_omgkpcm[0])
    #XXX the following is to check Coeficient is right
    #eigvaltest = np.zeros((len(FC2klist),P),dtype = np.complex_)
    #for _p in range(P):
    #    for kidx in range(len(FC2klist)):
    #        for kappa in range(P):
    #            atom1 = 3*(cellsnumber[0][kappa//3] - 1) + kappa%3
    #            for gamma in range(P):
    #                for midx in range(-endidx,endidx + 1):
    #                    atom2 = 3*(cellsnumber[midx][gamma//3] - 1) + gamma%3
    #                    eigvaltest[kidx][_p] += FC2[getidx(atom1,atom2)] * Coef_kp[kidx][kappa][_p] * Coef_kp[kidx][gamma][_p].conjugate()* math.e**(- 1j * midx * klistFC2[kidx] * math.pi) / (math.sqrt(massau[int(kappa>5)]*massau[int(gamma>5)]))
    #print(w_omgkp)
    #print(eigvaltest)

#For now I just calculate the neaby cells
#Fuvk is 18*18 for atoms in first cell but Force constant was store in 96*96 but in lower dense triangular form.
#XXX: u and v is the uth vth Cartesian Coordinates!!!

def getCoef_w_perk(kidx,Fcc):
    kk = FC2klist[kidx] + 0.1
    Fuvk = np.zeros((P,P),dtype = np.complex_)
    #XXX: m is just -1 0 1 for decane 

    for u in range(P):
        atom1 = 3*(FC2cellsnumber[0][u//3] - 1) + u%3
        for v in range(P):
            eachterm = 0.0
            for midx in range(-FC2endidx,FC2endidx+1):
                #  F u(0)v(m) : 
                #  Cell[m] [v//3] give us the atoms number in FC matrix XXX:which started with 1!
                #  atom2 is the nth coordinates of each atoms XXX: which started with 0!
                atom2 = 3*(FC2cellsnumber[midx][v//3] - 1) + v%3
                #  transfer to k space
                eachterm +=  Fcc[getidx(atom1,atom2)]* math.e ** (-1j * kk * midx*math.pi)#/(math.sqrt(massau[int(u>5)]*massau[int(v>5)]))

                #eachterm +=  Fcc[atom1,atom2]* math.e ** (-1j * kk * midx * len_a)
            #   mass weighted : if u and v is > 5 so it is not Carbon's coordinates
            Fuvk[u][v] = eachterm /(math.sqrt(massau[int(u>5)]*massau[int(v>5)]))
    eigval, eigvector = la.eigh(Fuvk)#hermition matrix to get real eigenvalue
    #print(eigval)
    for i in range(P):
        w_omgkp[kidx][i] = math.sqrt(abs(eigval[i]))
        w_omgkpcm[kidx][i] = math.sqrt(abs(eigval[i]*au_cm))/(2.99792458E10 * 2 * math.pi)
    print(w_omgkpcm[kidx])

    FC2Coef_kp[kidx] = eigvector.conjugate() #here we add v is a p*p matrix (p is branch number and number of atoms in cell
    return eigvector, Fuvk ,eigval
    #df = pd.DataFrame(w_omgkpcm)
    #df.to_csv('./w_omgkpcmNorm.csv')
    
def cleanFC2():
    #XXX My way

    #test = []
    #u = 1
    #atom1 = 3*(FC2cellsnumber[0][u//3] - 1) + u%3
    #v = 3
    #for midx in range(-FC2endidx,FC2endidx+1):
    #    atom2 = 3*(FC2cellsnumber[midx][v//3] - 1) + v%3
    #    test.append(FC2[getidx(atom1,atom2)])
    #print(test)
    #print(w)
    #XXX Sode way
    #mass weighted first
    FCinput = FC2.copy()
    for u in range(P):
        atom1 = 3*(FC2cellsnumber[0][u//3] - 1) + u%3
        for v in range(P):
            for midx in range(-FC2endidx,FC2endidx+1):
                atom2 = 3*(FC2cellsnumber[midx][v//3] - 1) + v%3
                FCinput[getidx(atom1,atom2)] = FC2[getidx(atom1,atom2)]/(math.sqrt(massau[int(u>5)]*massau[int(v>5)]))
    L,D0,k = getCoef_w_perk(0,FCinput.copy()) 
    print(L[:,0])
    print(L[:,1])
    print(L[:,2])
    print(L[:,3])
    #I = np.eye(P)
    #L1 = np.outer(L[:,0],L[:,0])
    #L2 = np.outer(L[:,1],L[:,1])
    #L3 = np.outer(L[:,2],L[:,2])
    #L4 = np.outer(L[:,3],L[:,3])
    ##Pp = (I - L1@L1)@(I - L2@L2)@(I - L3@L3)@(I - L4@L4)
    #Pp = (I - L4@L4)
    #corrct = (Pp@D0@Pp - D0)/(15)
    ##print(corrct.shape)
    #FC2new = np.zeros(FC2.shape,dtype = np.complex_)
    #for u in range(P):
    #    atom1 = 3*(FC2cellsnumber[0][u//3] - 1) + u%3
    #    for v in range(P):
    #        for midx in range(-FC2endidx,FC2endidx+1):
    #            atom2 = 3*(FC2cellsnumber[midx][v//3] - 1) + v%3
    #            FC2new[getidx(atom1,atom2)] = FCinput[getidx(atom1,atom2)] + corrct[u,v]
    #FCinput = FC2new.copy()
    #L,D0,k = getCoef_w_perk(1,FCinput.copy()) 



    

    
    

    #return Fnew
   

#XXX:Works really well! Check!
#def C14harmonicFreqCheck():
#    eigvaltestOriginFC3 = np.zeros((len(FC3klist),P),dtype = np.complex_)
#    for kk in range(len(FC3klist)):
#        Fuvk = np.zeros((P,P),dtype = np.complex_)
#        #XXX: m is just -1 0 1 for decane 
#        #Carbon 1
#        for kappa in range(P):
#            atom1 = 3*(FC3cellsnumber[0][kappa//3] - 1) + kappa%3
#            for gamma in range(P):
#                eachterm = 0.0
#                for midx in range(-FC3endidx,FC3endidx+1):
#                    #  F u(0)v(m) : 
#                    #  Cell[m] [v//3] give us the atoms number in FC matrix XXX:which started with 1!
#                    #  atom2 is the nth coordinates of each atoms XXX: which started with 0!
#                    atom2 = 3*(FC3cellsnumber[midx][gamma//3] - 1) + gamma%3
#                    #  transfer to k space
#                    eachterm +=  FC3FC2[getidx(atom1,atom2)]* math.e ** (-1j * klistFC3[kk] * midx * math.pi)
#                #   mass weighted : if u and v is > 5 so it is not Carbon's coordinates
#                Fuvk[kappa][gamma] = eachterm /(math.sqrt(massau[int(kappa>5)]*massau[int(gamma>5)]))
#        eigval, eigvector = la.eigh(Fuvk)#hermition matrix to get real eigenvalue
#        for i in range(P):
#            eigvaltestOriginFC3[kk][i] = math.sqrt(abs(eigval[i]*au_cm))/(2.99792458E10 * 2 * math.pi)
#    print(eigvaltestOriginFC3)
#    eigvaltestFC3 = np.zeros((len(FC3klist),P),dtype = np.complex_)
#    for _p in range(P):
#        for kidx in range(len(FC3klist)):
#            for kappa in range(P):
#                atom1 = 3*(FC3cellsnumber[0][kappa//3] - 1) + kappa%3
#                for gamma in range(P):
#                    for midx in range(-FC3endidx,FC3endidx + 1):
#                        atom2 = 3*(FC3cellsnumber[midx][gamma//3] - 1) + gamma%3
#                        eigvaltestFC3[kidx][_p] += FC3FC2[getidx(atom1,atom2)] * FC2Coef_kp[3*kidx][kappa][_p] * FC2Coef_kp[3*kidx][gamma][_p].conjugate()* math.e**(- 1j * midx * FC2klist[3* kidx] * math.pi) / (math.sqrt(massau[int(kappa>5)]*massau[int(gamma>5)]))
#            eigvaltestFC3[kidx][_p] = math.sqrt(abs(eigvaltestFC3[kidx][_p] * au_cm))/(2.99792458E10 * 2 * math.pi)
#    print(eigvaltestFC3)

            


#===============================ANHARM PART=============================#
#read in the csv file for force constant directly.
#TODO:Finish the code for polyethylene (already have FC) 
#TODO:- readin FC - transfer FC to k space - diagrams - find root - last step.
"""
FC3 is stored in csv file need to read in 
"""
#===============================HELPER FUNCTION========================#
"""
#helper function to readin the fchk FC2 and store in array and return copy
"""
def readFC2(filename):
    for fname in os.listdir('.'):
        if fname == filename + '.fchk':
            with open(fname) as f:
                search = f.readlines()
    for fcidx in range(len(search)):
        eachline = search[fcidx].split()
        if eachline and eachline[0] == "Cartesian" and eachline[1] == "Force":
            fcnum = int(eachline[5])
            break
    tempFC2 = [0]*fcnum
    i = 0
    plus = int(fcnum%5==0)
    for itr in range(fcidx+1, fcidx+int(fcnum)//5+2- plus):
        for ele in search[itr].split():
            tempFC2[i] = float(ele)
            i+=1
    return tempFC2

"""
#get idx of FCs
"""
def getidx(*args):#XXX:started with 0!
    output = list(args)
    if len(args)==2:
        output.sort()
        return int(output[1]*(output[1]+1)/2 + output[0])
    elif len(output) == 3:
        output.sort()
        return str(output[0]) + '_' + str(output[1]) + '_' + str(output[2])
    elif len(output) == 4:
        output.sort()
        return str(output[0]) + '_' + str(output[1]) + '_' + str(output[2]) +'_' + str(output[3])
    sys.exit("wrong input for idx()")
    return 0    

"""
#cells setting 
#return a numpy array of the index of the cell atoms
"""
def cellsetting():
    ##totalnum = len(FC2atomsname)
    ##assert (totalnum-2)%3 == 0
    ###eg carbon_num is 10
    FC2carbon_num = int((len(FC2atomsname)-2)/3)
    ###eg numcellused is 3
    FC2numcellused = int((FC2carbon_num-4)/2)
    ##assert (carbon_num-4)%2 == 0

    global FC2cellsnumber
    FC2cellsnumber = np.zeros((FC2numcellused,6))#XXX:we use EVEN number of carbon here!!! and cut off the end 4 carbons
    FC2cellsnumber = FC2cellsnumber.astype(int)

    FC2cellsnumber[:2,:2] = np.array([[1,2],[3,5]])
    FC2cellsnumber[FC2numcellused//2 + 1,:2] = np.array([FC2carbon_num - 4,FC2carbon_num - 6])
    for i in range(FC2numcellused):
        if i > 1 and i < FC2numcellused//2 + 1:
            FC2cellsnumber[i,:2] = FC2cellsnumber[i-1,:2] + 4
        elif i > FC2numcellused//2 + 1 :
            FC2cellsnumber[i,:2] = FC2cellsnumber[i-1,:2] - 4
        for j in range(1,3):
            FC2cellsnumber[i,2*j] = 2*(FC2cellsnumber[i,j-1]-1) +  FC2carbon_num  +1
            FC2cellsnumber[i,2*j+1] = 2*(FC2cellsnumber[i,j-1]-1) + FC2carbon_num  +2  
    FC2cellused = len(FC2cellsnumber)#XXX should be odd
    global FC2endidx
    FC2endidx = FC2cellused//2# if  cellused is 3 then endidx is 3//2 = 1 so the range is (-1, 2)
    print("For FC2 number of cells used is", FC2numcellused,"and the endidx is", FC2endidx)

    print(FC2cellsnumber)


'''
#get atoms name, charge, multi num, coordA(actually no use here)  
'''
def init_para():
    with open(filesname_FC2com + ".com") as f:
        read = f.readlines()
    for idx in range(len(read)):
        eachline = read[idx].split() 
        if eachline and  eachline[0] == "calculation":
            break
    idx += 3 #move from the title section to coordinates part
    while read[idx]!= '\n':
        eachline = read[idx].split()
        FC2atomsname.append(eachline[0])
        #for cdidx in range(1, len(eachline)):
            #coordA.append(float(eachline[cdidx]))
        idx+=1
    print("The number of FC2 atoms is",len(FC2atomsname))
    
    #readin the FC2 of the oject
    global FC2
    FC2 = np.array(readFC2(filesname_FC2fchk))

    global K
    K = 15 
    global K2
    K2 = 15 # number of cells harmonic
    K3 = 5 # number of cells FC3 
    K4 = 3 # number of cells FC4
    N = 6
    global P
    P = 3*N #branch number of normal modes in first BZ
    global FC2klist
    FC2klist = np.linspace(0,2,K2+1)[:K2//2+1]#XXX here still use 0-1 but later should times pi/len_a when using
    #FC3klist = np.linspace(0,2,K3+1)[:K3//2+1]#XXX here still use 0-1 but later should times pi/len_a when using
    #global FC3FC2
    #FC3FC2 = readFC2(filesname_FC3fchk)
    global w_omgkp
    global w_omgkpcm
    w_omgkp = np.zeros((len(FC2klist),P))#We just store the half BZ plus zero's w_omg since they are symmetric 
    w_omgkpcm = np.zeros((len(FC2klist),P)) 


#===================================TEST PART ==============================#
t1 = time.time()
init_para()
cellsetting()
#cleanFC2()

L,D0,k = getCoef_w_perk(0,FC2)
#do mass-weighted back
#print(L.real)
for i in range(4):
    temp = L[i,:].real.copy()
    print(temp)
    #for a in range(len(temp)):
    #    temp[a] *= (math.sqrt(massau[int(a>5)]*massau[int(i>5)]))

#print("cellsnumber is ,",FC2cellsnumber)#,FC3cellsnumber)
print(time.time()-t1)
#testpart(0)
