import subprocess as sbp
import sys
import os
import numpy as np
import numpy.linalg as la
import pandas as pd
import time
import math
import matplotlib.pyplot as plt 
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
filesname_FC3fchk = 'CubicC14'
filesname_FC2fchk = 'C34H70Freq'
#filesname_FC2fchk = 'C34H70HFixed'
#filesname_com = 'decfreq03cart'
filesname_FC3com = 'C14H30Freq'
filesname_FC2com = 'C34H70Freq'
#filesname_FC3csv = 'FC3_C14AnaHE3.csv'

#++++++++++++++constant setting++++++++++++++++++++++++++++++++++++++

meconstant = 1822.888486
Ang_bohr = 1.8897259886 
au_cm = 4.359743E-18/(1.660538E-27 * 0.5292E-10 * 0.5292E-10/meconstant)#hatree/(me a.u*bohr*bohr) it transfer to SI unit

len_a = 2.567381*Ang_bohr #The average length of cell(transfer to bohr)
massau = [12.0107*meconstant,1.00794*meconstant] #From NIST database
#XXX the K is depended on how many the cells we used. here FC only take neighbor 1 cell so in total is 3 cells
N = 6
P = 3*N #branch number of normal modes in first BZ
#........................... 2nd 
#coordA = []
#cal_method = '#p freq B3YLP/6-31G(d)'
#atomcharge = 0
#atommult = 1

'''
         H13 H14
           \/ 
C4-- C1 -- C2 -- C3
     /\    
   H11 H12
'''

"""
#cells setting 
#return a numpy array of the index of the cell atoms
"""
def cellsetting(FC2atomsname,FC3atomsname):
    '''
    ++++++++++++++++++++++++++++++++++++++++++++++cells number setting for FC2 and FC3++++++++++++++++++++
    '''

    FC2carbon_num = int((len(FC2atomsname)-2)/3)
    FC2numcellused = int((FC2carbon_num-4)/2)
    assert (FC2carbon_num-4)%2 == 0

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
            FC2cellsnumber[i,2*j] = 2*(FC2cellsnumber[i,j-1]-1) +  FC2carbon_num +1
            FC2cellsnumber[i,2*j+1] = 2*(FC2cellsnumber[i,j-1]-1) + FC2carbon_num +2  
    FC2cellused = len(FC2cellsnumber)#XXX should be odd
    global FC2endidx
    FC2endidx = FC2cellused//2# if  cellused is 3 then endidx is 3//2 = 1 so the range is (-1, 2)
    print("For FC2 number of cells used is", FC2numcellused,"and the endidx is", FC2endidx)
    global FC3numcellused 
    FC3carbon_num = int((len(FC3atomsname)-2)/3)
    FC3numcellused = int((FC3carbon_num-4)/2)
    assert (FC3carbon_num-4)%2 == 0

    global FC3cellsnumber
    FC3cellsnumber = np.zeros((FC3numcellused,6))#XXX:we use EVEN number of carbon here!!! and cut off the end 4 carbons
    FC3cellsnumber = FC3cellsnumber.astype(int)

    FC3cellsnumber[:2,:2] = np.array([[1,2],[3,5]])
    FC3cellsnumber[FC3numcellused//2 + 1,:2] = np.array([FC3carbon_num - 4,FC3carbon_num - 6])
    for i in range(FC3numcellused):
        if i > 1 and i < FC3numcellused//2 + 1:
            FC3cellsnumber[i,:2] = FC3cellsnumber[i-1,:2] + 4
        elif i > FC3numcellused//2 + 1 :
            FC3cellsnumber[i,:2] = FC3cellsnumber[i-1,:2] - 4
        #
        for j in range(1,3):
            FC3cellsnumber[i,2*j] = 2*(FC3cellsnumber[i,j-1]-1) +  FC3carbon_num +1
            FC3cellsnumber[i,2*j+1] = 2*(FC3cellsnumber[i,j-1]-1) + FC3carbon_num +2  
    global FC3endidx
    FC3endidx = FC3numcellused//2# if  cellused is 3 then endidx is 3//2 = 1 so the range is (-1, 2)
    print("For FC3 number of cells used is", FC3numcellused,"and the endidx is", FC3endidx)
    '''
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    '''

    #K2 = int(FC2carbon_num/2 - 2 )#get rid of side cell
    #K3 = int(FC3carbon_num/2 - 2) #get rid of side cell
    K3 = K2 = 40
    global FC2klist
    FC2klist = np.linspace(0,2,K2+1)[:-1]#XXX here still use 0-1 but later should times pi/len_a when using
    global FC3klist
    print("The k number list of Harmonic, 3rd order FC, 4th order FC are followed")
    FC2klist[K2//2+1:] -=2   
    FC3klist = np.linspace(0,2,K3+1)[:-1]#XXX here still use 0-1 but later should times pi/len_a when using
    FC3klist[K3//2+1:] -=2
    print("the k list for FC2 harmonic one is :\n",FC2klist)
    print("the k list for FC3 transformation is :\n",FC3klist)
    print("=======================================================================")

    global w_omgkpcm
    global w_omgkp
    w_omgkp = np.zeros((len(FC2klist),P))#We just store the half BZ plus zero's w_omg since they are symmetric 
    w_omgkpcm = np.zeros((len(FC2klist),P)) 

'''
#get atoms name, charge, multi num, coordA(actually no use here)  
'''
def init_para():
    FC2atomsname = []
    FC3atomsname = []
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
    
    with open(filesname_FC3com + ".com") as f:
        read = f.readlines()
    for idx in range(len(read)):
        eachline = read[idx].split() 
        if eachline and  eachline[0] == "calculation":
            break
    idx += 3 #move from the title section to coordinates part
    while read[idx]!= '\n':
        eachline = read[idx].split()
        FC3atomsname.append(eachline[0])
        #for cdidx in range(1, len(eachline)):
            #coordA.append(float(eachline[cdidx]))
        idx+=1
    print("The number of FC3 atoms is",len(FC3atomsname))
    cellsetting(FC2atomsname,FC3atomsname)
    
    #readin the FC2 of the oject
    global FC2origin
    FC2origin = readFC2(filesname_FC2fchk)
    #global FC3FC2
    #FC3FC2 = readFC2(filesname_FC3fchk)
    #global FC3orig
    #global FC3new
    #Ncoord = len(FC3atomsname)*3
    #leng = int(Ncoord*(Ncoord + 1)/2)
    #FC3orig = np.zeros((P, leng))
    #FC3new = np.zeros((P,P,len(FC3klist),P,len(FC3klist))) #XXX F kap 0 lamb m1 mu m2
    global FC2Coef_kp
    FC2Coef_kp = np.zeros((len(FC2klist),P,P),dtype = np.complex_)


                            
#===============================HARMONIC PART===============================#
def eachkfreq(kidx, Fcc):
    kk = FC2klist[kidx]#*math.pi/len_a
    Fuvk = np.zeros((P,P),dtype = np.complex_)
    #XXX: m is just -1 0 1 for decane 
    #Carbon 1
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
                eachterm +=  Fcc[getidx(atom1,atom2)]* math.e ** (-1j * kk * midx * math.pi)
            #   mass weighted : if u and v is > 5 so it is not Carbon's coordinates
            Fuvk[u][v] = eachterm# /(math.sqrt(massau[int(u>5)]*massau[int(v>5)]))
    eigval, eigvector = la.eigh(Fuvk)#hermition matrix to get real eigenvalue
    return eigval, eigvector, Fuvk

def harmFreq_per_k():

    transCon = math.sqrt(au_cm)/(2.99792458E10 * 2 * math.pi)
    #XXX clean accoustic branch
    #first got mass-weighted FC2
    FCinput = FC2origin.copy()
    for u in range(P):
        atom1 = 3*(FC2cellsnumber[0][u//3] - 1) + u%3
        for v in range(P):
            for midx in range(-FC2endidx,FC2endidx+1):
                atom2 = 3*(FC2cellsnumber[midx][v//3] - 1) + v%3
                FCinput[getidx(atom1,atom2)] = FC2origin[getidx(atom1,atom2)]/(math.sqrt(massau[int(u>5)]*massau[int(v>5)]))
    #self-consistantly correct the force constants
    #eigv,L,D0 = eachkfreq(0,FCinput.copy()) 
    #print(math.sqrt(eigv[3])*transCon)
    eigv,L,D0 = eachkfreq(0,FCinput.copy()) 
    #print(eigv[2],eigv[3])
    I = np.eye(P)
    L1 = np.outer(L[:,0],L[:,0])
    L2 = np.outer(L[:,1],L[:,1])
    L3 = np.outer(L[:,2],L[:,2])
    L4 = np.outer(L[:,3],L[:,3])
    Pp = (I - L1@L1)@(I - L2@L2)@(I - L3@L3)@(I - L4@L4)
    corrct = (Pp@D0@Pp - D0)/15
    FC2new = FCinput.copy()
    for u in range(P):
        atom1 = 3*(FC2cellsnumber[0][u//3] - 1) + u%3
        for v in range(P):
            for midx in range(-FC2endidx,FC2endidx+1):
                atom2 = 3*(FC2cellsnumber[midx][v//3] - 1) + v%3
                FC2new[getidx(atom1,atom2)] = FCinput[getidx(atom1,atom2)] + corrct[u,v]
    FCinput = FC2new.copy()
    for kidx in range(len(FC2klist)):
        eigval,eigvector,D0 = eachkfreq(kidx,FCinput.copy())
        for i in range(P):
            w_omgkp[kidx][i] = math.sqrt(abs(eigval[i]))
            w_omgkpcm[kidx][i] = math.sqrt(abs(eigval[i]*au_cm))/(2.99792458E10 * 2 * math.pi)

        FC2Coef_kp[kidx] =np.array(eigvector.conjugate()) #here we add v is a p*p matrix (p is branch number and number of atoms in cell
    print("The w (omg) in a.u is :\n")
    #print(w_omgkpcm)
    #plt.plot(FC2klist[:21],w_omgkpcm[:21,:4],'bo')
    #plt.show()


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

#For now I just calculate the neaby cells
#Fuvk is 18*18 for atoms in first cell but Force constant was store in 96*96 but in lower dense triangular form.
#XXX: u and v is the uth vth Cartesian Coordinates!!!


#XXX:Works really well! Check!
def C14harmonicFreqCheck():
    eigvaltestOriginFC3 = np.zeros((len(FC3klist),P),dtype = np.complex_)
    for kk in range(len(FC3klist)):
        Fuvk = np.zeros((P,P),dtype = np.complex_)
        #XXX: m is just -1 0 1 for decane 
        #Carbon 1
        for kappa in range(P):
            atom1 = 3*(FC3cellsnumber[0][kappa//3] - 1) + kappa%3
            for gamma in range(P):
                eachterm = 0.0
                for midx in range(-FC3endidx,FC3endidx+1):
                    #  F u(0)v(m) : 
                    #  Cell[m] [v//3] give us the atoms number in FC matrix XXX:which started with 1!
                    #  atom2 is the nth coordinates of each atoms XXX: which started with 0!
                    atom2 = 3*(FC3cellsnumber[midx][gamma//3] - 1) + gamma%3
                    #  transfer to k space
                    eachterm +=  FC3FC2[getidx(atom1,atom2)]* math.e ** (-1j * klistFC3[kk] * midx * math.pi)
                #   mass weighted : if u and v is > 5 so it is not Carbon's coordinates
                Fuvk[kappa][gamma] = eachterm /(math.sqrt(massau[int(kappa>5)]*massau[int(gamma>5)]))
        eigval, eigvector = la.eigh(Fuvk)#hermition matrix to get real eigenvalue
        for i in range(P):
            eigvaltestOriginFC3[kk][i] = math.sqrt(abs(eigval[i]*au_cm))/(2.99792458E10 * 2 * math.pi)
    print(eigvaltestOriginFC3)
    eigvaltestFC3 = np.zeros((len(FC3klist),P),dtype = np.complex_)
    for _p in range(P):
        for kidx in range(len(FC3klist)):
            for kappa in range(P):
                atom1 = 3*(FC3cellsnumber[0][kappa//3] - 1) + kappa%3
                for gamma in range(P):
                    for midx in range(-FC3endidx,FC3endidx + 1):
                        atom2 = 3*(FC3cellsnumber[midx][gamma//3] - 1) + gamma%3
                        eigvaltestFC3[kidx][_p] += FC3FC2[getidx(atom1,atom2)] * FC2Coef_kp[3*kidx][kappa][_p] * FC2Coef_kp[3*kidx][gamma][_p].conjugate()* math.e**(- 1j * midx * FC2klist[3* kidx] * math.pi) / (math.sqrt(massau[int(kappa>5)]*massau[int(gamma>5)]))
            eigvaltestFC3[kidx][_p] = math.sqrt(abs(eigvaltestFC3[kidx][_p] * au_cm))/(2.99792458E10 * 2 * math.pi)
    print(eigvaltestFC3)

            


#===============================ANHARM PART=============================#
"""
F_mkm pkp qkq = 
F kappa(0) lambd (m1) mu (m2)

"""
def checkkpp(kp,p):
    #kill the Yellow term
    #if (((p in [0,1,2,3]) and kp == 0) or (p == 2 and kp in list(range(1,10))+ list(range(32,41))) or (p == 1 and kp in list(range(10,32))) ):
    if ((p in [0,1,2,3]) and kp == 0):
    #if (((p in [0,1,2,3]) and kp == 0) or ( p in list(range(3)) and kp in list(range(1,10)) + list(range(32,41))) or (p == 1 and kp in list(range(10,32)) )):
        return True 
    else:
        return False 

def f2(m,km,v,Fpkpqkqrkr):
    ret = 0.0
    #F mkm pkp qkq
    #XXX FC3 Warning
    for _p in range(P):
        for _q in range(P):
            for kp in range(len(FC3klist)):
                kq = kqidxFC3gen(kp,km)
                if (checkkpp(kp,_p) or checkkpp(kq,_q))  :
                    pass
                else:
                    oneterm = Fpkpqkqrkr[km,kp,kq,m,_p,_q]/math.sqrt(len(FC3klist)*2**3*w_omgkp[km][m]*w_omgkp[kp][_p]*w_omgkp[kq][_q])
                    eachterm = (oneterm.real**2 + oneterm.imag**2) /(- v - w_omgkp[kp][_p] - w_omgkp[kq][_q])
                    ret += eachterm
    return ret/2

def e2(m,km,v,Fpkpqkqrkr):
    ret = 0.0
    idx = 0
    #F m-km pkp qkq
    #XXX: FC3 Warning 
    if (km != 0):
        km = len(FC3klist) - km 
    for _p in range(P):
        for _q in range(P):
            for kp in range(len(FC3klist)):
                kq = kqidxFC3gen(kp,km)
                if (checkkpp(kp,_p) or checkkpp(kq,_q) ) :
                    pass
                else:
                    oneterm = Fpkpqkqrkr[km,kp,kq,m,_p,_q]/math.sqrt(len(FC3klist)*2**3*w_omgkp[km][m]*w_omgkp[kp][_p]*w_omgkp[kq][_q])
                    check = (oneterm.real**2 + oneterm.imag**2)/( v - w_omgkp[kp][_p] - w_omgkp[kq][_q])
                    ret += check 
    #print(ret/2*transCon)
    return ret/2
def a2b2(m,km,Fpkpqkqrkr):
    ret = 0.0
    if (km != 0):
        negekm = len(FC3klist) - km
    else:
        negekm = km
    for kq in range(len(FC3klist)):
        if (kq!=0):
            negekq = len(FC3klist) - kq
        else:
            negekq = kq
        for p in range(4,P):
            for q in range(P):
                for r in range(P):
                    if (checkkpp(negekq,r) or checkkpp(kq,q) or checkkpp(negekm,m)):
                        pass
                    else:
                        one = Fpkpqkqrkr[0,km,negekm,p,m,m]/math.sqrt(len(FC3klist) *2**3*w_omgkp[0][p]*w_omgkp[km][m]*w_omgkp[negekm][m])
                        two = Fpkpqkqrkr[0,kq,negekq,p,q,r]/math.sqrt(len(FC3klist) *2**3*w_omgkp[0][p]*w_omgkp[kq][q]*w_omgkp[negekq][r])
                        each = one*two/(w_omgkp[0][p])
                        ret -= each 
    return ret

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

def readFC3(filename):
    for fname in os.listdir('.'):
        if fname == filename + '.fchk':
            with open(fname) as f:
                search = f.readlines()
    fcidx = 0
    for i in range(len(search)):
        eachline = search[i].split()
        if eachline and eachline[0] == "Cartesian" and eachline[2] == "derivatives":
            fcnum = int(eachline[5])
            fcidx = i
            break
    plus = int(fcnum%5==0)
    endis =  int(fcidx + fcnum//5 + 2 - plus)
    tempFC3 = [0]*fcnum
    idx = 0
    for itr in range(fcidx+1, endis):
        for ele in search[itr].split():
            tempFC3[idx] = float(ele)
            idx += 1
    global FC3new
    FC3new = tempFC3
    return tempFC3


"""
#get idx of FCs
"""
def getidx(*args):#XXX:started with 0!
    output = list(args)
    if len(args)==2:
        output.sort()
        return int(output[1]*(output[1]+1)/2 + output[0])
    elif len(args)==3:
        output.sort(reverse=True)
        i = output[0]
        #XXX this is the analytical sum for 1 + 1+2 + 1+2+3 ... 1+2..+k 
        # equals to (sum k + sum k**2)/2
        firstsum = (2*i**3 + 6*i**2 + 4*i)/12
        lasttwosum = output[1]*(output[1]+1)/2 + output[2]
        return int(firstsum + lasttwosum)
    return 0    


#===================================TEST PART ==============================#
def kqidxFC3gen(kp,kr):#XXX default for FC3 5 cells
    #XXX for 3rd order: 2n - kp - km equals to 2 - kp - km but need to shift back to 1st BZ  
    kq = (2*len(FC3klist) - kp - kr) % len(FC3klist) 
    return int(kq)
   
    
def allFC3inK():#kq kp kr is all index in klist
    #kp kq kr outermost
    F = np.zeros((len(FC3klist),len(FC3klist),len(FC3klist),P,P,P),dtype = np.complex_)
    for kp in range(len(FC3klist)):
        for kr in range(len(FC3klist)):
            kq  = kqidxFC3gen(kp,kr)
            FKapLam_r = np.zeros((P,P,P),dtype = np.complex_)
            FKap_q_r = np.zeros((P,P,P),dtype = np.complex_)
            Fp_q_r = np.zeros((P,P,P),dtype = np.complex_)
            for p in range(P):
                for q in range(P):
                    for r in range(P):
                        if (F[kp,kq,kr,p,q,r] == 0.):
                            if (Fp_q_r[p,q,r] != 0.):
                                F[kp,kq,kr,p,q,r] =F[kp,kr,kq,p,r,q] = F[kq,kr,kp,q,r,p] = F[kq,kp,kr,q,p,r] = F[kr,kp,kq,r,p,q] = F[kr,kq,kp,r,q,p] = Fp_q_r[p,q,r]#/math.sqrt(len(FC3klist)) 
                            else:
                                F[kp,kq,kr,p,q,r] =F[kp,kr,kq,p,r,q] = F[kq,kr,kp,q,r,p] = F[kq,kp,kr,q,p,r] = F[kr,kp,kq,r,p,q] = F[kr,kq,kp,r,q,p] = FCmemorize(kp,kq,kr,p,q,r,FKapLam_r,FKap_q_r, Fp_q_r)#/math.sqrt(len(FC3klist))
    return F

def FCmemorize(kp,kq,kr,p,q,r,FKapLam_r,FKap_q_r, Fp_q_r):
    #t1 = time.time()
    #muFwrap = np.zeros(P,dtype = np.complex_)
    #for mu in range(P):
    #    muFwrap[mu] = FCtimesE(kp,kq,kr,p,q,r,kap,lamb,mu)
    #muTotal = np.dot(muFwrap,FC2Coef_kp[3*kr][:,r])
    #t2 = time.time()
    #print(muTotal,"time is ",t2 - t1)
    tempp = 0.0
    for kap in range(P):
        if (FKap_q_r[kap,q,r] != 0.):
            tempp += FC2Coef_kp[kp][kap][p] * FKap_q_r[kap,q,r]
        else:
            temp = 0.0
            for lamb in range(P):
                if(FKapLam_r[kap,lamb,r] != 0.):
                    temp += FC2Coef_kp[kq][lamb][q] * FKapLam_r[kap,lamb,r]
                else:
                    ret = 0.0
                    for mu in range(P):
                        ret += FC2Coef_kp[kr][mu][r] * FCtimesE(kp,kq,kr,p,q,r,kap,lamb,mu)
                    FKapLam_r[kap,lamb,r] = ret 
                    temp += FC2Coef_kp[kq][lamb][q] * ret
            FKap_q_r[kap,q,r] = temp
            tempp += FC2Coef_kp[kp][kap][p] * temp
    Fp_q_r[p,q,r] = tempp
    return tempp
         
         
def FCtimesE(kp,kq,kr,p,q,r,kap,lamb,mu):
    #t1 = time.time()
    #ret= 0.0
    #for m1 in range(-FC3endidx,FC3endidx+1):
    #    atom1 = 3*(FC3cellsnumber[m1][lamb//3] - 1) + lamb%3
    #    for m2 in range(-FC3endidx,FC3endidx+1):
    #        atom2 = 3*(FC3cellsnumber[m2][mu//3] - 1) + mu%3
    #        ret += FC3[kap][getidx(atom1,atom2)]* math.e**(1j*(m1*FC2klist[3*kq]+ m2*FC2klist[3*kr]) * math.pi)/ math.sqrt(massau[int(kap)>5]*massau[int(lamb)>5]*massau[int(mu)>5])

    #XXX use numpy is faster!
    FCarraytemp = np.zeros(FC3numcellused*FC3numcellused,dtype = np.complex_)
    EE =  np.zeros(FC3numcellused*FC3numcellused,dtype =np.complex_)
    idx = 0
    for m1 in range(-FC3endidx,FC3endidx+1):
        for m2 in range(-FC3endidx,FC3endidx+1):
            atom1 = 3*(FC3cellsnumber[m1][lamb//3] - 1) + lamb%3
            atom2 = 3*(FC3cellsnumber[m2][mu//3] - 1) + mu%3
            atom3 = 3*(FC3cellsnumber[0][kap//3] - 1) + kap%3
            FCarraytemp[idx] = FC3new[getidx(atom1,atom2,atom3)]
            EE[idx] = math.e**(1j*(m1*FC3klist[kq]+ m2*FC3klist[kr])*math.pi)
            idx+=1
    ret = np.dot(FCarraytemp,EE)
    ret /=math.sqrt(massau[int(kap)>5]*massau[int(lamb)>5]*massau[int(mu)>5])
    return ret 
    

#=============================== Find Root ============================================#
def intervals(km,m,rangeipt):# it is for e2 omega's intervals
    #interv = w_omgkp.flatten()
    print(rangeipt)
    ret = [0.0]
    combination = [0.0]
    print( w_omgkpcm[km][m] - rangeipt[0], w_omgkpcm[km][m] - rangeipt[1])
    if (km != 0):
        km = len(FC3klist) - km
    for kp in range(len(FC3klist)):
        kq = kqidxFC3gen(kp,km)
        for _p in range(P):
            for _q in range(P):
                if (checkkpp(kp,_p) or checkkpp(kq,_q)):
                    pass
                else:
                    eachterm = w_omgkp[kp][_p] + w_omgkp[kq][_q]
                    if not (eachterm in ret):
                        #if(not((w_omgkpcm[kp][_p]< 10) or (w_omgkpcm[kq][_q]< 10))):
                        if (eachterm*transCon > w_omgkpcm[km][m] - rangeipt[0] and eachterm*transCon < w_omgkpcm[km][m] - rangeipt[1]):
                            ret.append(eachterm)
                            combination.append((kp,_p,kq,_q))
    ret.sort()
    return ret, combination

def diagramsVdep(m,km,v):
    return e2(m,km,v,Fk) + f2(m,km,v,Fk) #+ a2b2(m,km,Fk)

def fn_v_square(m,km,v,w,constdiag):
    return w**2 + 2 * w *(diagramsVdep(m,km,v) + constdiag) - v**2
#-------------------------------------------------------------------------
def bisect_root_find(m,km,interv,comb):
    constantdiag = a2b2(m,km,Fk)
    w = w_omgkp[km][m]
    roots = []
    realcomb = []
    for i in range(1,len(interv)):
        left = interv[i - 1]  + 1E-15
        right = interv[i] - 1E-15
        print("letf",left*transCon,"right",right*transCon)
        if (abs(left - right) < 1E-14):
            pass
        else:
            leftv = fn_v_square(m,km,left,w,constantdiag)
            rightv = fn_v_square(m,km,right,w,constantdiag)
            if np.sign(leftv) == np.sign(rightv):
                pass
            else:
                root = 0.0
                while abs(right - left) > 1E-16:
                    mid = left/2 + right/2
                    midv = fn_v_square(m,km,mid,w,constantdiag)
                    if abs(midv.real**2 + midv.imag**2) < 1E-16:
                        root = mid
                        break
                    elif np.sign(midv) == np.sign(leftv):
                        root = mid
                        left = mid
                        leftv = midv
                    elif np.sign(midv) == np.sign(rightv):
                        root = mid
                        right = mid
                        rightv = midv
                roots.append(root)
                realcomb.append(comb[i])
    return roots,realcomb

def f2trans(m,km,v,Fpkpqkqrkr):
    ret = 0.0
    #F mkm pkp qkq
    #XXX FC3 Warning
    for _p in range(P):
        for _q in range(P):
            for kp in range(len(FC3klist)):
                kq = kqidxFC3gen(kp,km)
                if (checkkpp(kp,_p) or checkkpp(kq,_q))  :
                    pass
                else:
                    oneterm = Fpkpqkqrkr[km,kp,kq,m,_p,_q]/math.sqrt(len(FC3klist)*2**3*w_omgkp[km][m]*w_omgkp[kp][_p]*w_omgkp[kq][_q])
                    eachterm = (oneterm.real**2 + oneterm.imag**2) /((- v - w_omgkp[kp][_p] - w_omgkp[kq][_q])**2)
                    #if (abs(eachterm*transCon) < 0.01):
                    ret += eachterm
    return ret/2

def e2trans(m,km,v,Fpkpqkqrkr):
    ret = 0.0
    idx = 0
    #F m-km pkp qkq
    #XXX: FC3 Warning 
    if (km != 0):
        km = len(FC3klist) - km 
    for _p in range(P):
        for _q in range(P):
            for kp in range(len(FC3klist)):
                kq = kqidxFC3gen(kp,km)
                if (checkkpp(kp,_p) or checkkpp(kq,_q) ) :
                    pass
                else:
                    oneterm = Fpkpqkqrkr[km,kp,kq,m,_p,_q]/math.sqrt(len(FC3klist)*2**3*w_omgkp[km][m]*w_omgkp[kp][_p]*w_omgkp[kq][_q])
                    check = (oneterm.real**2 + oneterm.imag**2)/(( v - w_omgkp[kp][_p] - w_omgkp[kq][_q])**2)
                    #judge = check*transCon
                    #if ( abs(judge) < 0.01) :
                    ret += check 
    #print(ret/2*transCon)
    return ret/2
def transfn(m,km,v,Fkpqr):
    retF = f2trans(m,km,v,Fkpqr) 
    retE = e2trans(m,km,v,Fkpqr) 
    derivative = -retE+retF
    return derivative

def transitions(m,km,roots,realcomb):
    transit = []
    availroots = []
    TheComb = []
    for i in range(len(roots)):
        v = roots[i]
        derivative = transfn(m,km,v,Fk)
        intenss = w_omgkp[km][m]/(v - w_omgkp[km][m]*derivative)
        print("roots ",roots[i]*transCon,"intesn ",intenss)
        if (intenss > 0.05):
            transit.append(intenss)
            availroots.append(v*transCon)
            TheComb.append(realcomb[i])
    return availroots,transit,TheComb

def DysonRootsIntesity():
    #km = 18 
    #m = 17
    #intervs,comb = intervals(km,m)
    #roots,realcomb = bisect_root_find(m,km,intervs,comb)
    #availroots,trans,TheComb = transitions(m,km,roots,realcomb)
    #print("======================= The One Pointe is ")
    #print("km ",km,"m ",m,"womg ",w_omgkpcm[km][m])
    #for i in range(len(availroots)):
    #    print(availroots[i], " ", trans[i],"  ",TheComb[i])

    RootOutput =    [[] for j in range(len(FC3klist)//2+1)] 
    TransOutput =   [[] for j in range(len(FC3klist)//2+1)] 
    TheCombOutput = [[] for j in range(len(FC3klist)//2+1)] 
    w_omgkpcmOutput = [[] for j in range(len(FC3klist)//2+1)] 
    rangeroot = [50,-10]
    
    #for km in range(21):
    #for m in range(4,P):
    m = 13 
    km = 0 
    intervs,comb = intervals(km,m,rangeroot)
    roots,realcomb = bisect_root_find(m,km,intervs,comb)
    availroots,trans,TheComb = transitions(m,km,roots,realcomb)
    RootOutput[km] = availroots
    TransOutput[km] = trans
    TheCombOutput[km] = TheComb 
    w_omgkpcmOutput[km] = w_omgkpcm[km][m]
    
    print("======================= The One Pointe is ")
    print("km ",km,"m ",m,"womg ",w_omgkpcm[km][m])
    for i in range(len(availroots)):
        print(availroots[i], " ", trans[i],"  ",TheComb[i])
    Tosave=np.array([w_omgkpcm[:,m],TransOutput,RootOutput,TheCombOutput]) 
    #np.save('1200DysonRootIntens_withoutYellow',Tosave)
    #np.save('1500DysonRt_all_4blue',Tosave)



#============================================================================TEST PART +++++++++++++++++++++++++++++++++++++++++++++#
t1 = time.time()
global transCon
transCon = math.sqrt(au_cm)/(2.99792458E10 * 2 * math.pi)
init_para()
print("cellsnumber is (for extracting force constants data)",FC2cellsnumber,FC3cellsnumber)
harmFreq_per_k()
#tempFC3 = readFC3(filesname_FC3fchk)
#Fret = allFC3inK()
#np.save('FC3CleanedfromK20',Fret)
global Fk
Fk = np.load("FC3CleanedfromK20.npy")
DysonRootsIntesity()
#print(transCon)

#m = 2
#km = 4
#intervs,comb = intervals(km,m)
#roots,realcomb = bisect_root_find(m,km,intervs,comb)
#availroots,trans,TheComb = transitions(m,km,roots,realcomb)
#print("======================= The One Pointe is ")
#print(w_omgkpcm[km][m])
#for i in range(len(availroots)):
#    print(availroots[i], " ", trans[i],"  ",TheComb[i])

#XXX test for Dyson Equation
#temp = np.zeros((2,21))
#m = 5
#km = 14
#intervs = intervals(km,m)
#roots = bisect_root_find(m,km,intervs)
#availroots, trans = transitions(m,km,roots)
#print(availroots)
#print(trans)
#for i in range(len(roots)):
#    output[0][i] = roots[i]*transCon
#    output[1][i] = trans[i]
#np.save('rootsandIntensity',output)
#oupt = np.load('rootsandIntensity.npy')


#XXX get the frequencies indep term
#print((diagramsVdep(1,2,w_omgkp[2][1])*transCon).real)
#temp = np.zeros((2,21,18),dtype=np.complex_)
#for m in range(4):
#    for km in range(1,3):
#        print(km,"  ",m,diagramsVdep(m,km,w_omgkp[km][m])*transCon)
#km = 0
#for m in range(4):
#    temp[0][0][m] = w_omgkp[0][m]*transCon
#for m in range(4,P):
#    temp[0][km][m] = w_omgkp[km][m]*transCon
#    temp[1][km][m] = ((w_omgkp[km][m] + diagramsVdep(m,km,w_omgkp[km][m]))*transCon).real
#for km in range(1,21):
#    for m in range(P):
#        temp[0][km][m] = w_omgkp[km][m]*transCon
#        temp[1][km][m] = ((w_omgkp[km][m] + diagramsVdep(m,km,w_omgkp[km][m]))*transCon).real

#np.save('AnharmPhononDisOnlyYellow',temp)



print("Time in hr is :",(time.time()-t1)/3600)
