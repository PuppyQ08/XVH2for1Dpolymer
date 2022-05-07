import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
from scipy.interpolate import make_interp_spline, BSpline

filename = 'K20AnharmPhononDisOnlynoYellow.npy'
temp = np.load(filename)

#w_omgkpcm = np.load('w_omg.npy')
w_omgkpcm = temp[0] 
print(w_omgkpcm)
klist = np.array([i for i in range(21)])
anh = temp[1]
#overtone = np.load('overtones.npy')
#selected = [[]]*21
#for i in range(21):
#    for each in overtone[i]:
#        a = w_omgkpcm[i][0]
#        b = w_omgkpcm[i][1]
#        c = w_omgkpcm[i][2]
#        d = w_omgkpcm[i][3]
        #if ((each > a - 10 and each < a+10) or (each > b - 10 and each < b+10) or(each > c - 10 and each < c+10) or(each > d - 10 and each < d+10) ):
#        if (each < 800):
#            selected[i].append(each)
#    plt.plot([klist[i]]*len(selected[i]),selected[i],'bx',markersize=3)

#print(selected)
#print(overtone.shape)
#print("done")
#print(selected[0])
#print(w_omgkpcm[0])

#plt.show()

#print(anh)
#print(temp)
#harmo = df[
#xnew = np.linspace(klist[:8].min(),klist[:8].max(),300)
#spl = make_interp_spline(klist[:8],w_omgkpcm[:8,:4],k=3)
#smooth = spl(xnew)
#XXX 0- 700 [0:4]

#print(anh[2])
def one():
    line1 = w_omgkpcm[:21,0]
    dot1 = anh[:21,0]

    line2 = np.zeros(21)
    line2 [:10] = w_omgkpcm[:10,1]
    line2[10:] = w_omgkpcm[10:,2]
    dot2 = np.zeros(21)
    dot2 [:10] = anh[:10,1]
    dot2[10:] =  anh[10:,2]

    line3 = np.zeros(21)
    line3[:10] = w_omgkpcm[:10,2]
    line3[10:] = w_omgkpcm[10:,1]
    dot3 = np.zeros(21)
    dot3[:10]= anh[:10,2]
    dot3[10:] = anh[10:,1]

    line4 = np.zeros(21)
    line4 = w_omgkpcm[:,3]
    dot4 = np.zeros(21)
    dot4 = anh[:,3]
    plt.plot(klist,line1,'b-')
    plt.plot(klist,line2,'r-')
    plt.plot(klist,line3,'y-')
    plt.plot(klist,line4,'g-')

    plt.plot(klist,dot1,'bo',markersize=1)
    plt.plot(klist,dot2,'ro',markersize=1)
    plt.plot(klist,dot3,'yo',markersize=1)
    plt.plot(klist,dot4,'go',markersize=1)

    plt.show()

#XXX 700 - 1500 [4:8]
def two():
    line1 = w_omgkpcm[:,4]
    dot1 = anh[:,4]
    
    line2 = np.zeros(21)
    line2[:4] = w_omgkpcm[:4,5]
    line2[4:6] = w_omgkpcm[4:6,6]
    line2[6:] = w_omgkpcm[6:,7]
    dot2 = np.zeros(21)
    dot2[:4] = anh[:4,5]
    dot2[4:6]= anh[4:6,6]
    dot2[6:] = anh[6:,7]
    
    line3 = np.zeros(21)
    line3[:4] = w_omgkpcm[:4,6]
    line3[4:] = w_omgkpcm[4:,5]
    dot3 = np.zeros(21)
    dot3[:4] = anh[:4,6]
    dot3[4:] = anh[4:,5]
    
    line4 = np.zeros(21)
    line4[:6] = w_omgkpcm[:6,7]
    line4[6:] = w_omgkpcm[6:,6]
    dot4 = np.zeros(21)
    dot4[:6] = anh[:6,7]
    dot4[6:] = anh[6:,6]
    plt.plot(klist,line1 ,'b-')
    plt.plot(klist,line2 ,'r-')
    plt.plot(klist,line3 ,'y-')
    plt.plot(klist,line4 ,'g-')
    plt.plot(klist,dot1,'bo')
    plt.plot(klist,dot2,'ro')
    plt.plot(klist,dot3,'yo')
    plt.plot(klist,dot4,'go')
    plt.show()

#XXX 1500 - 2000 8:14

def three():
    line1 = w_omgkpcm[:,8]
    dot1 = anh[:,8]
    
    line2 = np.zeros(21)
    line2[:17] = w_omgkpcm[:17,9]
    line2[17:] = w_omgkpcm[17:,10]
    dot2 = np.zeros(21)
    dot2[:17] = anh[:17,9]
    dot2[17:] = anh[17:,10]
    
    line3 = np.zeros(21)
    line3[:17] = w_omgkpcm[:17,10]
    line3[17:] = w_omgkpcm[17:,9]
    dot3 = np.zeros(21)
    dot3[:17] = anh[:17,10]
    dot3[17:] = anh[17:,9]
    
    line4 = w_omgkpcm[:,11]
    line5 = w_omgkpcm[:,12]
    line6 = w_omgkpcm[:,13]
    dot4 = anh[:,11]
    dot5 = anh[:,12]
    dot6 = anh[:,13]
    plt.plot(klist,line1 ,'b-')
    plt.plot(klist,line2 ,'r-')
    plt.plot(klist,line3 ,'y-')
    plt.plot(klist,line4 ,'g-')
    plt.plot(klist,line5 ,'c-')
    plt.plot(klist,line6 ,'k-')
    plt.plot(klist,dot1 ,'bo')
    plt.plot(klist,dot2 ,'ro')
    plt.plot(klist,dot3 ,'yo')
    plt.plot(klist,dot4 ,'go')
    plt.plot(klist,dot5 ,'co')
    plt.plot(klist,dot6 ,'ko')
    plt.show()

#XXX 3000 14:18
def four():
    #plt.plot(klist,w_omgkpcm[:,14:18],'bo')
    plt.plot(klist,w_omgkpcm[:,14],'b-')
    plt.plot(klist,w_omgkpcm[:,15],'r-')
    plt.plot(klist,w_omgkpcm[:,16],'y-')
    plt.plot(klist,w_omgkpcm[:,17],'g-')
    plt.plot(klist,anh[:,14],'bo')
    plt.plot(klist,anh[:,15],'ro')
    plt.plot(klist,anh[:,16],'yo')
    plt.plot(klist,anh[:,17],'go')
    plt.show()
one()
#two()
#three()
#four()


plt.show()
