import numpy as np


def extent(r1,r2,n):
    lr1=np.log10(r1)
    lr2=np.log10(r2)

    return 10.0**(np.linspace(lr1,lr2,n))

def expand2d(a, repFact):
    """Rebin a 2d array to a large size.
    output will be xrepfact,xrepfact of the original
    Basic memory enhancements included to speed things up.
    """
    (A,B)=a.shape
    #both of the below methods take equal length of time to run regardless of array size
    #but much much faster for larger array sizes. Making code easier with the faster version
    #old version kept here.
    #if A*repFact<1000 and B*repFact<1000:
    #    return np.repeat(np.repeat(a,repFact,axis=0),repFact,axis=1)/(repFact*repFact)
    #else:
    #    out=np.zeros((A*repFact,B*repFact),dtype=a.dtype)
    #    for i in range(A):
    #        r=np.repeat(a[i],repFact)
    #        for j in range(repFact):
    #            out[i*repFact+j,:]=r
    #    return out/(float(repFact)*float(repFact))
    out = np.zeros((A * repFact, B * repFact), dtype=a.dtype)
    for i in range(A):
        r = np.repeat(a[i], repFact)
        for j in range(repFact):
            out[i * repFact + j, :] = r
    return out / (float(repFact) * float(repFact))
    

try:
    from numba import jit

    @jit
    def downSample2d(a, sampFact):
        (A, B) = a.shape
        o = np.zeros((int(A/sampFact),int(B/sampFact))).astype('float64')
        for i in range(0,A,sampFact):
            for j in range(0,B,sampFact):
                s = 0.0
                for k in range(sampFact):
                    for l in range(sampFact):
                        s += a[i+k,j+l]
                o[int(i/sampFact),int(j/sampFact)] = s
        return o/float(sampFact*sampFact)


except:
    def downSample2d(a,sampFact):
        (A,B)=a.shape
        A = int(A/sampFact)
        B = int(B/sampFact)
        return np.array([np.sum(a[i:i+sampFact,j:j+sampFact]) for i in range(0,sampFact*A,sampFact) for j in range(0,sampFact*B,sampFact)]).reshape(A,B)/(sampFact*sampFact)



"""
def downSample2d(a, sampFact):
    (A, B) = a.shape
    o = np.zeros((A/sampFact,B/sampFact)).astype('float64')
    summer(a,o,A,B,sampFact)
    return o
@jit(nopython=True)
def summer(a,o,A,B,sampFact):
    sf2 = sampFact*sampFact
    for i in range(0,A,sampFact):
        for j in range(0,B,sampFact):
            s = 0.0
            for k in range(sampFact):
                for l in range(sampFact):
                    s += a[i+k,j+l]
            o[i/sampFact,j/sampFact] = s/sf2
    return o
"""


class line:
    def __init__(self,p1,p2):
        self.m = (p2[1]-p1[1])/(p2[0]-p1[0])
        self.b = p2[1]-self.m*p2[0]
        self.xlim = np.array([min(p1[0],p2[0]),max(p1[0],p2[0])])
        self.ylim = np.array([min(p1[1],p2[1]),max(p1[1],p2[1])])

    def __call__(self,x):
        return self.m*x+self.b
