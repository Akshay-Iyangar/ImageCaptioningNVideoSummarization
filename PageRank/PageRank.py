import sentence_similarity as ss

import numpy as np
from scipy.sparse import csc_matrix

#save global mapping for frame to index
dictVidFrameToIdx=dict()
dictIdxToVidFrame=dict()
dictGraph=dict()

def pageRankFrameSim(G, alp = 0.85, convergeThreshold = .01, maxIter=50):
    """
    Computes the pagerank significance measure for the video frame.
    G is a matrix representing normalized values for similarity extracted from the input 
    graph of task 2
    ----------
    alpha is the probability of picking a frame connected to this frame based on similrity. 1-alpha is the probability of teleporting. Defaults to 0.85.
    convergeThreshold is a value at which the algorithm is said to converge when the difference between the previous and current page rank values is less than or equal to it. Defaults to 0.001
    """
    dims = G.shape
    if dims[0] != dims[1]:
        print "Matrix is not square!"
        sys.exit(1)
    n=dims[0]
    print n
    
    # transform G into sparsematrix M
    M = csc_matrix(G,dtype=np.float)
    #print M
    
    
    rsum = np.array(M.sum(1))
    rowsum=rsum[:,0]

    # Compute pagerank r until we converge
    po, p= np.zeros(n), np.ones(n)
    flag= True
    iter=0
    while np.sum(np.abs(p-po)) > convergeThreshold and iter<maxIter:
        iter=iter+1
        print iter
        po = p.copy()
        for i in xrange(0,n):
            # in-similarity values of frame i, i.e. similarity scores vector corresponding to frames that have i as the top-k similar frame
            Ii = np.array(M[:,i].todense())[:,0]
            # account for teleportation to frame i
            Ti = np.ones(n) / float(n)
            # Weighted PageRank Equation
            #r[i] = ro.dot( Ii*alpha/2.0 + Si*s + Ti*(1-s)*G[i] ) 
            p[i] = po.dot( Ii*alp + Ti*(1-alp))
        print "current diff :"
        print np.sum(np.abs(p-po))
    # return normalized pagerank
    print "paused at :"
    print np.sum(np.abs(p-po))
    return p/sum(p)

import csv
def readCSV(file):
    idx=0;
    with open(file, 'rb') as csvfile:

        graphReader = csv.reader(csvfile, delimiter=',')
        for inRow in graphReader:
            row=list(map(float, inRow))
            print '{} {} {} {} {}'.format(row[0], row[1], row[2], row[3], row[4])
            '''
            store a mapping of video#|frame# to graph index in dictVidFrameToIdx
            and an inverse mapping in dictIdxToVidFrame
            '''
            f1=inRow[0]+'|'+inRow[1]
            f2=inRow[2]+'|'+inRow[3]
            n1=dictVidFrameToIdx.get(f1, idx) # get the index of frame f1, if f1 isn't in the dictinary, get() will return idx
            if(n1==idx):#new frame -- add to dictionary
                dictVidFrameToIdx[f1]=idx
                dictIdxToVidFrame[idx]=f1
                idx=idx+1
            n2=dictVidFrameToIdx.get(f2, idx)
            if(n2==idx):
                dictVidFrameToIdx[f2]=idx;
                dictIdxToVidFrame[idx]=f2
                idx=idx+1
            '''
            store the score in the graph dictionary 
            using the tuple of f1 and f2 indices as key
            '''
            dictGraph[(n1, n2)]=row[4]
        '''
        at the end of this loop the idx value is the 
        total number of nodes in our graph
        '''
        #from scipy.sparse import dok_matrix
        #G = dok_matrix((idx, idx), dtype=np.float32)
        G=np.empty((idx,idx), dtype=float)
        for i in range(idx-1):
            for j in range(idx-1):
                G[i, j] =  dictGraph.get((i, j), 0)   # create the final graph from the graph dictionary-setting to 0 where there is no corresponding value
    from sklearn.preprocessing import normalize
    normG = normalize(G, norm='l1', axis=1)
    return normG

import sys, getopt
if __name__=='__main__':
    '''
    #Test matrix
    G = np.array([[0,0,1,0,0,0,0],
                  [0,1,1,8,0,0,0],
                  [1,0,0,1.3,0,0,0],
                  [0,0,0,1,1,0,0],
                  [0,0,0,0,0,0,1],
                  [0,8,0,0,0,1.6,1.1],
                  [0,8,0,1,1.2,0,1.1]])
    
    from sklearn.preprocessing import normalize
    GNorm = normalize(G, norm='l1', axis=1)
    print GNorm
    '''
    # Read the csv file
    #filename='filename_d_k.csv'
    filename=sys.argv[1]
    m=int(sys.argv[2])
#    GNorm=readCSV(filename);
    

    with open('imagedescription.txt') as f:
        sentences = f.read().splitlines()

    #sentences = sentence[0:7]

    print sentences

    #for sent1 in sentences:
    #    for sent2 in sentences:
    #        print "%s\t%s\t%.3f\t" % (sent1, sent2, similarity(sent1, sent2, True))


    sim_matrix = []
    for sent1 in sentences:
        temp = [0]*len(sentences)
        i=0
        for sent2 in sentences:
            #print(sent1+ '  ' +sent2)
            if(sent1 != sent2):
                temp[i] = ss.similarity(sent1,sent2,True)
            i = i+1
        sim_matrix.append(temp)

    #print(sim_matrix)
    similarityMatrix = np.array([sim_matrix])
    sortedIndices = np.argsort(similarityMatrix,axis=2)

    print(similarityMatrix)
    print(sortedIndices)

    for val in sortedIndices:
        i=0
        for row in val:
            print row
            n=len(row)
            j=0
            for f in row:
                print f
                if j<n-2:
                    similarityMatrix[0][i][f]=0
                j=j+1
            i=i+1
    print "Sparse Matrix"
    print similarityMatrix
    from sklearn.preprocessing import normalize
    for G in similarityMatrix:
        GNorm=normalize(G, 'l1')

frameSignificance=pageRankFrameSim(GNorm,.86)
print frameSignificance
#get sorted indices for significance vector
sortIndices=np.argsort(frameSignificance)
print sortIndices
revSortIndices = np.argsort(sortIndices[::-1])
#Use the reverse sorted indices to get the top m frames since this is similarity from the index to frame dictionary
outFile = open('PageRank.csv', 'w')
'''
for i in xrange(0,m):
    out=dictIdxToVidFrame[revSortIndices[i]].replace('|', ',')
    print out
    outFile.write(out)
    outFile.write("\n")
outFile.close()
'''
