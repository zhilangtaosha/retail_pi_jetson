import sys
sys.path.append("/sptag/SPTAG/Release")
import SPTAG
import numpy as np

n = 100
k = 3
r = 3

def testBuild(algo, distmethod, x, out):
   i = SPTAG.AnnIndex(algo, 'Float', x.shape[1])
   i.SetBuildParam("NumberOfThreads", '4')
   i.SetBuildParam("DistCalcMethod", distmethod)
   ret = i.Build(x, x.shape[0])
   i.Save(out)

def testBuildWithMetaData(algo, distmethod, x, s, out):
   i = SPTAG.AnnIndex(algo, 'Float', x.shape[1])
   i.SetBuildParam("NumberOfThreads", '4')
   i.SetBuildParam("DistCalcMethod", distmethod)
   if i.BuildWithMetaData(x, s, x.shape[0]):
       i.Save(out)

def testSearch(index, q, k):
   j = SPTAG.AnnIndex.Load(index)
   for t in range(q.shape[0]):
       result = j.Search(q[t], k)
       print (result[0]) # ids
       print (result[1]) # distances

def testSearchWithMetaData(index, q, k):
   j = SPTAG.AnnIndex.Load(index)
   j.SetSearchParam("MaxCheck", '1024')
   for t in range(q.shape[0]):
       result = j.SearchWithMetaData(q[t], k)
       print (result[0]) # ids
       print (result[1]) # distances
       print (result[2]) # metadata

def testAdd(index, x, out, algo, distmethod):
   if index != None:
       i = SPTAG.AnnIndex.Load(index)
   else:
       i = SPTAG.AnnIndex(algo, 'Float', x.shape[1])
   i.SetBuildParam("NumberOfThreads", '4')
   i.SetBuildParam("DistCalcMethod", distmethod)
   if i.Add(x, x.shape[0]):
       i.Save(out)

def testAddWithMetaData(index, x, s, out, algo, distmethod):
   if index != None:
       i = SPTAG.AnnIndex.Load(index)
   else:
       i = SPTAG.AnnIndex(algo, 'Float', x.shape[1])
   i = SPTAG.AnnIndex(algo, 'Float', x.shape[1])
   i.SetBuildParam("NumberOfThreads", '4')
   i.SetBuildParam("DistCalcMethod", distmethod)
   if i.AddWithMetaData(x, s, x.shape[0]):
       i.Save(out)

def testDelete(index, x, out):
   i = SPTAG.AnnIndex.Load(index)
   ret = i.Delete(x, x.shape[0])
   print (ret)
   i.Save(out)

def Test(algo, distmethod):
   x = np.ones((n, 512), dtype=np.float32) * np.reshape(np.arange(n, dtype=np.float32), (n, 1))
   q = np.ones((r, 512), dtype=np.float32) * np.reshape(np.arange(r, dtype=np.float32), (r, 1)) * 2
   m = ''
   for i in range(n):
       m += str(i) + '\n'

   m = m.encode()

   print ("Build.............................")
   testBuild(algo, distmethod, x, 'testindices')
   testSearch('testindices', q, k)
   print ("Add.............................")
   testAdd('testindices', x, 'testindices', algo, distmethod)
   testSearch('testindices', q, k)
   print ("Delete.............................")
   testDelete('testindices', q, 'testindices')
   testSearch('testindices', q, k)

   print ("AddWithMetaData.............................")
   testAddWithMetaData(None, x, m, 'testindices', algo, distmethod)
   print ("Delete.............................")
   testSearchWithMetaData('testindices', q, k)
   testDelete('testindices', q, 'testindices')
   testSearchWithMetaData('testindices', q, k)

if __name__ == '__main__':
   Test('BKT', 'L2')
   Test('KDT', 'L2')
