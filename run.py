#!/usr/bin/env python
# coding: utf-8

# In[139]:


from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
mnist = fetch_openml('mnist_784', version= 1, as_frame= False)


# In[140]:


print(mnist.keys())
print(mnist['data'].shape)
print(mnist['target'].shape)
print(mnist['data'])


# In[141]:


fig = plt.figure(figsize = (15,3))
fig.patch.set_facecolor('white')
for i in range(9):
    plt.subplot(191 + i)
    plt.imshow(mnist['data'][i].reshape(28,28),'gray')
    plt.title(mnist['target'][i])
    plt.axis('off')
plt.close('all')


# In[142]:


# Q1
import numpy as np
totalImage = np.array(mnist['data'])
SumtotalImage = np.sum(totalImage,axis=0)
# print(SumtotalImage)
SumtotalImage=SumtotalImage/70000
plt.imshow(SumtotalImage.reshape(28,28),'gray')
plt.savefig('Q1.jpg')
plt.close('all')


# In[143]:


# Q2
fiveImagelabel = np.array(np.where(mnist['target'] == '5'))
print(fiveImagelabel.shape)
fiveImagelabel = fiveImagelabel.T[:,0:1]
print(fiveImagelabel.shape)
fiveImagedatas = totalImage[fiveImagelabel[:,0],:]


# In[144]:


mean1 = np.mean(fiveImagedatas,axis=0)
mean1 = np.tile(mean1,(6313,1))
fiveImagedatas = fiveImagedatas - mean1 
ScatterMatrix = np.matmul(fiveImagedatas.T,fiveImagedatas)
# print(K)
from scipy.linalg import eigh
eigenvalues, vectors1 = eigh(ScatterMatrix,eigvals=(0,783))
vectors1 = vectors1.T


# In[145]:


# ideigenvalues = eigenvalues.argsort()
# eigenvalues = eigenvalues[ideigenvalues]
# eigenvectors = eigenvectors[:, ideigenvalues]
# lastIndex = len(eigenvalues)-1
# OneRdVal = eigenvalues[lastIndex]
# OneRdVec = eigenvectors[lastIndex]
# OneRdVec = OneRdVec.T
print(eigenvalues)
print(vectors1)


# In[146]:


final_img1 = vectors1[783,:].reshape(28,28)
imgplot2 = plt.imshow(final_img1,plt.cm.gray)
print(eigenvalues[783])


# In[147]:


final_img2 = vectors1[782,:].reshape(28,28)
imgplot3 = plt.imshow(final_img2,plt.cm.gray)
plt.show()
print(eigenvalues[782])


# In[148]:


final_img3 = vectors1[781,:].reshape(28,28)
imgplot4 = plt.imshow(final_img3,plt.cm.gray)
plt.show()
print(eigenvalues[781])


# In[149]:


imgShowQ2 = np.vstack((vectors1[783,:],vectors1[782,:]))
imgShowQ2 = np.vstack((imgShowQ2,vectors1[782,:]))
imgeigenvalues = np.vstack((eigenvalues[783],eigenvalues[782]))
imgeigenvalues = np.vstack((imgeigenvalues,eigenvalues[781]))
for i in range(len(imgShowQ2[:,0])):
    plt.subplot(131 + i)
    plt.imshow(imgShowQ2[i,:].reshape(28,28),'gray')
    formatnumber = (('%.3e' % imgeigenvalues[i]))
    plt.title('λ='+formatnumber)
    plt.axis('off')
plt.savefig('Q2.jpg')
plt.close('all')


# In[150]:


# Q3
ExtractNumber = 3
Top3eigenvectors = vectors1[(784-ExtractNumber):784,:]
ExtractNumber = 10
Top10eigenvectors = vectors1[(784-ExtractNumber):784,:]
ExtractNumber = 30
Top30eigenvectors = vectors1[(784-ExtractNumber):784,:]
ExtractNumber = 100
Top100eigenvectors = vectors1[(784-ExtractNumber):784,:]
# print(Top3eigenvectors[0,:])
# print(first5Image.shape)
# print(Top3eigenvectors.shape)



# In[151]:


fiveImagedatasO = fiveImagedatas + mean1
first5Image = fiveImagedatasO[0,:]
# print(first5Image.shape)
# imgplotfirst5Original = plt.imshow(first5Image.reshape(28,28),plt.cm.gray)
plt.show()
fianl_img5top3 = np.matmul(first5Image,np.matmul(Top3eigenvectors.T,Top3eigenvectors))
fianl_img5top10 = np.matmul(first5Image,np.matmul(Top10eigenvectors.T,Top10eigenvectors))
fianl_img5top30 = np.matmul(first5Image,np.matmul(Top30eigenvectors.T,Top30eigenvectors))
fianl_img5top100 = np.matmul(first5Image,np.matmul(Top100eigenvectors.T,Top100eigenvectors))
# print("3維",fianl_img5top3)

# imgplotfirst5Top3 = plt.imshow(fianl_img5top3.reshape(28,28),plt.cm.gray)
# plt.show()
# imgplotfirst5Top10 = plt.imshow(fianl_img5top10.reshape(28,28),plt.cm.gray)
# plt.show()
# imgplotfirst5Top30 = plt.imshow(fianl_img5top30.reshape(28,28),plt.cm.gray)
# plt.show()
# imgplotfirst5Top100 = plt.imshow(fianl_img5top100.reshape(28,28),plt.cm.gray)
# plt.show()

final_img5Series = np.vstack((first5Image,fianl_img5top3))
final_img5Series = np.vstack((final_img5Series,fianl_img5top10))
final_img5Series = np.vstack((final_img5Series,fianl_img5top30))
final_img5Series = np.vstack((final_img5Series,fianl_img5top100))
print(final_img5Series.shape)
plt.figure(figsize=(10,10))
for i in range(5):
    plt.subplot(151 + i)
    plt.imshow(final_img5Series[i,:].reshape(28,28),'gray')
    plt.axis('off')
plt.savefig('Q3.jpg')
plt.close('all')


# In[152]:


# Q4
first10000img = totalImage[0:10000,:]
first10000label = np.array(mnist['target'][0:10000])
print(first10000label.shape)
print(first10000img.shape)
Imagelabel1 = np.array(np.where((first10000label == '1')))
Imagelabel2 = np.array(np.where((first10000label == '3')))
Imagelabel3 =  np.array(np.where((first10000label == '6')))
print(Imagelabel1.shape)
print(Imagelabel2.shape)
print(Imagelabel3.shape)

Imagelabel123 = np.hstack((Imagelabel1,Imagelabel2))
Imagelabel123 = np.hstack((Imagelabel123,Imagelabel3))
print(Imagelabel123.shape)
Imagelabel123 = Imagelabel123.T[:,0:1]
OneThreeSiximg = first10000img[Imagelabel123[:,0],:]
print(OneThreeSiximg.shape)


# In[153]:


meanonethreesix = np.mean(OneThreeSiximg,axis=0)
meanonethreesix = np.tile(meanonethreesix,(3173,1))
OneThreeSiximg = OneThreeSiximg - meanonethreesix 
ScatterMatrixOnethreeSix = np.matmul(OneThreeSiximg.T,OneThreeSiximg)
# print(K)
from scipy.linalg import eigh
eigenvaluesOnethreeSix, vectorsOnethreesix = eigh(ScatterMatrixOnethreeSix,eigvals=(0,783))
vectorsOnethreesix = vectorsOnethreesix.T


# In[154]:


ExtractTwo = 2
TopTwoEigenvector = vectorsOnethreesix[(784-ExtractTwo):784,:]
print(TopTwoEigenvector.shape)
twodimentiondata = np.matmul(OneThreeSiximg,TopTwoEigenvector.T)
print(twodimentiondata)
print(twodimentiondata[0,:])


# In[155]:


plt.scatter(twodimentiondata[0:1127,0],twodimentiondata[0:1127,1])
plt.scatter(twodimentiondata[1127:1127+1032,0],twodimentiondata[1127:1127+1032,1])
plt.scatter(twodimentiondata[1127+1032:1127+1032+1014,0],twodimentiondata[1127+1032:1127+1032+1014,1])
plt.savefig('Q4.jpg')
plt.close('all')


# In[156]:


# Q5
OMPfirst10000 = np.array(mnist['data'][0:10000,:])
import operator
print(OMPfirst10000.shape)
norm = np.zeros(10000)
for i in range(10000):
    norm[i] = np.linalg.norm(OMPfirst10000[i,:])
print("shape",norm.shape)
print("OMPfirst10000",OMPfirst10000.shape)
biUnitO=OMPfirst10000
for i in range(len(norm)):
    biUnitO[i,:] = OMPfirst10000[i,:]/norm[i]
print(OMPfirst10000)
test = np.array(mnist['data'][10000,:])
print(biUnitO.shape)
b=np.matmul(biUnitO,test.T)
print(b.shape)
r=test
sparsity = 5
base = np.zeros((sparsity,784))
def OMP(sparsity,b,biUnit,r,test,base):
    biUnit1 = biUnit
    img = np.zeros((sparsity,784))
    for i in range(0,sparsity,1):   
        b=np.matmul(biUnit1,r.T)
        index,value=max(enumerate(b),key=operator.itemgetter(1))
    #     print(index)
    #     print(Maxvalue)
        base[i,:] = biUnit1[index,:]
        base2=base[0:(i+1),:]
        scatter = np.matmul(base2,base2.T)
    #     print("scatter",scatter)
        inv=np.linalg.inv(scatter)
        cn = np.matmul(inv,np.matmul(base2,test.T))
        r=test-(np.matmul(base2.T,cn.T)).T
        biUnit1 = np.delete(biUnit1,index,0)
        img = np.matmul(base2.T,cn.T)
    return img
img5 = OMP(sparsity,b,biUnitO,r,test,base)
print(img5.shape)


# In[157]:


# print(img[0,:])
# print(img[1,:])
# print(img[2,:]==img[3,:])
plt.figure(figsize=(10,10))
print(base.shape)
for i in range(sparsity):
    plt.subplot(151 + i)
    
    plt.imshow(base[i,:].reshape(28,28),'gray')
    plt.axis('off')
plt.savefig('Q5.jpg')
plt.close('all')


# In[158]:


plt.imshow(test.reshape(28,28),'gray')


# In[159]:


# Q6
OMPfirst10000 = np.array(mnist['data'][0:10000,:])
import operator
print(OMPfirst10000.shape)
norm = np.zeros(10000)
for i in range(10000):
    norm[i] = np.linalg.norm(OMPfirst10000[i,:])
print("shape",norm.shape)
print("OMPfirst10000",OMPfirst10000.shape)
biUnitO=OMPfirst10000
for i in range(len(norm)):
    biUnitO[i,:] = OMPfirst10000[i,:]/norm[i]
print(OMPfirst10000)
test = np.array(mnist['data'][10001,:])
print(biUnitO.shape)
b=np.matmul(biUnitO,test.T)
print(b.shape)
base5 = np.zeros((5,784))
base10 = np.zeros((10,784))
base40 = np.zeros((40,784))
base200 = np.zeros((200,784))
r=test
img5=OMP(5,b,biUnitO,r,test,base5)
img10=OMP(10,b,biUnitO,r,test,base10)
img40=OMP(40,b,biUnitO,r,test,base40)
img200=OMP(200,b,biUnitO,r,test,base200)
plotrec = np.vstack((test,img5))
plotrec = np.vstack((plotrec,img10))
plotrec = np.vstack((plotrec,img40))
plotrec = np.vstack((plotrec,img200))
print(plotrec.shape)
plt.figure(figsize=(10,10))
for i in range(5):
    plt.subplot(151 + i)
    plt.imshow(plotrec[i,:].reshape(28,28),'gray')
    formatnumber = (('%.2f' % np.linalg.norm(plotrec[i,:]-test)))
    plt.title('L2-Norm='+formatnumber)
    plt.axis('off')
plt.savefig('Q6.jpg')
plt.close('all')


# In[160]:


totalImage = np.array(mnist['data'])
EightImagelabel = np.array(np.where(mnist['target'] == '8'))
print(EightImagelabel.shape)
EightImagelabel = EightImagelabel.T[:,0:1]
print(EightImagelabel.shape)
EightImagedatas = totalImage[EightImagelabel[:,0],:]


# In[161]:


# Q7-1
meanQ7 = np.mean(EightImagedatas,axis=0)
meanQ7 = np.tile(meanQ7,(6825,1))
EightImagedatas = EightImagedatas - meanQ7 
ScatterMatrix = np.matmul(EightImagedatas.T,EightImagedatas)
# print(K)
from scipy.linalg import eigh
eigenvalues, vectors1 = eigh(ScatterMatrix,eigvals=(0,783))
vectors1 = vectors1.T

ExtractNumber = 5
Top5eigenvectors = vectors1[(784-ExtractNumber):784,:]

EightImagedatasO = EightImagedatas + meanQ7
first8Image = EightImagedatasO[0,:]
print(first8Image.shape)
fianl_img8top5 = np.matmul(first8Image,np.matmul(Top5eigenvectors.T,Top5eigenvectors))
imgplotfirst8Top5 = plt.imshow(fianl_img8top5.reshape(28,28),plt.cm.gray)
plt.savefig('Q7-1.jpg')
plt.close('all')


# In[162]:


import numpy as np
from sklearn.decomposition import PCA
X = EightImagedatas
pca = PCA(n_components=5)
imgPCA = pca.fit(X)
print(imgPCA.components_[0,:].shape)
print(first8Image.shape)
fianl_img8top5 = np.matmul(first8Image,np.matmul(imgPCA.components_.T,imgPCA.components_))
plt.imshow(fianl_img8top5.reshape(28,28),'gray')


# In[163]:


# Q7-2
import numpy as np
totalImage = np.array(mnist['data'])
EightImagelabel = np.array(np.where(mnist['target'] == '8'))
print(EightImagelabel.shape)
EightImagelabel = EightImagelabel.T[:,0:1]
print(EightImagelabel.shape)
EightImagedatas = totalImage[EightImagelabel[:,0],:]
EightImagetrain = EightImagedatas[0:6824,:]
print(EightImagetrain.shape)
norm = np.zeros(len(EightImagetrain[:,0]))
for i in range(len(norm)):
    norm[i] = np.linalg.norm(EightImagetrain[i,:])
print("shape",norm.shape)
print("EightImagedatas",EightImagetrain.shape)
biUnitO=EightImagetrain
for i in range(len(norm)):
    biUnitO[i,:] = EightImagetrain[i,:]/norm[i]
print(EightImagetrain)
test = np.array(EightImagedatas[6824,:])
print(biUnitO.shape)
b=np.matmul(biUnitO,test.T)
print(b.shape)
r=test
baseQ72 = np.zeros((5,784))
img8Q2 = OMP(5,b,biUnitO,r,test,baseQ72)
print(img8Q2.shape)
plt.imshow(img8Q2.reshape(28,28),'gray')
plt.savefig('Q7-2.jpg')
plt.close('all')


# In[164]:


from sklearn.linear_model import OrthogonalMatchingPursuit as OMP

omp = OMP(n_nonzero_coefs=5, normalize=False)
omp.fit(biUnitO.T,test.T)
c = omp.predict(biUnitO.T) #  (n_samples, n_features) --> (n_samples,)

# reconstruction_image = np.matmul(base.T,c.T)

plt.subplot(111) # Three integers (nrows, ncols, index).
plt.imshow(c.reshape(28,28),'gray')
plt.axis('off')


# In[165]:


from sklearn import linear_model
print(EightImagetrain.shape )
clf = linear_model.Lasso(alpha=0.5)
clf.fit(EightImagetrain.T,test.T)
# print(test.shape)
img82=clf.predict(EightImagetrain.T)
# print(img82)
imgplotf= plt.imshow(img82.reshape(28,28),plt.cm.gray)
plt.savefig('Q7-3.jpg')
plt.close('all')


# In[ ]:




