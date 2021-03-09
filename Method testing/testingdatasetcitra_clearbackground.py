# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:39:47 2020

@author: Deo Haganta Depari
"""

import numpy as np
import cv2 as cv
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import sys

for x in range(1,3):
    if x==1:
        a="log_nri_uniformmethod2.txt"
        b="nri_uniform"
    else:
        a="log_rormethod2.txt"
        b="ror"
    stdoutOrigin=sys.stdout 
    sys.stdout = open(a, "w")
    
    for i in range(1,17):
        for j in range(26,51):
            #Settings for LBP
            print("-----------")
            print("Radius=",j)
            radius=j
            n_points=8*radius
            METHOD=b #uniform,nri_uniform, #default, ror
            plt.rcParams['font.size']=9
            
            #membaca gambar dan lakukan konversi ke grayscale, lakukan feature extraction dengan LBP
            data1_lbp = local_binary_pattern(cv.imread('Dataset setelah diproses/img_clearbackground/apple_1.png', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
            data2_lbp = local_binary_pattern(cv.imread('Dataset setelah diproses/img_clearbackground/apple_2.png', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
            data3_lbp = local_binary_pattern(cv.imread('Dataset setelah diproses/img_clearbackground/apple_3.png', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
            data4_lbp = local_binary_pattern(cv.imread('Dataset setelah diproses/img_clearbackground/apple_4.png', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
            data5_lbp = local_binary_pattern(cv.imread('Dataset setelah diproses/img_clearbackground/apple_5.png', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
            data6_lbp = local_binary_pattern(cv.imread('Dataset setelah diproses/img_clearbackground/apple_6.png', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
            data7_lbp = local_binary_pattern(cv.imread('Dataset setelah diproses/img_clearbackground/apple_7.png', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
            data8_lbp = local_binary_pattern(cv.imread('Dataset setelah diproses/img_clearbackground/apple_8.png', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
            data9_lbp = local_binary_pattern(cv.imread('Dataset setelah diproses/img_clearbackground/apple_9.png', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
            data10_lbp = local_binary_pattern(cv.imread('Dataset setelah diproses/img_clearbackground/apple_10.png', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
            
            data11_lbp = local_binary_pattern(cv.imread('Dataset setelah diproses/img_clearbackground/apple_11.png', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
            data12_lbp = local_binary_pattern(cv.imread('Dataset setelah diproses/img_clearbackground/apple_12.png', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
            data13_lbp = local_binary_pattern(cv.imread('Dataset setelah diproses/img_clearbackground/apple_13.png', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
            data14_lbp = local_binary_pattern(cv.imread('Dataset setelah diproses/img_clearbackground/apple_14.png', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
            data15_lbp = local_binary_pattern(cv.imread('Dataset setelah diproses/img_clearbackground/apple_15.png', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
            data16_lbp = local_binary_pattern(cv.imread('Dataset setelah diproses/img_clearbackground/apple_16.png', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
            data17_lbp = local_binary_pattern(cv.imread('Dataset setelah diproses/img_clearbackground/apple_17.png', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
            data18_lbp = local_binary_pattern(cv.imread('Dataset setelah diproses/img_clearbackground/apple_18.png', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
            data19_lbp = local_binary_pattern(cv.imread('Dataset setelah diproses/img_clearbackground/apple_19.png', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)
            data20_lbp = local_binary_pattern(cv.imread('Dataset setelah diproses/img_clearbackground/apple_20.png', cv.IMREAD_GRAYSCALE), n_points, radius, METHOD)    
            
            #mendapatkan histogram dari LBP untuk dataset
            data1_lbp_hist,bins=np.histogram(data1_lbp.ravel(),256,[0,256])
            data2_lbp_hist,bins=np.histogram(data2_lbp.ravel(),256,[0,256])
            data3_lbp_hist,bins=np.histogram(data3_lbp.ravel(),256,[0,256])
            data4_lbp_hist,bins=np.histogram(data4_lbp.ravel(),256,[0,256])
            data5_lbp_hist,bins=np.histogram(data5_lbp.ravel(),256,[0,256])
            data6_lbp_hist,bins=np.histogram(data6_lbp.ravel(),256,[0,256])
            data7_lbp_hist,bins=np.histogram(data7_lbp.ravel(),256,[0,256])
            data8_lbp_hist,bins=np.histogram(data8_lbp.ravel(),256,[0,256])
            data9_lbp_hist,bins=np.histogram(data9_lbp.ravel(),256,[0,256])
            data10_lbp_hist,bins=np.histogram(data10_lbp.ravel(),256,[0,256])
            
            data11_lbp_hist,bins=np.histogram(data11_lbp.ravel(),256,[0,256])
            data12_lbp_hist,bins=np.histogram(data12_lbp.ravel(),256,[0,256])
            data13_lbp_hist,bins=np.histogram(data13_lbp.ravel(),256,[0,256])
            data14_lbp_hist,bins=np.histogram(data14_lbp.ravel(),256,[0,256])
            data15_lbp_hist,bins=np.histogram(data15_lbp.ravel(),256,[0,256])
            data16_lbp_hist,bins=np.histogram(data16_lbp.ravel(),256,[0,256])
            data17_lbp_hist,bins=np.histogram(data17_lbp.ravel(),256,[0,256])
            data18_lbp_hist,bins=np.histogram(data18_lbp.ravel(),256,[0,256])
            data19_lbp_hist,bins=np.histogram(data19_lbp.ravel(),256,[0,256])
            data20_lbp_hist,bins=np.histogram(data20_lbp.ravel(),256,[0,256])
            
            #ubah vektor ke matriks dan lakukan transpose matriks untuk dataset
            data1_lbp_hist=np.transpose(data1_lbp_hist[0:18,np.newaxis])
            data2_lbp_hist=np.transpose(data2_lbp_hist[0:18,np.newaxis])
            data3_lbp_hist=np.transpose(data3_lbp_hist[0:18,np.newaxis])
            data4_lbp_hist=np.transpose(data4_lbp_hist[0:18,np.newaxis])
            data5_lbp_hist=np.transpose(data5_lbp_hist[0:18,np.newaxis])
            data6_lbp_hist=np.transpose(data6_lbp_hist[0:18,np.newaxis])
            data7_lbp_hist=np.transpose(data7_lbp_hist[0:18,np.newaxis])
            data8_lbp_hist=np.transpose(data8_lbp_hist[0:18,np.newaxis])
            data9_lbp_hist=np.transpose(data9_lbp_hist[0:18,np.newaxis])
            data10_lbp_hist=np.transpose(data10_lbp_hist[0:18,np.newaxis])
            
            data11_lbp_hist=np.transpose(data11_lbp_hist[0:18,np.newaxis])
            data12_lbp_hist=np.transpose(data12_lbp_hist[0:18,np.newaxis])
            data13_lbp_hist=np.transpose(data13_lbp_hist[0:18,np.newaxis])
            data14_lbp_hist=np.transpose(data14_lbp_hist[0:18,np.newaxis])
            data15_lbp_hist=np.transpose(data15_lbp_hist[0:18,np.newaxis])
            data16_lbp_hist=np.transpose(data16_lbp_hist[0:18,np.newaxis])
            data17_lbp_hist=np.transpose(data17_lbp_hist[0:18,np.newaxis])
            data18_lbp_hist=np.transpose(data18_lbp_hist[0:18,np.newaxis])
            data19_lbp_hist=np.transpose(data19_lbp_hist[0:18,np.newaxis])
            data20_lbp_hist=np.transpose(data20_lbp_hist[0:18,np.newaxis])
            
            # gabungkan data citra menjadi satu matriks data training
            trainData=np.concatenate((
            data1_lbp_hist,data2_lbp_hist,data3_lbp_hist,data4_lbp_hist,
            data5_lbp_hist,data6_lbp_hist,data7_lbp_hist,data8_lbp_hist,
            data9_lbp_hist,data10_lbp_hist,data11_lbp_hist,data12_lbp_hist,
            data13_lbp_hist,data14_lbp_hist,data15_lbp_hist,data16_lbp_hist,
            ),axis=0).astype(np.float32)
            
            
            trainTest=np.concatenate((data17_lbp_hist,data18_lbp_hist,data19_lbp_hist,data20_lbp_hist),axis=0).astype(np.float32)
            
            
            print("Neighbors Check=",i)
            
            
            #Target
            responses1=np.array([3,3,2,1,1,2,2,1,1,2,1,1,1,1,1,2]).astype(np.float32)
            
            #kNN
            knn=KNeighborsClassifier(n_neighbors=i) #define K=3
            knn.fit(trainData,responses1)
            res = knn.predict(trainTest)
            print("Res=",res)
            
            #penambahan library
            from sklearn.model_selection import cross_val_score
            
            #Data tidak dipisah ke dalam data training dan data testing, tapi digabungkan keselurahan data
            Dataset=np.concatenate((
                data1_lbp_hist,data2_lbp_hist,data3_lbp_hist,data4_lbp_hist,
                data5_lbp_hist,data6_lbp_hist,data7_lbp_hist,data8_lbp_hist,
                data9_lbp_hist,data10_lbp_hist,data11_lbp_hist,data12_lbp_hist,
                data13_lbp_hist,data14_lbp_hist,data15_lbp_hist,data16_lbp_hist,
                data17_lbp_hist,data18_lbp_hist,data19_lbp_hist,data20_lbp_hist
                ),axis=0).astype(np.float32)
            
            #karena dataset tidak dipisah ke dalam data training dan data testing,
            #maka variable responses dibuat seperti ini
            responses2 = np.array([3,3,2,1,1,2,2,1,1,2,1,1,1,1,1,2,3,3,2,3]).astype(np.float32)
            
            #klasifikasi kNN dengan k=3, dan hitung score dengan cross validation
            #menggunakan perhitungan akurasi dengan parameter cross validation =5
            
            knn=KNeighborsClassifier(n_neighbors=i)
            print("KFOLD=",5)
            score=cross_val_score(knn,Dataset,responses2,cv=5,scoring='accuracy')
            print("Score=",score.mean())
            print("-----------")
            
    sys.stdout.close()
    sys.stdout=stdoutOrigin