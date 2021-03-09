# -*- coding: utf-8 -*-
"""
Created on Fri May 29 14:47:58 2020

@author: Deo Haganta Depari
"""

import numpy as np
import cv2 as cv

#membaca citra RGB

rgb = cv.imread('Dataset setelah diproses/images_512x512/apple_1.jpg')
ycrcb = cv.cvtColor(rgb, cv.COLOR_BGR2YCrCb)
lower_blue = np.array([40,150,50])
upper_blue = np.array([200,255,150])

# Threshold the HSV image to get only blue colors
img_biner = cv.inRange(ycrcb, lower_blue, upper_blue)

#mendefinisikan structuring element 3x3 dan 5x5
se_3 = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
se_5 = cv.getStructuringElement(cv.MORPH_RECT,(5,5))

#opearsi morfologi
dst_dilate = cv.dilate(img_biner, se_3, iterations = 1)
dst_erosi = cv.erode(dst_dilate, se_3, iterations = 2)
dst_dilate2 = cv.dilate(dst_erosi, se_5, iterations = 2)
dst_erosi2 = cv.erode(dst_dilate2, se_5, iterations = 3)
dst_dilate3 = cv.dilate(dst_erosi2, se_3, iterations = 1)

res = cv.bitwise_and(rgb,rgb, mask= dst_dilate3)

cv.imshow('src', rgb)
cv.imshow('rgb', dst_dilate3)
cv.imshow('objek', res)
cv.waitKey(0)
cv.destroyAllWindows()