import numpy as np
import random
import math
from decimal import Decimal
from time import time
import cv2
import matplotlib.pyplot as plt
from queue import PriorityQueue
from skimage.measure import compare_ssim
import argparse
import imutils
#import opencv
#from PIL import Image

def colorToBW(img):
	bw = cv2.imread('256x256bb.jpeg', cv2.IMREAD_GRAYSCALE)
	(rows, cols, channels) = img.shape
		
	for i in range(rows):
		for j in range(cols):

			k = img[i,j]

			(kr, kg, kb) = k 

			kGray = (0.21*kr) + (0.72*kg) + (0.07*kb)

			bw[i,j] = kGray
			
	return bw

def generateRandomColor():

	r = random.randint(0,255)
	g = random.randint(0,255)
	b = random.randint(0,255)

	return (r,g,b)


#takes an image and returns a list of the 5 representative colors
def k_means(img):

	

	#picking 5 colors at random

	center1 = generateRandomColor()
	center2 = generateRandomColor()
	center3 = generateRandomColor()
	center4 = generateRandomColor()
	center5 = generateRandomColor()

	colors = []

	(rows, cols, channels) = img.shape
	for i in range(rows):
		for j in range(cols):
			k = img[i,j]
			colors.append(k)
	diff1 = diff2 = diff3 = diff4 = diff5 = 0

	while True:

		list1 = []
		list2 = []
		list3 = []
		list4 = []
		list5 = []

		for c in colors:

			group, gc = representativeColor([center1, center2, center3, center4, center5], c)

			if group == 1:
				list1.append(c)
			elif group == 2:
				list2.append(c)
			elif group == 3:
				list3.append(c)
			elif group == 4:
				list4.append(c)
			elif group == 5:
				list5.append(c)

		r1 = b1 = g1 = 0
		for item1 in list1:
			(r,g,b) = item1
			r1 = r1 + r
			g1 = g1 + g
			b1 = b1 + b

		list1Len = len(list1)
		if (list1Len==0):
			center1 = generateRandomColor()
		else:
			center1New = (r1/list1Len, g1/list1Len, b1/list1Len)
			diff1 = colorDiff(center1, center1New)
			center1 = center1New

		r2 = b2 = g2 = 0
		for item2 in list2:
			(r,g,b) = item2
			r2 = r2 + r
			g2 = g2 + g
			b2 = b2 + b
		
		list2Len = len(list2)
		if (list2Len==0):
			center2 = generateRandomColor()
		else:
			center2New = (r2/list2Len, g2/list2Len, b2/list2Len)
			diff2 = colorDiff(center2, center2New)
			center2 = center2New

		r3 = b3 = g3 = 0
		for item3 in list3:
			(r,g,b) = item3
			r3 = r3 + r
			g3 = g3 + g
			b3 = b3 + b
		
		list3Len = len(list3)
		if (list3Len==0):
			center3 = generateRandomColor()
		else:
			center3New = (r3/list3Len, g3/list3Len, b3/list3Len)
			diff3 = colorDiff(center3, center3New)
			center3 = center3New

		r4 = b4 = g4 = 0
		for item4 in list4:
			(r,g,b) = item4
			r4 = r4 + r
			g4 = g4 + g
			b4 = b4 + b
		
		list4Len = len(list4)
		if (list4Len==0):
			center4 = generateRandomColor()
		else:
			center4New = (r4/list4Len, g4/list4Len, b4/list4Len)
			diff4 = colorDiff(center4, center4New)
			center4 = center4New

		r5 = b5 = g5 = 0
		for item5 in list5:
			(r,g,b) = item5
			r5 = r5 + r
			g5 = g5 + g
			b5 = b5 + b
		
		list5Len = len(list5) 
		if (list5Len==0):
			center5 = generateRandomColor()
		else:
			center5New = (r5/list5Len, g5/list5Len, b5/list5Len)
			diff5 = colorDiff(center5, center5New)
			center5 = center5New

		if diff1 < 2 and diff2 < 2 and diff3 < 2 and diff4 < 2 and diff5 < 2:

			break; 

	return [center1, center2, center3, center4, center5]


def colorDiff(c1, c2):

	(r1, g1, b1) = c1
	(r2, g2, b2) = c2

	return math.sqrt((r1-r2)**2 + (g1-g2)**2 + (b1-b2)**2)

#input:
#colors:the list of representative colors
#c: one color
#returns the representative color that c is closest to
def representativeColor(colors, c):

	min = -1
	mincolor = -1

	for i in range(len(colors)):
		diff = colorDiff(c, colors[i])
		if (diff < min) or (min == -1) :
			min = diff
			mincolor = i+1

	return mincolor, colors[mincolor - 1]

def getPatch(i,j, rows, cols):

	list = [] 
	list.append((i-1,j-1))
	list.append((i-1,j))
	list.append((i-1,j+1))
	list.append((i+1,j-1))
	list.append((i+1,j))
	list.append((i+1,j+1))
	list.append((i,j+1))
	list.append((i,j-1))
	list.append((i,j))
	if(i-1<0):
		if (i-1,j) in list:
			list.remove((i-1,j))
		if (i-1,j+1) in list:
			list.remove((i-1,j+1))
		if (i-1,j-1) in list:
			list.remove((i-1,j-1))
	if(i+1>=rows):
		if (i+1,j-1) in list:
			list.remove((i+1,j-1))
		if (i+1,j) in list:
			list.remove((i+1,j))
		if (i+1,j+1) in list:
			list.remove((i+1,j+1))
	if(j-1<0):
		if (i-1,j-1) in list:
			list.remove((i-1,j-1))
		if (i+1,j-1) in list:
			list.remove((i+1,j-1))
		if (i,j-1) in list:
			list.remove((i,j-1))
	if(j+1>=cols):
		if (i-1,j+1) in list:
			list.remove((i-1,j+1))
		if (i,j+1) in list:
			list.remove((i,j+1))
		if (i+1,j+1) in list:
			list.remove((i+1,j+1))
	return list

def getTrainingPatchList(bwImg):

	result = []

	(rows, cols) = bwImg.shape

	i=1
	while i < rows-1:

		j = 1
		while j < (cols/2)-1:
 
			result.append(getPatch(i,j,rows, cols))
			j += 1

		i+=1

	return result

def getPatchDiff(patch1, patch2, img):

	sum = 0

	for i in range(9):
		(a,b) = patch1[i]
		(c,d) = patch2[i]
		x = int(img[a,b])
		y = int(img[c,d])
		sum += abs(x - y)
	sum = int(sum)
	return math.sqrt(sum)


def get6TrainingColors(patch, trainingPatches, bwImage, img):
	
	#Traverse through each path in TrainingPatches
	result = []
	(rows, cols, channels) = img.shape
	q = PriorityQueue()

	#for i in range(6):

	#min = -1
	minPatch = []
	for patchT in trainingPatches:
		diff = getPatchDiff(patch, patchT, bwImage)
		q.put((diff, patchT))
			#if (diff < min or min == -1):
			#min = diff
			#minPatch = patchT
	i=0
	while i<6:
		(x, y)=q.get()
		result.append(y)
		i+=1
		#if minPatch is not []:
		#result.append(minPatch)
		#if minPatch in trainingPatches:
		#trainingPatches.remove(minPatch)

	resultColors = []
	for item in result:
		x = len(item)
		if x != 0:
			(a,b) = item[x-1]
			if((a<(rows)) and (b<(cols))):
				(r, g, b) = img[a,b]
				resultColors.append((r,g,b))

	return resultColors	

def getMajorityColor(colorsList):

	thisDict = {}
	maxColor = colorsList[0]

	for color in colorsList:
		x = thisDict.keys()
		if color not in x:
			thisDict[color] = 1
		else:
			thisDict[color] += 1

	max = 7 
	
	for color in thisDict:
		if thisDict[color] > max or max == 7:
			max = thisDict[color]
			maxColor = color
	
	#we need to make sure that this max value does not occur for more than one color

	num = 0
	
	for color in thisDict:
		if thisDict[color] == max:
			num += 1

	if num > 1:
		#There is a tie
		return colorsList[0]
	else:
		#no tie
		return maxColor

def detectDiff(img1, img2):
	bw1 = colorToBW(img1)
	bw2 = colorToBW(img2)
	(score, diff) = compare_ssim(bw1, bw2, full=True)
	diff = (diff * 255).astype("uint8")
	return score

#The basic agent could take a colored image as input
#it would return the recolorized version
def basic_agent(img):

	#converting this image to a black and white image
	bwImage = colorToBW(img)

	#getting a list of the representative colors
	repColors = k_means(img)

	(rows, cols, channels) = img.shape
	for i in range(rows):
		for j in range(int(cols/2)):
			extra, newcolor = representativeColor(repColors, img[i,j])
			img[i,j] = newcolor
	trainingPatches = getTrainingPatchList(bwImage)
	print("Starting right side")
	i = 1
	while i < rows-1:
		j = int((cols/2) )
		while j < cols-1:
			testList = getPatch(i,j,rows,cols)
			trainingOutput = get6TrainingColors(testList, trainingPatches, bwImage, img)
			finalColor = getMajorityColor(trainingOutput)
			(r, g, b) = finalColor
			print(finalColor)
			r = int(r)
			g = int(g)
			b = int(b)
			img[i,j] = (r, g, b)
			j += 1
		i += 1	
	return img


#main

image = cv2.imread('256x256bb.jpeg')
b,g,r = cv2.split(image)
image1 = cv2.merge([r,g,b])
#plt.imshow(image1)
#plt.show()

#bwimage = colorToBW(image1)
#plt.imshow(bwimage, cmap="gray")
#plt.show()

result = basic_agent(image1)
plt.imshow(result)
plt.show()

img2 = cv2.imread('11.png')
(r, c, ch) = image1.shape
img2 = cv2.resize(img2, (r, c))   
b,g,r = cv2.split(img2)
img2 = cv2.merge([r,g,b])
score = detectDiff(image1, img2)
print(score)



