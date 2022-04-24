import cv2 as cv
import numpy as np
import argparse
import math
import copy

W = 10     		# window size is WxW (kernel)
C_Thr = 0.4    	# threshold for coherency
LowThr = 30   		# threshold1 for orientation, it ranges from 0 to 180
HighThr = 60   	# threshold2 for orientation, it ranges from 0 to 180
ker_size = 31
kernel = np.zeros((ker_size, ker_size), np.uint8)
ancx = 15
ancy = 15

def first_filt(inIMG, filterSize):
	kernel = cv.getStructuringElement(cv.MORPH_RECT, filterSize)
	#inIMG = cv.cvtColor(inIMG, cv.COLOR_BGR2GRAY)
	
	ret = cv.Canny(inIMG, 50, 100)
	filtIMG = cv.morphologyEx(ret, cv.MORPH_CLOSE, kernel)	
	
	return filtIMG

def calcTenz(inputIMG, w):
	img = inputIMG.astype(np.float32)		#to float
   
	# J =  (J11 J12; J12 J22) - Tenzor
	imgDiffX = cv.Sobel(img, cv.CV_32F, 1, 0, 3)
	imgDiffY = cv.Sobel(img, cv.CV_32F, 0, 1, 3)
	imgDiffXY = cv.multiply(imgDiffX, imgDiffY)
    
	imgDiffXX = cv.multiply(imgDiffX, imgDiffX)
	imgDiffYY = cv.multiply(imgDiffY, imgDiffY)
	J11 = cv.boxFilter(imgDiffXX, cv.CV_32F, (w,w))
	J22 = cv.boxFilter(imgDiffYY, cv.CV_32F, (w,w))
	J12 = cv.boxFilter(imgDiffXY, cv.CV_32F, (w,w))


	# lambda1 = 0.5*(J11 + J22 + sqrt((J11-J22)^2 + 4*J12^2))
	# lambda2 = 0.5*(J11 + J22 - sqrt((J11-J22)^2 + 4*J12^2))
	tmp1 = J11 + J22
	tmp2 = J11 - J22
	tmp2 = cv.multiply(tmp2, tmp2)
	tmp3 = cv.multiply(J12, J12)
	tmp4 = np.sqrt(tmp2 + 4.0 * tmp3)
	lambda1 = 0.5*(tmp1 + tmp4)    		# biggest eigenvalue
	lambda2 = 0.5*(tmp1 - tmp4)    		# smallest eigenvalue
	# Coherency = (lambda1 - lambda2)/(lambda1 + lambda2)) - measure of anisotropism
	# Coherency is anisotropy degree (consistency of local orientation)
	imgCoherencyOut = cv.divide(lambda1 - lambda2, lambda1 + lambda2)
	# Coherency calculation (stop)
	# orientation angle calculation
	# tan(2*Alpha) = 2*J12/(J22 - J11)
	# Alpha = 0.5 atan2(2*J12/(J22 - J11))
	imgOrientationOut = cv.phase(J22 - J11, 2.0 * J12, angleInDegrees = True)
	imgOrientationOut = 0.5 * imgOrientationOut

	return imgCoherencyOut, imgOrientationOut
	
def get_kernel(angle, kern, ancx, ancy):
	
	cos = math.cos(angle * 3.1415 + 3.1415/2)
	sin = math.sin(angle * 3.1415 + 3.1415/2)
	
	kern[ancx][ancy] = 1;
	
	for i in range(1, kern.shape[0]):
		x = round(i * cos)
		y = round(i * sin)
		
		if x + ancx >= kern.shape[0] or y + ancy >= kern.shape[1] or ancx + x < 0 or ancy + y < 0:
			return kern
		
		kern[x + ancx][y + ancy] = 1
		kern[ancx - x][ancy - y] = 1
		

	return kern 

def paint_hole(kern, imgIn, i, j, color):

	for k in range(0, kern.shape[0]):
		for p in range(0, kern.shape[1]):
							
			im = i + k - ancx
			jm = j + p - ancy
											
			if im >= 0 and jm >= 0 and im < imgIn.shape[0] and jm < imgIn.shape[1]:
				if kern[k][p] != 0 : 
					imgIn[im][jm] = 0
	return imgIn				
				
	
def my_dilate(img, ker, orient, ax, ay, imgIn):

	res = copy.deepcopy(imgIn)

	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			
			if(orient.item(i, j) > 0 and img.item(i, j) != 0):
				
				ker = np.zeros((ker_size, ker_size), np.uint8)
				max_val = 0
				ker = get_kernel(orient.item(i, j), ker, ax, ay)
				#print(ker)
				
				ghor_up = 0
				ghor_down = 0
				vert_right = 0
				vert_left = 0
				
				param = 0
				
				for k in range(0, ker_size):
					for p in range(0, ker_size):
							
						im = i + k - ancx
						jm = j + p - ancy
											
						if im >= 0 and jm >= 0 and im < img.shape[0] and jm < img.shape[1]:
							if ker[k][p] * img[im][jm] == 0:
								param = 1
								break
					if param == 1:
						break
						
				if param == 0:
					continue				
				
				for k in range(ker.shape[0]):	#p = 0
					im = i + k - ancx
					jm = j - ancy
											
					if im >= 0 and jm >= 0 and im < img.shape[0] and jm < img.shape[1]:
						vert_left = vert_left + ker[k][0] * img[im][jm]
						
						if ker[k][0] * img[im][jm] != 0:
							color = imgIn[im][jm]
						
				for k in range(ker.shape[0]):	#p = ker.shape[1] - 1
					im = i + k - ancx
					jm = j - ancy + ker.shape[1] - 1
											
					if im >= 0 and jm >= 0 and im < img.shape[0] and jm < img.shape[1]:
						vert_right = vert_right + ker[k][ker.shape[1] - 1] * img[im][jm]
						#remember color
						if ker[k][ker.shape[1] - 1] * img[im][jm] != 0:
							color = imgIn[im][jm]
				
				for p in range(ker.shape[1]):	#k = 0
					im = i - ancx
					jm = j - ancy + p
											
					if im >= 0 and jm >= 0 and im < img.shape[0] and jm < img.shape[1]:
						ghor_up = ghor_up + ker[0][p] * img[im][jm]
						#remember color
						if ker[0][p] * img[im][jm] != 0:
							color = imgIn[im][jm]
				
				for p in range(ker.shape[1]):	#k = ker.shape[0] - 1
					im = i - ancx + ker.shape[0] - 1
					jm = j - ancy + p
											
					if im >= 0 and jm >= 0 and im < img.shape[0] and jm < img.shape[1]:
						ghor_down = ghor_down + ker[ker.shape[0] - 1][p] * img[im][jm]
						#remember color
						if ker[ker.shape[0] - 1][p] * img[im][jm] != 0:
							color = imgIn[im][jm]
								
				
				if(ghor_up + vert_right > 250 and ghor_down + vert_left > 250):
					res = paint_hole(ker, res, i, j, color)		
			
	return res

    
def main():
	imgIn = cv.imread('image.png', cv.IMREAD_GRAYSCALE)

	if imgIn is None:
		print('Could not open or find the image: {}'.format(args.input))
		exit(0)
		
	#first filt
	filt_img = first_filt(imgIn, (4, 4))
    
	#orientation 
	imgCoherency, imgOrientation = calcTenz(filt_img, W)
	_, imgCoherencyBin = cv.threshold(imgCoherency, C_Thr, 255, cv.THRESH_BINARY)
	_, imgOrientationBin = cv.threshold(imgOrientation, LowThr, HighThr, cv.THRESH_BINARY)
	imgBin = cv.bitwise_and(imgCoherencyBin, imgOrientationBin)
	
	#Normalize
	imgCoherency = cv.normalize(imgCoherency, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
	imgOrientation = cv.normalize(imgOrientation, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
	
	cv.imshow('filter1.png', filt_img)
	
	#morf filt in some direction
	result = my_dilate(filt_img, kernel, imgOrientation, ancx, ancy, imgIn)
	#kerne = get_kernel(0.1, kernel, ancx, ancy)
	#print(kerne)
	

	cv.imshow('Orientation.png', imgOrientation)
	cv.imshow('imgbin.png', imgBin)
	cv.imshow('coherency.png', imgCoherency)	
	cv.imshow('filter.png', filt_img)			
	cv.imshow('result.png', result)
	cv.waitKey(0)
	
if __name__ == "__main__":
	main()
