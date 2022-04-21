import cv2 as cv
import numpy as np
import argparse
import math

W = 10     		# window size is WxW (kernel)
C_Thr = 0.43    	# threshold for coherency
LowThr = 30   		# threshold1 for orientation, it ranges from 0 to 180
HighThr = 60   	# threshold2 for orientation, it ranges from 0 to 180
line = 15

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
	
def paint_lines(filtIMG, leng, orientationIMG, inIMG):

	for i in range(filtIMG.shape[0]):
		for j in range(filtIMG.shape[1]):
			if(filtIMG.item(i, j) > 0):
				#print(imgOrientation.item(i,j))
				start_point = (j, i)
				end_point = (j, i)
				cos = math.cos(orientationIMG.item(i, j) * 3.1415 + 3.1415/2)
				sin = math.sin(orientationIMG.item(i, j) * 3.1415 + 3.1415/2)
			
				end_point = (int(line * sin) + j, int(line * cos) + i)
			
				color = (0, 0, 0)
				thickness = 1
				if j + line < filtIMG.shape[1] and i + line < filtIMG.shape[0] : 
					inIMG = cv.line(inIMG, start_point, end_point, color, thickness)
					
	return inIMG
	
    
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
	
	#draw lines
	result = paint_lines(filt_img, line, imgOrientation, imgIn)

	#cv.imshow('result.png', np.uint8(0.5*(imgIn + imgBin)))
	cv.imshow('Orientation.png', imgOrientation)
	cv.imshow('imgbin.png', imgBin)
	cv.imshow('coherency.png', imgCoherency)	
	cv.imshow('filter.png', filt_img)			
	cv.imshow('result.png', result)
	cv.waitKey(0)
	
if __name__ == "__main__":
	main()
