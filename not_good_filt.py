#!/usr/bin/python3

import cv2
import numpy as np

img = cv2.imread('image.png')

filterSize =(20, 20)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
	 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret = cv2.Canny(img, 50, 100)
closing = cv2.morphologyEx(ret, cv2.MORPH_CLOSE, kernel)	

answ = img	                      

def main():
	cv2.imshow('filt', closing)
	cv2.destroyAllWindows
	
	px = 50
	
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if closing.item(i, j) > 200 :
				if img.item(i, j) < 100 :
					px = img.item(i, j)
				else :	
					answ.itemset((i, j), px)
						
	cv2.imshow('try.png', answ)
	cv2.waitKey(0)
	
if __name__ == "__main__":
    main()
