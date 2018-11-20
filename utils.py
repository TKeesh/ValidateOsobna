import cv2
import numpy as np
from os.path import exists


global log_path

# logiranje u ./log.txt
toLog = True
# ispis u konzolu
toPrint = True

def log(s, toLog=toLog, toPrint=toPrint):
	global log_path
	if toLog:
		with open(log_path, 'a') as f:
			f.write(s + '\n')
	if toPrint:
		print (s)

def init_log(path):
	global log_path
	log_path = path
	if not exists(log_path): open(log_path, 'w').close()

def clear_log():
	global log_path
	open(log_path, 'w').close()


# Kut izmedju dvije tocke i y osi
def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


# Funkcija za normalizaciju luminosity-a, patterna (grb) na temlju slike
#	prima: 
#		sliku ili None
#		pattern (grb)
#	vraca: normalizirani pattern
def adjust_luma(img, pattern):
	if not img is None:
		img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
		minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(img_yuv[:,:,0])
	else:
		minVal, maxVal = 0, 255

	pattern_yuv = cv2.cvtColor(pattern, cv2.COLOR_BGR2YUV)
	y, u, v = cv2.split(pattern_yuv)
	cv2.normalize(y, y, minVal, maxVal, cv2.NORM_MINMAX)
	pattern = cv2.cvtColor(cv2.merge((y,u,v)), cv2.COLOR_YUV2BGR)

	return pattern


# Slicno kao gore samo sto normalizira sve kanale slike
#	radi za bilo koji prostor slika dubine 3
def adjust_all(img, pattern):
	if img != None:
		img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
		y0, y1, minLoc, maxLoc = cv2.minMaxLoc(img_yuv[:,:,0])
		u0, u1, minLoc, maxLoc = cv2.minMaxLoc(img_yuv[:,:,1])
		v0, v1, minLoc, maxLoc = cv2.minMaxLoc(img_yuv[:,:,2])
	else:
		y0, y1 = 0, 255
		u0, u1 = 0, 255
		v0, v1 = 0, 255

	pattern_yuv = cv2.cvtColor(pattern, cv2.COLOR_BGR2YUV)
	y, u, v = cv2.split(pattern_yuv)
	cv2.normalize(y, y, y0, y1, cv2.NORM_MINMAX)
	cv2.normalize(u, u, u0, u1, cv2.NORM_MINMAX)
	cv2.normalize(v, v, v0, v1, cv2.NORM_MINMAX)
	pattern = cv2.cvtColor(cv2.merge((y,u,v)), cv2.COLOR_YUV2BGR)

	return pattern


# Detekcija nosa
def detect_nose(img_main):
	height, width, channels = img_main.shape
	img_main = img_main[int(height*0.2):int(height*0.9), int(width*0.3):int(width*0.9)]
	img_gray = cv2.cvtColor(img_main, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(6,6))
	cl1 = clahe.apply(img_gray)

	face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_mcs_nose.xml')
	faces = face_cascade.detectMultiScale(cl1, 1.1, 4)

	faces = sorted(faces, key=lambda x: (x[2]*x[3]))

	# x0, y0, area = -1, -1, 2000
	# for (x,y,w,h) in faces:
	# 	if w*h < area:
	# 		area = w*h
	# 		x0, y0 = x+int(width*0.3)+int(w/2), y+int(height*0.2)+int(h/2)
	# 	cv2.rectangle(img_gray,(x,y),(x+w,y+h),(255,0,0),2)
	#cv2.circle(img_main, (x0,y0), 5, 255, -1)
	# cv2.imshow('nose', img_gray)

	x, y, w, h = faces[0]

	return x+int(width*0.3)+int(w/2), y+int(height*0.2)+int(h/2), w*h


# Detekcija desnog oka
#	nazalost nekad detektira lijevo oko
def detect_right_eye(img_main):
	height, width, channels = img_main.shape
	img_main = img_main[int(height*0.2):int(height*0.7), int(width*0.2):int(width*0.9)]
	face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_mcs_righteye.xml')
	faces = face_cascade.detectMultiScale(img_main, 1.1, 4)

	faces = sorted(faces, key=lambda x: (x[0], x[2]*x[3]))

	#x0, y0, area = -1, -1, 1500
	# for (x,y,w,h) in faces:
	# 	if w*h < area:
	# 		area = w*h
	# 		x0, y0 = x+int(width*0.2)+int(w/2), y+int(height*0.2)+int(h/2)
	# 	cv2.rectangle(img_main,(x,y),(x+w,y+h),(255,0,0),2)
	#cv2.circle(img_main, (x0,y0), 5, 255, -1)

	x, y, w, h = faces[-1]
	# cv2.rectangle(img_main,(x0,y0),(x0+w,y0+h),(255,0,0),2)
	# cv2.imshow('eye', img_main)

	return x+int(width*0.2)+int(w/2), y+int(height*0.2)+int(h/2), w*h


# Detekcija portreta
#	prima:
#		sliku
#		(x, y) - grba
#	vraca: poziciju centra portreta
def detect_face(img_main):
	height, width, channels = img_main.shape
	x_grb, y_grb = int(width * 0.4), 0

	# Na temelju pozicije grba (x, y) uzima podrucje gdje bi trebao biti portret 
	img_main = img_main[y_grb+40:int(height*0.97), x_grb-10:int(width*0.97)]
	#cv2.imshow('im', img_main)
	
	# Normalizira luminosity (0 - 255), moze pomoci u nekim slucajevima, za sada se cini da nije potrebno
	#img_main = adjust_luma(None, img_main)
	#img_gray = cv2.GaussianBlur(img_gray,(3,3),0)
	#clahe!?!

	# Detekcija portreta pomocu haarcascade naucenih znacajki - neovisno o orijentaciji, kontrastu, velicini...
	face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt.xml')
	faces = face_cascade.detectMultiScale(img_main, 1.1, 2)
	x0, y0, area = -1, -1, -1
	for (x,y,w,h) in faces:
		if w*h > area:
			area = w*h
			x0, y0 = x+(x_grb-10)+int(w/2), y+(y_grb+40)+int(h/2)
		# cv2.rectangle(img_main,(x,y),(x+w,y+h),(255,0,0),2)
	#cv2.imshow('faces', cl1)
	if len(faces) > 1:
		log( '      broj pronadjenih portreta: {0}; uzimam najveci kao relevantan'.format(str(len(faces))) )

	# Vraca najvece detektirano lice na ulaznom podrucju. Gotovo uvijek ce biti 0 ili 1 detektirano, ali za svaki slucaj
	return x0, y0


def detect_MRZ(img_main):
	h, w, c = img_main.shape
	rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 4)) #9,4
	sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 13)) #5,13

	img_gray = cv2.cvtColor(img_main, cv2.COLOR_BGR2GRAY)
	img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
	
	#equ_main = cv2.equalizeHist(img_gray)
	clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
	cl1 = clahe.apply(img_gray)

	blackhat = cv2.morphologyEx(cl1, cv2.MORPH_BLACKHAT, rectKernel)
	#cv2.imshow('blackhat', blackhat)

	gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
	gradX = np.absolute(gradX)
	(minVal, maxVal) = (np.min(gradX), np.max(gradX))
	gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
	#cv2.imshow('gradX', gradX)

	gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
	#cv2.imshow('gradx0', gradX)
	thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	#cv2.imshow('thresh1', thresh)

	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
	#cv2.imshow('thresh2', thresh)
	#threshX = cv2.erode(thresh, None, iterations=1)
	#cv2.imshow('threshX', threshX)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,2))
	thresh = cv2.erode(thresh, kernel, iterations=1)
	#cv2.imshow('thresh3', thresh)

	p = int(w * 0.05)
	thresh[:, 0:p] = 0
	#cv2.imshow('thresh4', thresh)
	thresh[:, w - p:] = 0
	#cv2.imshow('thresh', thresh)

	ret, x0, y0, w0, h0 = 0, 0, 0, 0, 0
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	for c in cnts:
		(x0, y0, w0, h0) = cv2.boundingRect(c)
		h_cpy = h0
		ar = w0 / float(h0)
		crWidth = w0 / float(w)

		if ar > 5 and crWidth > 0.60:
			rect = cv2.minAreaRect(c)
			om = rect[-1]
			if om*-1 > 45:
				om = 90 + rect[-1]
			else:
				om *= -1
			om = 1 - abs(om)*0.1
			#print (om)

			approx = cv2.approxPolyDP(c,5,True)
			approx = sorted(approx, key=lambda x: (x[0][0], x[0][1]))
			w_trans = abs(approx[1][0][0] - approx[-2][0][0])
			#print('w: ', w_trans, w0, w0-w_trans)
			fi = 1 - (w0-w_trans)*0.01
			#print ('fi: ', fi**2)

			pX = int((x0 + w0) * 0.03)
			pY = int((y0 + h0) * 0.03)
			(x0, y0) = (x0 - pX, y0 - pY)
			(w0, h0) = (w0 + (pX * 2), h0 + (pY * 2))

			#roi = img_main[y0:y0 + h0, x0:x0 + w0].copy()
			#cv2.rectangle(img_main, (x0, y0), (x0 + w0, y0 + h0), (0, 255, 0), 2)

			#cv2.rectangle(img_main, (x0, y0-int(3.35*h_cpy * om * fi)), (x0 + w0, y0 + h0), (0, 255, 0), 2)
			x0, y0, w0, h0 = x0, y0-int(3.35*h_cpy * om * fi), w0, h0

			ret = 1
			break

	#cv2.imshow('main', img_main)
	#try:cv2.imshow('roi', roi)
	#except: pass

	return ret, x0, y0, w0, h0