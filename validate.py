import cv2
import numpy as np
from datetime import datetime # MOZE SE ZAKOMENTIRAT - samo u mainu
import os # MOZE SE ZAKOMENTIRAT - samo u mainu


# Funkcija za normalizaciju luminosity-a, patterna (grb) na temlju slike
#	prima: 
#		sliku ili None
#		pattern (grb)
#	vraca: normalizirani pattern
def adjust_luma(img, pattern):
	if len(img):
		img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
		minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(img_yuv[:,:,0])
	else:
		minVal, maxVal = 0, 255
	#minValU, maxValU, minLocU, maxLocU = cv2.minMaxLoc(img_yuv[:,:,1]) # za normalizaciju U komponente
	#minValV, maxValV, minLocV, maxLocV = cv2.minMaxLoc(img_yuv[:,:,2]) # za normalizaciju V komponente
	pattern_yuv = cv2.cvtColor(pattern, cv2.COLOR_BGR2YUV)
	y, u, v = cv2.split(pattern_yuv)
	cv2.normalize(y, y, minVal, maxVal, cv2.NORM_MINMAX)
	#cv2.normalize(u, u, minValU, maxValU, cv2.NORM_MINMAX) # za normalizaciju U komponente
	#cv2.normalize(v, v, minValV, maxValV, cv2.NORM_MINMAX) # za normalizaciju V komponente
	pattern = cv2.cvtColor(cv2.merge((y,u,v)), cv2.COLOR_YUV2BGR)
	return pattern


# Detekcija grba na temelju ORB znacajki
#	prima: sliku
#	vraca:
#		(x, y) detektiranog grba
#		broj znacajki koje se poklapaju
def features_matching(img_main):
	lower_red = np.array([80,70,50])
	upper_red = np.array([100,255,255])
	
	height, width, channels = img_main.shape	
	
	# Podrucje slike za detekciju gdje bi trebao biti grb, dosta siroko
	img_main = img_main[0:int(height*0.4), int(width*0.4):int(width*0.85)]
	#img_main = cv2.medianBlur(img_main, 3)

	# Detekcija crvene boje preko HSV prostora boja. img_main_inv jest maskirana slika - sadrzi sve crno osim onog sto je crveno
	img_main_inv = cv2.bitwise_not(img_main)
	img_main_inv = cv2.cvtColor(img_main_inv, cv2.COLOR_BGR2HSV)
	img_main_mask = cv2.inRange(img_main_inv, lower_red, upper_red)
	img_main_inv = cv2.bitwise_and(img_main, img_main, mask=img_main_mask)

	# Citanje patterna (grb). Prilagodjavanje luminosity-a na temelju sliku. Denoisanje
	img_grb = cv2.imread('./grb.jpg')
	img_grb = adjust_luma(img_main, img_grb)
	img_grb = cv2.medianBlur(img_grb, 3)

	# Detekcija crvene boje na slici grba, isto kao i gore, kako bi znacajke bile preciznije
	img_grb_inv = cv2.bitwise_not(img_grb)
	img_grb_inv = cv2.cvtColor(img_grb_inv, cv2.COLOR_BGR2HSV)
	img_grb_mask = cv2.inRange(img_grb_inv, lower_red, upper_red)
	img_grb_inv = cv2.bitwise_and(img_grb, img_grb, mask=img_grb_mask)

	# Racunanje znacajki nad izrezanom maskiranom slikom i maskiranom grbu
	orb = cv2.ORB_create()
	kp1, des1 = orb.detectAndCompute(img_main_inv,None)
	kp2, des2 = orb.detectAndCompute(img_grb_inv,None)
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des1,des2)

	# Sortiranje znacajki koje medjusobno odgovaraju po tocnosti (prvih 10)
	matches = sorted(matches, key = lambda x:x.distance)
	list_kp1 = [kp1[mat.queryIdx].pt for mat in matches[:10]] 
	list_kp2 = [kp2[mat.trainIdx].pt for mat in matches[:10]]
	
	# Racunanje srednje pozicije najtocnijih detektiranih znacajki
	x_sum, y_sum = 0, 0
	for i in range(len(list_kp1)):		
		x, y = int(list_kp1[i][0]), int(list_kp1[i][1])
		x_sum += x
		y_sum += y
	x_sum /= len(list_kp1)
	y_sum /= len(list_kp1)
	x_sum, y_sum = int(x_sum), int(y_sum)	
	
	return x_sum+int(width*0.4), y_sum, len(matches)


# Detekcija portreta
#	prima:
#		sliku
#		(x, y) - grba
#	vraca: poziciju centra portreta
def detect_face(img_main, x_grb, y_grb):
	height, width, channels = img_main.shape

	# Na temelju pozicije grba (x, y) uzima podrucje gdje bi trebao biti portret 
	img_main = img_main[y_grb+40:int(height*0.97), x_grb-10:int(width*0.97)]
	
	# Normalizira luminosity (0 - 255), moze pomoci u nekim slucajevima, za sada se cini da nije potrebno
	#img_main = adjust_luma(None, img_main)
	
	img_gray = cv2.cvtColor(img_main, cv2.COLOR_BGR2GRAY)
	
	# Moze posluziti za denoise, ali uglavnom nije potrebno
	#cl1 = cv2.medianBlur(cl1, 3)

	# CLAHE filter - normalizacija histograma na temelju kontrasta, otklanjanje odsjaja (da ne bi blic zasvijetlio portret)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	cl1 = clahe.apply(img_gray)	

	# Detekcija portreta pomocu haarcascade naucenih znacajki - neovisno o orijentaciji, kontrastu, velicini...
	face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface.xml')
	faces = face_cascade.detectMultiScale(cl1, 1.3, 5)
	x, y, area = -1, -1, -1
	for (x,y,w,h) in faces:
		if w*h > area:
			area = w*h
			x, y = x+(x_grb-10)+int(w/2), y+(y_grb+40)+int(h/2)
		cv2.rectangle(cl1,(x,y),(x+w,y+h),(255,0,0),2)
	#cv2.imshow('faces', cl1)
	print('faces len: ', len(faces))

	# Vraca najvece detektirano lice na ulaznom podrucju. Gotovo uvijek ce biti 0 ili 1 detektirano, ali za svaki slucaj
	return x, y


# Kut izmedju dvije tocke i y osi
def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


# Glavna funkcija
#	prima: img_path
#	vraca:
#		0  - nije osobna
#		-1 - blurrana osobna
#		>0 - ok sobna, vrijednost = blurriness
def validate_front(img_path):
	# Citanje slike
	img_main = cv2.imread(img_path)
	h, w, c = img_main.shape
	
	# Detekcija pozicije graba. 
	# try-except blok osigurava crash features_matching funkcije. To je indikacija da detekcija grba nije uspijela. Isto se koristi za detekciju portreta.
	try:
		x_grb, y_grb, score = features_matching(img_main.copy())
	except:
		print('NE0!!!')
		return 0
	
	print ('grb (x, y, score): ', x_grb, y_grb, score)
	# Hardkodirani thresholdi pozicije grba i broja dobro spojenih znacajki
	if (205 < x_grb < 275) and (5 < y_grb < 65) and (score > 5):
		print('OK1')
		
		# Ako je detekcija unutar threshola idemo na detekciju portreta. try-except kao i gore.
		try:
			x_face, y_face = detect_face(img_main.copy(), x_grb, y_grb)
		except:
			print('NE2o!!!')
			return 0
		print ('face (x, y): ', x_face, y_face)
		
		# Racunanje udaljenosti i kuta izmadju grba i portreta
		grb = np.array((x_grb,y_grb))
		face = np.array((x_face,y_face))
		dist = np.linalg.norm(grb - face)
		angle = angle_between(face, grb)
		print ('distance (grb - face): ', dist)
		print ('angle (face - grb): ', angle)

		# Hardkodirani thresholdi za provjeru udaljenosti i kuta grba i portreta
		if (115 < dist < 155) and (0 < angle < 40):
			print ('OK2')
			# Prikaz detektiranih podrucja: (x, y) - grba, (x, y) - portreta
			cv2.circle(img_main, (x_grb,y_grb), 5, 255, -1)
			cv2.circle(img_main, (x_face,y_face), 5, 255, -1)
			cv2.imshow('MAIN', img_main)
			
			# Jednostavno racunanje blurrinessa slike. Hardkoridan threshold za provjeru.
			blurriness = cv2.Laplacian(img_main[int(h*0.05):int(h*0.95), int(w*0.05):int(w*0.95)], cv2.CV_64F).var()
			print('blurriness: ', blurriness)		
			if blurriness > 80:
				return blurriness
			else:
				print('BLURRED!!!')
				return -1
		else:
			print('NE2!!!')
			return 0
	else:
		print('NE1!!!')
		return 0


# Main 
#	Cita ./test/ direktorij te validira slike u njemu
if __name__ == '__main__':
	imgs = os.listdir('./test/')
	imgs = [img for img in imgs if img.endswith('.jpg')]
	cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
	for img_path in imgs:
		print (img_path)
		startTime = datetime.now()
		print('VALID: ', validate_front(img_path))
		print (datetime.now() - startTime)
		inKey = cv2.waitKey(0) & 0xFF
		if inKey == ord('q'):
			break
	cv2.destroyAllWindows()

