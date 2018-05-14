import cv2
import numpy as np
from datetime import datetime # MOZE SE ZAKOMENTIRAT - samo u mainu
import os # MOZE SE ZAKOMENTIRAT - samo u mainu
import config


def log(s, toLog=config.toLog, toPrint=config.toPrint):
	if toLog:
		with open('./log.txt', 'a') as f:
			f.write(s + '\n')
	if toPrint:
		print (s)


def clear_log():
	open('./log.txt', 'w').close()


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
	img_main = img_main[int(height*config.grb_y0):int(height*config.grb_y1), int(width*config.grb_x0):int(width*config.grb_x1)]
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
	if len(faces) > 1:
		log( '      broj pronadjenih portreta: {0}; uzimam najveci kao relevantan'.format(str(len(faces))) )

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
#		>0 - ok sobna, vrijednost = sharpness
def validate_front(img_path):
	global img_to_show
	# Citanje slike
	img_main = cv2.imread(img_path)	
	h, w, c = img_main.shape
	r = 380 / w
	img_main = cv2.resize(img_main, (385, int(h * r)))
	h, w, c = img_main.shape
	img_to_show = img_main.copy()
	
	# Detekcija pozicije graba. 
	# try-except blok osigurava crash features_matching funkcije. To je indikacija da detekcija grba nije uspijela. Isto se koristi za detekciju portreta.
	try:
		x_grb, y_grb, score = features_matching(img_main.copy())
	except:
		log('   grb nije detektiran !!')
		return 0
	
	log('   grb detektiran (x y score): {0} {1} {2}'.format(str(x_grb), str(y_grb), str(score)))
	cv2.circle(img_to_show, (x_grb,y_grb), 5, 255, -1)
	
	# Hardkodirani thresholdi pozicije grba i broja dobro spojenih znacajki
	if (config.x_grb_l < x_grb < config.x_grb_h) and (config.y_grb_l < y_grb < config.y_grb_h) and (score > config.score):
		log('   grb unutar dozvoljene pozicije')
		
		# Ako je detekcija unutar threshola idemo na detekciju portreta. try-except kao i gore.
		try:
			x_face, y_face = detect_face(img_main.copy(), x_grb, y_grb)
		except:
			log('   portret nije detektiran !!')
			return 0
		
		log('   portret detektiran (x y): {0} {1}'.format(str(x_face), str(y_face)))
		cv2.circle(img_to_show, (x_face,y_face), 5, 255, -1)
		
		# Racunanje udaljenosti i kuta izmadju grba i portreta
		grb = np.array((x_grb,y_grb))
		face = np.array((x_face,y_face))
		dist = np.linalg.norm(grb - face)
		angle = angle_between(face, grb)
		log('   udaljenost grba i portreta: {0}'.format(str(dist)))
		log('   kut grba i portreta i y-osi: {0}'.format(str(angle)))

		# Hardkodirani thresholdi za provjeru udaljenosti i kuta grba i portreta
		if (config.dist_l < dist < config.dist_h) and (config.angle_l < angle < config.angle_h):
			log('   udaljenost i kut unutar dozvoljenih vrijednosti')									

			# Jednostavno racunanje sharpness slike. Hardkoridan threshold za provjeru.
			sharpness = cv2.Laplacian(img_main[int(h*0.05):int(h*0.95), int(w*0.05):int(w*0.95)], cv2.CV_64F).var()
			log('   sharpness: {0}'.format(str(sharpness)))		
			if sharpness > config.sharpness:
				return sharpness
			else:
				log('   previse blurana !!')
				return -1
		else:
			log('   udaljenost ili kut odstupaju !!')
			return 0
	else:
		log('   grb van dozvoljene pozicije !!')
		return 0


img_to_show = 0
# Main 
#	Cita ./test/ direktorij te validira slike u njemu
if __name__ == '__main__':
	global img_to_show

	if config.toLog: clear_log()
	cv2.namedWindow('img')

	imgs = os.listdir('./test/')
	imgs = [img for img in imgs if img.endswith('.jpg')]
	for img_path in imgs:		
		log('Validating img: ' + img_path)

		img_path = './test/' + img_path
		
		startTime = datetime.now()

		valid = validate_front(img_path)
		if valid:
			cv2.putText(img_to_show, 'Valid', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA) 
			log('VALID: ' + str(valid))
		else: 
			cv2.putText(img_to_show, 'NOT valid', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA) 
			log('NOT VALID !!')

		cv2.imshow('img', img_to_show)
		log('execution time: ' + str(datetime.now() - startTime))
		
		inKey = cv2.waitKey(0) & 0xFF
		if inKey == ord('q'):
			break
		log('')

	cv2.destroyAllWindows()

