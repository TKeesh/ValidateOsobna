import cv2
import numpy as np
from datetime import datetime # MOZE SE ZAKOMENTIRAT - samo u mainu
import os # MOZE SE ZAKOMENTIRAT - samo u mainu
import config


# Globalna varijabla slike - samo za debug
global img_to_show
global log_path


def log(s, toLog=config.toLog, toPrint=config.toPrint):
	global log_path
	if toLog:
		with open(log_path, 'a') as f:
			f.write(s + '\n')
	if toPrint:
		print (s)


def init_log(path):
	global log_path
	log_path = './log_' + path.strip('/').split('_')[1] + '.txt'
	open(log_path, 'w').close()


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


# Detekcija grba na temelju ORB znacajki
#	prima: sliku
#	vraca:
#		(x, y) detektiranog grba
#		broj znacajki koje se poklapaju
def features_matching(img_main):	
	height, width, channels = img_main.shape	

	# Podrucje slike za detekciju gdje bi trebao biti grb, dosta siroko
	img_main = img_main[int(height*config.grb_y0):int(height*config.grb_y1), int(width*config.grb_x0):int(width*config.grb_x1)]
	#img_main = cv2.medianBlur(img_main, 3)
	#img_main = cv2.GaussianBlur(img_main,(3,3),0)

	img_gray = cv2.cvtColor(img_main, cv2.COLOR_BGR2GRAY)
	equ_main = cv2.equalizeHist(img_gray)
	clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
	cl1 = clahe.apply(img_gray)

	# Citanje patterna (grb). Prilagodjavanje luminosity-a na temelju sliku. Denoisanje
	img_grb = cv2.imread('./grb.jpg')

	img_grb = adjust_luma(img_main, img_grb)
	#img_grb = adjust_all(img_main, img_grb)
	#img_grb = cv2.medianBlur(img_grb, 3)

	img_grb_gray = cv2.cvtColor(img_grb, cv2.COLOR_BGR2GRAY)
	equ_grb = cv2.equalizeHist(img_grb_gray)
	cl2 = clahe.apply(img_grb_gray)
	#cl2 = cv2.GaussianBlur(cl2,(3,3),0)
	#img_grb = cv2.GaussianBlur(img_grb,(7,7),0)

	#cv2.imshow('t1', cl1)
	#cv2.imshow('t2', cl2)

	# Racunanje znacajki nad izrezanom maskiranom slikom i maskiranom grbu
	orb = cv2.ORB_create()
	kp1, des1 = orb.detectAndCompute(cl1,None)
	kp2, des2 = orb.detectAndCompute(cl2,None)
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


# Detekcija nosa
def detect_nose(img_main):
	height, width, channels = img_main.shape
	img_gray = cv2.cvtColor(img_main[int(height*0.2):int(height*0.8), int(width*0.1):int(width*0.7)], cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(6,6))
	cl1 = clahe.apply(img_gray)

	face_cascade = cv2.CascadeClassifier('./haarcascade_mcs_nose.xml')
	faces = face_cascade.detectMultiScale(cl1, 1.1, 4)

	x0, y0, area = -1, -1, 2000
	for (x,y,w,h) in faces:
		if w*h < area:
			area = w*h
			x0, y0 = x+int(width*0.1)+int(w/2), y+int(height*0.2)+int(h/2)
		#cv2.rectangle(img_gray,(x,y),(x+w,y+h),(255,0,0),2)
	#cv2.circle(img_main, (x0,y0), 5, 255, -1)
	#cv2.imshow('nose', img_gray)

	return x0, y0, area


# Detekcija desnog oka
#	nazalost nekad detektira lijevo oko
def detect_right_eye(img_main):
	height, width, channels = img_main.shape
	img_main = img_main[int(height*0.2):int(height*0.7), int(width*0.2):int(width*0.9)]
	face_cascade = cv2.CascadeClassifier('./haarcascade_mcs_righteye.xml')
	faces = face_cascade.detectMultiScale(img_main, 1.1, 4)

	x0, y0, area = -1, -1, 1500
	for (x,y,w,h) in faces:
		print ('TEST: ', w*h)
		if w*h < area:
			area = w*h
			x0, y0 = x+int(width*0.2)+int(w/2), y+int(height*0.2)+int(h/2)
		cv2.rectangle(img_main,(x,y),(x+w,y+h),(255,0,0),2)
	#cv2.circle(img_main, (x0,y0), 5, 255, -1)
	#cv2.imshow('eye', img_main)

	return x0, y0, area


# Detekcija dijelova lica
#	poziva se kada portret nije detektiran
def detect_face_parts(img_main, x_grb, y_grb):
	height, width, channels = img_main.shape
	img_main = img_main[y_grb+40:int(height*0.97), x_grb-10:int(width*0.97)]

	try: x_nose, y_nose, area_nose = detect_nose(img_main.copy())
	except: x_nose, y_nose = -1, -1

	try: x_eye, y_eye, area_eye = detect_right_eye(img_main.copy())
	except: x_eye, y_eye = -1, -1

	#detect_mouth(img_main.copy())

	# cv2.circle(img_main, (x_nose,y_nose), 5, 255, -1)
	# cv2.circle(img_main, (x_eye,y_eye), 5, 255, -1)
	# cv2.imshow('test', img_main)

	if x_nose != -1 and x_eye != -1:
		return int((x_nose+x_eye) / 2)-7 + (x_grb-10), int((y_nose+y_eye) / 2)+7 + (y_grb+40)
	elif x_nose != -1:
		return x_nose + (x_grb-10), y_nose + (y_grb+40)
	elif x_eye != -1:
		return x_eye-10 + (x_grb-15), y_eye+15 + (y_grb+40)
	else:
		return -1, -1


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
	#img_gray = cv2.GaussianBlur(img_gray,(3,3),0)
	#clahe!?!

	# Detekcija portreta pomocu haarcascade naucenih znacajki - neovisno o orijentaciji, kontrastu, velicini...
	face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')
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


# Kut izmedju dvije tocke i y osi
def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


# Glavna funkcija za validaciju prednje strane osobne
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
	except Exception as e:
		print( "Error: %s" % e )
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
			x_face, y_face = -1, -1
			log('   portret nije detektiran')

		if x_face < 0 or y_face < 0:
			x_face, y_face = detect_face_parts(img_main.copy(), x_grb, y_grb)
		
		if x_face < 0 or y_face < 0:
			log('   niti jedan dio lica nije detektiran !!')
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


# Glavna funkcija za validaciju straznje strane osobne
#	prima: img_path
#	vraca:
#		0  - nije osobna
#		-1 - blurrana osobna
#		>0 - ok sobna, vrijednost = sharpness
def validate_back(img_path):
	global img_to_show
	# Citanje slike
	img_main = cv2.imread(img_path)	
	h, w, c = img_main.shape
	r = 380 / w
	img_main = cv2.resize(img_main, (385, int(h * r)))
	h, w, c = img_main.shape
	img_to_show = img_main.copy()

	return 1



# Main 
#	Cita ./test/ direktorij te validira slike u njemu
if __name__ == '__main__':
	global img_to_show

	# Mode: front or back
	mode = 'front'

	if mode == 'front':
		path = './test_front/'
	elif mode == 'back':
		path = './test_back/'
	else:
		print ('Wrong mode !!')
		quit()

	if config.toLog: init_log(path)
	cv2.namedWindow('img')

	imgs = os.listdir(path)
	imgs = [img for img in imgs if img.endswith('.jpg')]
	to_end = False
	for img_path in imgs:		
		img_path = path + img_path
		log('Validating img: ' + img_path)
		
		startTime = datetime.now()

		if mode == 'front':
			valid = validate_front(img_path)
		else:
			valid = validate_back(img_path)

		if valid > 0:
			cv2.putText(img_to_show, 'Valid', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA) 
			log('VALID: ' + str(valid))
		else: 
			cv2.putText(img_to_show, 'NOT valid', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA) 
			log('NOT VALID !!')

		cv2.imshow('img', img_to_show)
		log('execution time: ' + str(datetime.now() - startTime))
		
		if not to_end:
			inKey = cv2.waitKey(0) & 0xFF
			if inKey == ord('q'):
				break
			elif inKey == ord('e'):
				to_end = True
		log('')

	cv2.destroyAllWindows()

