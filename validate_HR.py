from utils import *
import config_HR as cfg


# Detekcija grba na temelju ORB znacajki
#	prima: sliku
#	vraca:
#		(x, y) detektiranog grba
#		broj znacajki koje se poklapaju
def features_matching(img_main):
	height, width, channels = img_main.shape	

	# Podrucje slike za detekciju gdje bi trebao biti grb, dosta siroko
	img_main = img_main[int(height*cfg.grb_y0):int(height*cfg.grb_y1), int(width*cfg.grb_x0):int(width*cfg.grb_x1)]
	# sharpness = cv2.Laplacian(img_main, cv2.CV_64F).var()
	# print(sharpness)
	#cv2.imshow('grb', img_main)
	#img_main = cv2.medianBlur(img_main, 3)
	#img_main = cv2.GaussianBlur(img_main,(3,3),0)

	img_gray = cv2.cvtColor(img_main, cv2.COLOR_BGR2GRAY)
	#equ_main = cv2.equalizeHist(img_gray)
	clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
	cl1 = clahe.apply(img_gray)

	# Citanje patterna (grb). Prilagodjavanje luminosity-a na temelju sliku. Denoisanje
	img_grb = cv2.imread('./templates/grb_HR.jpg')

	img_grb = adjust_luma(img_main, img_grb)
	# sharpness = cv2.Laplacian(img_grb, cv2.CV_64F).var()
	# print(sharpness)
	#img_grb = adjust_all(img_main, img_grb)
	#img_grb = cv2.medianBlur(img_grb, 3)

	img_grb_gray = cv2.cvtColor(img_grb, cv2.COLOR_BGR2GRAY)
	#equ_grb = cv2.equalizeHist(img_grb_gray)
	#clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
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


# Detekcija dijelova lica
#	poziva se kada portret nije detektiran
def detect_face_parts(img_main):	
	height, width, channels = img_main.shape
	x_grb, y_grb = int(width * 0.4), 0
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


# Glavna funkcija za validaciju prednje strane osobne
#	prima: img_path
#	vraca:
#		0  - nije osobna
#		-1 - blurrana osobna
#		>0 - ok sobna, vrijednost = sharpness
def validate_front_HR(img_path):
	# Citanje slike
	img_main = cv2.imread(img_path)	
	h, w, c = img_main.shape
	r = 380 / w
	img_main = cv2.resize(img_main, (380, int(h * r)))
	h, w, c = img_main.shape

	# img_to_show = img_main.copy() # MOZE SE ZAKOMENTIRAT - samo u mainu
	# cv2.rectangle(img_to_show,(int(w*cfg.f_w),int(h*cfg.f_h)),(w,h),(255,0,0),2)
	# cv2.imshow('img', img_to_show) # MOZE SE ZAKOMENTIRAT - samo u mainu

	try:
		x_face, y_face = detect_face(img_main.copy())
	except:
		x_face, y_face = -1, -1

	if x_face < 0 or y_face < 0:
		log('   portret nije detektiran')
		try:
			x_face, y_face = detect_face_parts(img_main.copy())
		except:
			x_face, y_face = -1, -1

	if x_face < w * cfg.f_w or y_face < h * cfg.f_h:
		log('   portret van dozvoljene pozicije (x y): {0} {1}'.format(x_face, y_face))
		x_face, y_face = -1, -1

	if x_face < 0 or y_face < 0:
		# Detekcija pozicije graba. 
		# try-except blok osigurava crash features_matching funkcije. To je indikacija da detekcija grba nije uspijela. Isto se koristi za detekciju portreta.	
		try:
			x_grb, y_grb, score = features_matching(img_main.copy())
		except Exception as e:
			#print( "Error: %s" % e )
			log('   grb nije detektiran !!')
			log('   Exception: {0}'.format(e))
			return 0, 'grb nije detektiran'

		log('   grb detektiran (x y score): {0} {1} {2}'.format(str(x_grb), str(y_grb), str(score)))
		# cv2.circle(img_to_show, (x_grb,y_grb), 5, 255, -1) # MOZE SE ZAKOMENTIRAT - samo u mainu
		# cv2.imshow('img', img_to_show) # MOZE SE ZAKOMENTIRAT - samo u mainu

		# Hardkodirani thresholdi pozicije grba i broja dobro spojenih znacajki
		if (cfg.x_grb_l < x_grb < cfg.x_grb_h) and (cfg.y_grb_l < y_grb < cfg.y_grb_h) and (score > cfg.score):
			log('   grb unutar dozvoljene pozicije')
		else:
			log('   grb van dozvoljene pozicije: {0} {1} {2} !!'.format(x_grb, y_grb, score))
			return 0, 'grb van dozvoljene pozicije'
	else:
		log('   portret detektiran (x y): {0} {1}'.format(str(x_face), str(y_face)))
		# cv2.circle(img_to_show, (x_face,y_face), 5, 255, -1) # MOZE SE ZAKOMENTIRAT - samo u mainu
		# cv2.imshow('img', img_to_show) # MOZE SE ZAKOMENTIRAT - samo u mainu

	# Jednostavno racunanje sharpness slike. Hardkoridan threshold za provjeru.
	sharpness = cv2.Laplacian(img_main[int(h*0.05):int(h*0.95), int(w*0.05):int(w*0.95)], cv2.CV_64F).var()
	log('   sharpness: {0}'.format(str(sharpness)))		
	if sharpness > cfg.sharpness_FRONT:
		return sharpness, ''
	else:
		log('   previse blurana: {0} !!'.format(sharpness))
		return -1, 'previse blurana'	


# Glavna funkcija za validaciju straznje strane osobne
#	prima: img_path
#	vraca:
#		0  - nije osobna
#		-1 - blurrana osobna
#		>0 - ok sobna, vrijednost = sharpness
def validate_back_HR(img_path):
	# Citanje slike
	img_main = cv2.imread(img_path)	
	h, w, c = img_main.shape
	r = 380 / w
	img_main = cv2.resize(img_main, (380, int(h * r)))
	h, w, c = img_main.shape

	ret, x0, y0, w0, h0 = detect_MRZ(img_main.copy())

	if ret and (x0 > 0) and (y0 > 0) and (x0 + w0 < w) and (y0 + h0 < h):
		log('   MRZ detektirana (x y w h): {0} {1} {2} {3}'.format(x0, y0, w0, h0))
		sharpness = cv2.Laplacian(img_main[int(h*0.05):int(h*0.95), int(w*0.05):int(w*0.95)], cv2.CV_64F).var()
		if sharpness > cfg.sharpness_BACK:
			return sharpness, ''
		else:
			log('   previse blurana: {0} !!'.format(sharpness))
			return -1, 'previse blurana'
	else:
		log('   MRZ nije detektirana: {0} {1} {2} {3} {4} !!'.format(ret, x0, y0, w0, h0))
		return 0, 'MRZ nije detektirana'