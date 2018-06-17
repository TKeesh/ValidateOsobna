from datetime import datetime
startTime = datetime.now()
import cv2
print('OpenCV import time: ' + str(datetime.now() - startTime) + '\n')
import os
from datetime import datetime

from validate import *


country = 'CRO'
# Side: front or back
side = 'front'


# Main za local test
#	Cita ./test/ direktorij te validira slike u njemu
if __name__ == '__main__':

	path = './test_' + side + '/'

	log_path = './logs/log_' + country.upper() + '_' + side + '.txt'
	init_log(log_path)
	clear_log()

	cv2.namedWindow('img')

	imgs = os.listdir(path)
	imgs = [img for img in imgs if img.lower().endswith('.jpg')]
	to_end = False
	for img_path in imgs:		
		img_path = path + img_path
		
		startTime = datetime.now()

		valid = validate(img_path, 'cro', side)
		
		if not to_end or not valid:
			inKey = cv2.waitKey(0) & 0xFF
			if inKey == ord('q'):
				break
			elif inKey == ord('e'):
				to_end = True

	cv2.destroyAllWindows()