import sys
import json
from datetime import datetime

from utils import log, init_log, clear_log


# Glavna funkcija za pozivanje na serveru
#	prima:
#		img_path - lokalna putanju do slike
#		country  - {'CRO', } case insensitive
#		side     - {'F', 'FRONT', 'B', 'BACK', ...} case insensitive, bitno je samo prvo slovo
#	vraca:
#		0 - nije osobna
#		1 - ok osobna
def validate(img_path, country, side):
	startTime = datetime.now()
	log('Validating img: ' + img_path)

	from validate_CRO import validate_front_CRO, validate_back_CRO

	execution = {
		'cro' : (validate_front_CRO, validate_back_CRO),
	}

	front_back = 0 if side[0] == 'f' else 1 
	valid, reason = execution[country][front_back](img_path)

	if valid > 0:
		log('VALID: ' + str(valid))
	else: 
		log('NOT VALID !!')
	log('execution time: ' + str(datetime.now() - startTime))
	log('')

	json_data = {"Success": 1 if valid > 0 else 0, "ErrorReason": reason}	
	print (json.dumps(json_data, ensure_ascii=True, indent=None, sort_keys=True))

	return 1 if valid > 0 else 0


# Main 
#	Cita ./test/ direktorij te validira slike u njemu
if __name__ == '__main__':

	img_path = sys.argv[1]
	country = sys.argv[2].lower()
	side = sys.argv[3].lower()

	log_path = './logs/log_' + country.upper() + '_' + side + '.txt'
	init_log(log_path)

	try: 
		validate(img_path, country, side)
	except Exception as e:
		json_data = {"Success": 0, "ErrorReason": 'execution error'}	
		print (json.dumps(json_data, ensure_ascii=True, indent=None, sort_keys=True))
		log('   ERROR: {0}'.format(e))
		log('')