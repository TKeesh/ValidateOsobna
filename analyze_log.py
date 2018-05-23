import sys

if __name__ == '__main__':
	try:
		with open(sys.argv[1], 'r') as f:
			log = f.readlines()
	except:
		with open('./log.txt', 'r') as f:
			log = f.readlines()
	total, valid, not_valid = 0, 0, 0
	reasons = {}
	tmp_reson = ''
	for l in log:
		if 'VALID' in l:
			total += 1
			if 'NOT' in l: 
				not_valid += 1
				try: reasons[tmp_reson] += 1
				except: reasons[tmp_reson] = 1
			else: 
				valid += 1
		elif '!!' in l:
			tmp_reson = l[3:-3]
	print('Total: ', total)
	print('Valid: ', valid)
	print('Not valid: ', not_valid)
	print('Not valid reasons:')
	print(reasons)



