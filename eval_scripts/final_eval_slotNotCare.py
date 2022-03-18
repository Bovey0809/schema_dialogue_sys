import json
import pickle as pkl
import os


def resolv_schema(raw, slots_categorical, slots_api, intents_api):
	resolved = {}
	resolved['service_name'] = raw['service_name']
	resolved['description'] = raw['description']
	resolved['slots'] = {}
	for item in raw['slots']:
		slots_categorical[item['name']] = item['is_categorical']
		if item['name'] not in slots_api:
			slots_api[item['name']] = []
		slots_api[item['name']].append(raw['service_name'])
		resolved['slots'][item['name']] = item
		numVals = len(item['possible_values'])

	resolved['intents'] = {}
	for item in raw['intents']:
		if item['name'] not in intents_api:
			intents_api[item['name']] = []
		intents_api[item['name']].append(raw['service_name'])
		resolved['intents'][item['name']] = item

	return resolved

def resolv_train_dial(path):
	file_list = os.listdir(path)
	api2dial_idx = {}
	slot2dial_idx = {}
	for file in file_list:
		if 'dialogue' in file:
			# print(path+file)
			fp = open(path + file, 'r')
			_data = json.loads(fp.read())
			for x in _data:
				if x['services'][0] not in api2dial_idx:
					api2dial_idx[x['services'][0]] = []
			if file not in api2dial_idx[x['services'][0]]:
				api2dial_idx[x['services'][0]].append(file)

	return api2dial_idx

def load_shemas(path):

	slots_categorical = {}
	slots_api = {}
	intents_api = {}

	fp = open(path, 'r')
	data = json.loads(fp.read())
	print('#number of schemas:', len(data))
	fp.close()

	schemas = {}
	for item in data:
		schemas[item['service_name']] = resolv_schema(item, slots_categorical, slots_api, intents_api)

	return schemas, slots_categorical, slots_api, intents_api

def process_slotCopy_pred(slots_categorical):


	fn_samples = []
	tp_samples = []
	fp_samples = []
	tn_samples = []

	slots_tag_results = {}
	for i in range(len(total_eval[0])):

		slot_name = total_eval[4][i]

		if total_eval[1][i] == 'SYSTEM':
			continue

		# if slots_categorical[slot_name]:
		# 	continue

		if slot_name not in slots_tag_results:
			slots_tag_results[slot_name] = {}
			slots_tag_results[slot_name]['tp'] = 0
			slots_tag_results[slot_name]['fn'] = 0
			slots_tag_results[slot_name]['tn'] = 0
			slots_tag_results[slot_name]['fp'] = 0
			slots_tag_results[slot_name]['p'] = 0
			slots_tag_results[slot_name]['r'] = 0
			slots_tag_results[slot_name]['f1'] = 0

		if total_eval[-2][i] == 0:
			if total_eval[-1][i] == 0:
				slots_tag_results[slot_name]['tp'] += 1
				tp_samples.append(i)
			else:
				slots_tag_results[slot_name]['fn'] += 1
				fn_samples.append(i)
		else:
			if total_eval[-1][i] == 1:
				slots_tag_results[slot_name]['tn'] += 1
				tn_samples.append(i)
			else:
				slots_tag_results[slot_name]['fp'] += 1
				fp_samples.append(i)

	for i in tp_samples:
		slot = total_eval[4][i]
		related_api_in_train = []
		related_files_in_train = 0
		if slot in slots_api_train:
			related_api_in_train = slots_api_train[slot]
			for api in related_api_in_train:
				if api in api2dial_idx_train:
					related_files_in_train += len(api2dial_idx_train[api])

		print('#tp samples: ', 'in train: %s'%str(total_eval[4][i] in slots_api_train), \
				related_api_in_train, related_files_in_train, \
				 total_eval[0][i], total_eval[1][i],\
					total_eval[2][i], total_eval[4][i], total_eval[-2][i], total_eval[-1][i], '%.4f'%(total_eval[-3][i]), '\n')

	# for i in tn_samples[:200]:
	# 	slot = total_eval[4][i]
	# 	related_api_in_train = []
	# 	related_files_in_train = 0
	# 	if slot in slots_api_train:
	# 		related_api_in_train = slots_api_train[slot]
	# 		for api in related_api_in_train:
	# 			if api in api2dial_idx_train:
	# 				related_files_in_train += len(api2dial_idx_train[api])

	# 	print('#tn samples: ', 'in train: %s'%str(total_eval[4][i] in slots_api_train), \
	# 			related_api_in_train, related_files_in_train, total_eval[5][i], total_eval[6][i], \
	# 			 total_eval[7][i], total_eval[8][i], '\n', total_eval[0][i], total_eval[1][i],\
	# 				total_eval[2][i], total_eval[4][i], total_eval[-2][i], total_eval[-1][i], '\n')
	
	print('#'*50)

	for i in fn_samples:
		slot = total_eval[4][i]
		related_api_in_train = []
		related_files_in_train = 0
		related_dials_in_train = []
		if slot in slots_api_train:
			related_api_in_train = slots_api_train[slot]
			for api in related_api_in_train:
				if api in api2dial_idx_train:
					related_files_in_train += len(api2dial_idx_train[api])
					related_dials_in_train += api2dial_idx_train[api]

		print('#fn samples: ', 'in train: %s'%str(total_eval[4][i] in slots_api_train), \
				related_api_in_train, related_files_in_train, '\n', total_eval[0][i], total_eval[1][i],\
					total_eval[2][i], total_eval[4][i], total_eval[-2][i], total_eval[-1][i], '%.4f'%(total_eval[-3][i]), '\n')

	for i in fp_samples:
		slot = total_eval[4][i]
		related_api_in_train = []
		related_files_in_train = 0
		related_dials_in_train = []
		if slot in slots_api_train:
			related_api_in_train = slots_api_train[slot]
			for api in related_api_in_train:
				if api in api2dial_idx_train:
					related_files_in_train += len(api2dial_idx_train[api])
					related_dials_in_train += api2dial_idx_train[api]
		print('#fp samples: ', 'in train: %s'%str(total_eval[4][i] in slots_api_train), \
				related_api_in_train, related_files_in_train, '\n', total_eval[0][i], total_eval[1][i],\
					total_eval[2][i], total_eval[4][i], total_eval[-2][i], total_eval[-1][i], '%.4f'%(total_eval[-3][i]),'\n')
	TP, FN, TN, FP = 0, 0, 0, 0
	for k in slots_tag_results:
		TP += slots_tag_results[k]['tp']
		FN += slots_tag_results[k]['fn']
		TN += slots_tag_results[k]['tn']
		FP += slots_tag_results[k]['fp']

		if slots_tag_results[k]['tp'] + slots_tag_results[k]['fp'] > 0 and \
				slots_tag_results[k]['tp'] + slots_tag_results[k]['fn'] > 0:
			slots_tag_results[k]['p'] = slots_tag_results[k]['tp']/(slots_tag_results[k]['tp'] \
														+ slots_tag_results[k]['fp'])
			slots_tag_results[k]['r'] = slots_tag_results[k]['tp']/(slots_tag_results[k]['tp'] + \
														slots_tag_results[k]['fn'])
			if slots_tag_results[k]['p'] == 0:
				slots_tag_results[k]['f1'] = 0
			else:
				slots_tag_results[k]['f1'] = 2*slots_tag_results[k]['p']*slots_tag_results[k]['r']/\
						(slots_tag_results[k]['p'] + slots_tag_results[k]['r'])


	print('# micro')
	print('slot, tp, fn, tn, fp, p, r, f1')

	for k in slots_tag_results:
		print('%s, %d, %d, %d, %d, %.3f, %.3f, %.3f'%\
					(k, slots_tag_results[k]['tp'], slots_tag_results[k]['fn'],\
					slots_tag_results[k]['tn'], slots_tag_results[k]['fp'], \
					slots_tag_results[k]['p'], slots_tag_results[k]['r'], slots_tag_results[k]['f1']))


	precision_micro = TP/(TP+FP)
	recall_micro = TP/(TP+FN)
	F1_micro = 2*precision_micro*recall_micro/(precision_micro+recall_micro)
	print('#Micro TP:%d, FN:%d, TN:%d, FP:%d'%(TP, FN, TN, FP))
	print('#Micro precision: %.4f, recall: %.4f, F1: %.4f'%(precision_micro, recall_micro, F1_micro))

	precision_macro = 0
	recall_macro = 0
	num_slots_consider = 0
	for k in slots_tag_results:
		if slots_tag_results[k]['tp'] == 0 and slots_tag_results[k]['fn'] == 0 \
			and slots_tag_results[k]['fp'] == 0:
			continue
		num_slots_consider += 1
		precision_macro += slots_tag_results[k]['p']
		recall_macro += slots_tag_results[k]['r']
	precision_macro = precision_macro/num_slots_consider
	recall_macro = recall_macro/num_slots_consider
	F1_macro = 2*precision_macro*recall_macro/(precision_macro+recall_macro)
	print('#Macro precision: %.4f, recall: %.4f, F1: %.4f'%(precision_macro, recall_macro, F1_macro))

	'''
	save formatted predicted results as json file
	'''
	results = {}
	for i in range(len(total_eval[3])):
		slot = total_eval[4][i]
		dial_id = total_eval[0][i]
		uttr = total_eval[2][i]

		# if total_eval[-1][i] == 1:
		# 	continue

		if dial_id not in results:
			results[dial_id] = {}

		if uttr not in results[dial_id]:
			results[dial_id][uttr] = {}

		if len(results[dial_id][uttr]) == 0:
			results[dial_id][uttr][slot] = '%.6f'%(total_eval[-3][i])
		else:
			for k in [k for k in results[dial_id][uttr]]:
				if total_eval[-3][i] > float(results[dial_id][uttr][k]):
					results[dial_id][uttr][slot] = '%.6f'%(total_eval[-3][i])
					del results[dial_id][uttr][k]


	with open("slotNotCare_pred_0p0sampleDev.json","w") as dump_f:
		json.dump(results, dump_f, sort_keys=True, indent=4, separators=(',', ':'))


schemas_test, slots_categorical_test, slots_api_test, intents_api_test = \
							load_shemas('../../dstc8-schema-guided-dialogue/dev/schema.json')

schemas_train, slots_categorical_train, slots_api_train, intents_api_train = \
							load_shemas('../../dstc8-schema-guided-dialogue/train/schema.json')

api2dial_idx_train = resolv_train_dial('../../dstc8-schema-guided-dialogue/train/')
print(api2dial_idx_train)

# best, bert-large, squad_v1, multi_sample
# total_eval = list(pkl.load(open('../detailed_results/2019-08-02_06_38_59_total_test_results.pkl', 'rb')))

total_eval = list(pkl.load(open('../detailed_results/2019-09-24_22_30_08_total_eval_results.pkl', 'rb')))

process_slotCopy_pred(slots_categorical_test)
# process_Cate_cls(slots_categorical_test)
# 


