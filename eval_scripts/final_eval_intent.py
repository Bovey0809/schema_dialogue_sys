import json
import pickle as pkl
import os
import nltk
import re
from pytorch_pretrained_bert.tokenization import BertTokenizer

tokenizer_bert = BertTokenizer.from_pretrained('../../pre_trained_models_base_cased/', \
	do_lower_case=False, cache_dir='../../pre_trained_models_base_cased/')


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

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
      return True
    return False

def default_tokenize(utterance):
		uttr_words = []
		prev_is_whitespace = True
		for c in utterance:
			if is_whitespace(c):
				prev_is_whitespace = True
			else:
				if prev_is_whitespace:
					uttr_words.append(c)
				else:
					uttr_words[-1] += c
				prev_is_whitespace = False

		uttr_words_tokenized = []
		for idx, w in enumerate(uttr_words):
			words = nltk.word_tokenize(w)
			uttr_words_tokenized += words

		word_to_BertTokenized = {}
		uttr_words_BertTokenized = []
		for idx, w in enumerate(uttr_words_tokenized):
			words = tokenizer_bert.tokenize(w)
			word_to_BertTokenized[idx] = list(range(len(uttr_words_BertTokenized), \
				len(uttr_words_BertTokenized)+len(words)))
			uttr_words_BertTokenized += words

		return uttr_words_BertTokenized

def resolv_dial(raw):
		resolved = {}
		resolved['dialogue_id'] = raw['dialogue_id']
		resolved['services'] = raw['services']
		resolved['file_name'] = raw['file_name']

		resolved['turns'] = []

		slots_sys_his = {}
		slots_usr_his = {}
		for _service in resolved['services']:
				slots_sys_his[_service] = set([])
				slots_usr_his[_service] = set([])

		for item in raw['turns']:
			temp_turn = {}
			temp_turn['utterance'] = item['utterance']
			# if temp_turn['utterance'].isupper():
			# 	temp_turn['utterance'] = temp_turn['utterance'].lower()
			temp_turn['speaker'] = item['speaker']
			temp_turn['frames'] = {}

			for frame in item['frames']:
				if temp_turn['speaker'] == 'SYSTEM':
					temp_turn['frames'][frame['service']] = {}
					temp_turn['frames'][frame['service']]['service'] = frame['service']

					temp_turn['frames'][frame['service']]['slots'] = {}
					for slot in frame['slots']:
						temp_turn['frames'][frame['service']]['slots'][slot['slot']] = slot

					temp_turn['frames'][frame['service']]['actions'] = {}
					for act in frame['actions']:
						temp_turn['frames'][frame['service']]['actions'][act['slot']] = act

					slots_sys_his[frame['service']] = set([key for key in temp_turn['frames'] \
																[frame['service']]['actions']])

				else:
					temp_turn['frames'][frame['service']] = {}
					temp_turn['frames'][frame['service']]['state'] = frame['state']
					temp_turn['frames'][frame['service']]['service'] = frame['service']
					temp_turn['frames'][frame['service']]['slots'] = {}

					for slot in frame['slots']:
						temp_turn['frames'][frame['service']]['slots'][slot['slot']] = slot

					slots_usr = set([key for key in frame['state']['slot_values']])
					slots_usr_new = set([key for key in temp_turn['frames'][frame['service']]['slots']])
					slots_usr_temp = slots_usr - slots_usr_new - slots_usr_his[frame['service']]

					# if len(slots_usr_temp - slots_sys_his[frame['service']]) > 0 and \
					# 		                                              len(slots_usr_temp) > 0:
					# 	print('#1: slot from last sys:', item['utterance'])

					# if len(slots_usr_temp) == 0:
					# 	print('#2:', item['utterance'])

					slots_usr_his[frame['service']] = slots_usr

			resolved['turns'].append(temp_turn)

		return resolved

def load_dev_dial(path):
	file_list = os.listdir(path)
	data = []
	for file in file_list:
		if 'dialogue' in file:
			# print(path+file)
			fp = open(path + file, 'r')
			_data = json.loads(fp.read())
			for x in _data:
				x['file_name'] = file
			data += _data
			fp.close()
	print('#number of dials:', len(data))

	dials = {}
	for item in data:
		dials[item['dialogue_id']] = resolv_dial(item)

	for dialogue_id in dials:
		_dial = dials[dialogue_id]
		last_sys_turn = {}
		last_sys_turn['utterance'] = ''
		_dial['inpUttr2orUttr'] = {}
		for turn in _dial['turns']:
			if turn['speaker'] == 'SYSTEM':	
				last_sys_turn = turn
			else:
				uttr_with_sys = 'sys : ' +last_sys_turn['utterance'] + ' usr : ' \
												+ turn['utterance']
				uttr_inp = default_tokenize(uttr_with_sys)
				uttr_inp = ['keep', '[SEP]'] + uttr_inp
				uttr_inp = uttr_inp[-96:]
				if 'keep [SEP]' in ' '.join(uttr_inp):
					uttr_inp = ' '.join(uttr_inp[2:])
				else:
					uttr_inp = ' '.join(uttr_inp)

				_dial['inpUttr2orUttr'][uttr_inp] = last_sys_turn['utterance'] + ' # ' + turn['utterance']
				_dial[turn['utterance']] = {}
				_dial[turn['utterance']]['uttr_with_sys'] = uttr_with_sys

	return dials

def process_intent():

	print('#'*20)
	print('derive results for Intent prediciton')
	print('#'*20)

	fn_samples = []
	tp_samples = []

	for i in range(len(total_eval[0])):

		if total_eval[1][i] == 'SYSTEM':
			continue

		if total_eval[-1][i] == total_eval[-2][i]:
			tp_samples.append(i)
		else:
			fn_samples.append(i)

	for i in fn_samples:
		pred = total_eval[4][i][total_eval[-1][i]]
		ground = total_eval[4][i][total_eval[-2][i]]
		print('Error samples: ', 'intent in Train: %s'%(total_eval[4][i][0] in intents_api_train), \
			total_eval[0][i], total_eval[2][i], ground, pred)

	print('Accuracy:%.4f'%(len(tp_samples)/len(total_eval[0])))

	'''
	save formatted predicted results as json file
	'''
	results = {}
	for i in range(len(total_eval[0])):
		if total_eval[1][i] == 'SYSTEM':
			continue

		pred = total_eval[4][i][total_eval[-1][i]]
		dial_id = total_eval[0][i]
		uttr = total_eval[2][i]

		if dial_id not in results:
			results[dial_id] = {}
		results[dial_id][uttr] = pred

	with open("intent_pred.json","w") as dump_f:
		json.dump(results, dump_f, sort_keys=True, indent=4, separators=(',', ':'))

def process_intent_infer():

	print('#'*20)
	print('derive results for Intent prediciton')
	print('#'*20)

	'''
	save formatted predicted results as json file
	'''
	tmp_results = {}
	for i in range(len(total_eval[0])):
		dial_id = total_eval[0][i]
		api_id = total_eval[1][i][0]
		assert len(total_eval[1][i]) == 1
	
		turn_id = total_eval[2][i]
		last_intent_label = total_eval[6][i]
		if dial_id not in tmp_results:
			tmp_results[dial_id] = {}
		if turn_id not in tmp_results[dial_id]:
			tmp_results[dial_id][turn_id] = {}
		if api_id not in tmp_results[dial_id][turn_id]:
			tmp_results[dial_id][turn_id][api_id] = {}
		
		if 'jump [SEP]' in ' '.join(total_eval[4][i]) or 'keep [SEP]' in ' '.join(total_eval[4][i]):
			uttr_inp = ' '.join(total_eval[4][i][2:])
		else:
			uttr_inp = ' '.join(total_eval[4][i])

		if uttr_inp not in dev_dials[dial_id]['inpUttr2orUttr']:
			print(dial_id, uttr_inp)
			print(dev_dials[dial_id]['inpUttr2orUttr'])
			exit()
		uttr_inp = dev_dials[dial_id]['inpUttr2orUttr'][uttr_inp]

		tmp_results[dial_id][turn_id][api_id][last_intent_label] = (total_eval[0][i], total_eval[1][i], total_eval[2][i], \
			uttr_inp, total_eval[4][i], total_eval[5][i], total_eval[6][i], total_eval[7][i], total_eval[8][i])
	
	right = 0
	wrong = 0
	for dial_id in tmp_results:
		for i in range(len(tmp_results[dial_id])):
			for api in tmp_results[dial_id][i]:

				intent = tmp_results[dial_id][i][api][0][5]
				try:
					last_intent_label = tmp_results[dial_id][i-1][api]['intent_label']
				except KeyError:
					last_intent_label = tmp_results[dial_id][i][api][0][5].index('none')
				
				tmp_results[dial_id][i][api]['intent_label'] = tmp_results[dial_id][i][api][last_intent_label][-1]

				tmp_results[dial_id][i][api]['pred'] = intent[tmp_results[dial_id][i][api]['intent_label']]


				if tmp_results[dial_id][i][api]['intent_label'] == tmp_results[dial_id][i][api][last_intent_label][-2]:
					right += 1
				else:
					wrong +=1
					utterance = tmp_results[dial_id][i][api][0][3]
					ground = intent[tmp_results[dial_id][i][api][0][-2]]
					pred = intent[tmp_results[dial_id][i][api]['intent_label']]
					print('Error samples: ', 'intent in Train: %s'%(intent[0] in intents_api_train), \
				dial_id, utterance, api, ground, pred)

	print('right:%d, wrong:%d, acc:%.4f'%(right, wrong, 1.*right/(right+wrong)))

	results = {}
	for dial_id in tmp_results:
		for turn_id in tmp_results[dial_id]:
			for api in tmp_results[dial_id][turn_id]:
				uttr_inp = tmp_results[dial_id][turn_id][api][0][3]
				pred = tmp_results[dial_id][turn_id][api]['pred']

				if dial_id not in results:
					results[dial_id] = {}
				if uttr_inp not in results[dial_id]:
					results[dial_id][uttr_inp] = {}
				results[dial_id][uttr_inp][api] = pred

	with open("intent_pred_0p99sampleDev_onTest.json","w") as dump_f:
		json.dump(results, dump_f, sort_keys=True, indent=4, separators=(',', ':'))


schemas_test, slots_categorical_test, slots_api_test, intents_api_test = \
							load_shemas('../../dstc8-schema-guided-dialogue/test/schema.json')

schemas_train, slots_categorical_train, slots_api_train, intents_api_train = \
							load_shemas('../../dstc8-schema-guided-dialogue/train/schema.json')

api2dial_idx_train = resolv_train_dial('../../dstc8-schema-guided-dialogue/train/')
print(api2dial_idx_train)

dev_dials = load_dev_dial('../../dstc8-schema-guided-dialogue/test/')


total_eval = list(pkl.load(open('../detailed_results/2019-10-10_18_31_06_total_eval_results.pkl', 'rb')))

process_intent_infer()


