import json
import os
import copy

fp = open('cate_pred_0p0sampleDev_onTest.json', 'r')
cate_pred = json.loads(fp.read())
fp.close()

fp = open('intent_pred_0p0sampleDev_onTest.json', 'r')
intent_pred = json.loads(fp.read())
fp.close()

fp = open('nonCate_pred_0p0sampleDev_onTest.json', 'r')
nonCate_pred = json.loads(fp.read())
fp.close()

fp = open('request_pred_0p0Dev_onTest.json', 'r')
request_pred = json.loads(fp.read())
fp.close()


fp = open('slotCopy_pred_0p0sampleDev_testGround_onTest.json', 'r')
slotCopy_pred = json.loads(fp.read())
fp.close()

fp = open('slotCross_pred_0p0sampleDev_testGround_onTest.json', 'r')
slotCross_pred = json.loads(fp.read())
fp.close()

# fp = open('slotNotCare_pred_0p0sampleDev_preFineTuned.json', 'r')
# slotNotCare_pred = json.loads(fp.read())
# fp.close()

fp = open('../../dstc_data0913/test/schema.json', 'r')
_schemas = json.loads(fp.read())
fp.close()

fp = open('../../dstc_data0913/train/schema.json', 'r')
_schemas_train = json.loads(fp.read())
fp.close()

def resolv_schema(raw):
	resolved = {}
	resolved['service_name'] = raw['service_name']
	resolved['description'] = raw['description']
	resolved['slots'] = {}
	for item in raw['slots']:

		resolved['slots'][item['name']] = item

	resolved['intents'] = {}
	for item in raw['intents']:
		resolved['intents'][item['name']] = item

	return resolved

def resolv_dial(raw):
	resolved = {}
	resolved['dialogue_id'] = raw['dialogue_id']
	resolved['services'] = raw['services']

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


		resolved['turns'].append(temp_turn)

	return resolved


def process_file(path, file):
	fp = open(path+file, 'r')
	raw = json.loads(fp.read())
	fp.close()

	schemas = {}
	for x in _schemas:
		schemas[x['service_name']] = resolv_schema(x)
		schemas[x['service_name']]['slots_needed'] = {}
		for intent in schemas[x['service_name']]['intents']:
			intent_info = schemas[x['service_name']]['intents'][intent]
			for k in intent_info['required_slots']:
				schemas[x['service_name']]['slots_needed'][k] = ''
			for k in intent_info['optional_slots']:
				schemas[x['service_name']]['slots_needed'][k] = ''

	schemas_train = {}
	for x in _schemas_train:
		schemas_train[x['service_name']] = resolv_schema(x)
		schemas_train[x['service_name']]['slots_needed'] = {}
		for intent in schemas_train[x['service_name']]['intents']:
			intent_info = schemas_train[x['service_name']]['intents'][intent]
			for k in intent_info['required_slots']:
				schemas_train[x['service_name']]['slots_needed'][k] = ''
			for k in intent_info['optional_slots']:
				schemas_train[x['service_name']]['slots_needed'][k] = ''

	data = []
	for x in raw:
		x = resolv_dial(x)
		dial_id = x['dialogue_id']
		slot_values_history = {}
		sys_slot_provide = {}
		frame_slots_history = {}

		if dial_id not in intent_pred:
			print('### %s not in preds'% dial_id)
			continue

		pred_x = {}
		pred_x['dialogue_id'] = dial_id
		pred_x['services'] = x['services']
		pred_x['turns'] = []

		for _service in x['services']:
				sys_slot_provide[_service] = {}
				slot_values_history[_service] = {}
				frame_slots_history[_service] = {}
		last_sys_uttr = ''
		for turn in x['turns']:
			pred_turn = {}
			pred_turn['speaker'] = turn['speaker']
			pred_turn['utterance'] = turn['utterance']

			if turn['speaker'] == 'SYSTEM':
				pred_turn['frames'] = []
				last_sys_uttr = turn['utterance']
				for frame_id in turn['frames']:
					frame = turn['frames'][frame_id]

					for act in sys_slot_provide[frame_id]:
						sys_slot_provide[frame_id][act][2] += 1

					for act in frame['actions']:
						if frame['actions'][act]['act'] in set(['OFFER', 'INFORM', 'CONFIRM']):
							sys_slot_provide[frame_id][act] = [frame['actions'][act]['values'],\
													frame['actions'][act]['act'], 0]
							frame_slots_history[frame_id][act] = frame['actions'][act]['values']

						if frame['actions'][act]['act'] == 'REQUEST':
							if len(frame['actions'][act]['values']) == 1:
								sys_slot_provide[frame_id][act] = [frame['actions'][act]['values'],\
												frame['actions'][act]['act'], 0]

					frame['actions'] = [frame['actions'][k] for k in frame['actions']]
					frame['slots'] = [frame['slots'][k] for k in frame['slots']]
					pred_turn['frames'].append(frame)


			else:
				pred_turn['frames'] = []
				uttr = turn['utterance']
				uttr_inp = last_sys_uttr + ' # ' + uttr
				uttr = uttr_inp
				for frame_id in turn['frames']:
					frame = turn['frames'][frame_id]
					try:
						frame['state']['active_intent'] = intent_pred[dial_id][uttr][frame_id]
					except KeyError:
						print(dial_id, uttr, frame_id)
						exit()
					if frame['state']['active_intent'] == "none":
						frame['state']['active_intent'] = "NONE"
				
					try:
						frame['state']['requested_slots'] = []
						for k in request_pred[dial_id][uttr][frame_id]:
							if float(request_pred[dial_id][uttr][frame_id][k]) > 0.9:
								frame['state']['requested_slots'].append(k)
						#frame['state']['requested_slots'] = [k for k in request_pred[dial_id][uttr][frame_id]]
					except KeyError:
						frame['state']['requested_slots'] = []

					slot_values = slot_values_history[frame['service']]
					slots = []

					try:
						for k in nonCate_pred[dial_id][uttr][frame_id]:
							if float(nonCate_pred[dial_id][uttr][frame_id][k][2]) > 1.0:
								slot_values[k] = [nonCate_pred[dial_id][uttr][frame_id][k][0]]
								p1 = nonCate_pred[dial_id][uttr][frame_id][k][1][0]
								p2 = nonCate_pred[dial_id][uttr][frame_id][k][1][1]
								slots.append({"exclusive_end":p2, "slot":k, "start":p1})
					except KeyError:
						1

					try:
						for k in cate_pred[dial_id][uttr][frame_id]:
							if float(cate_pred[dial_id][uttr][frame_id][k][1]) > 0.8 and k not in frame['state']['requested_slots']:
								slot_values[k] = [cate_pred[dial_id][uttr][frame_id][k][0]]
					except KeyError:
						1
					
					# # copy slots from other frame-history
					try:
						slots_cross = slotCross_pred[dial_id][uttr][frame_id]
						for k in slots_cross:
							if float(slots_cross[k][2]) > 0.9:
								api_cross = slots_cross[k][0]
								slot_cross = slots_cross[k][1]
								if k not in slot_values and slot_cross in frame_slots_history[api_cross]:
									slot_values[k] = frame_slots_history[api_cross][slot_cross]
									print('cross-copied %s from %s in %s'%(k, slot_cross, api_cross))
					except KeyError:
						1


					# # # copy slots from system by preds(manner-2)
					try:
						slots_needed = slotCopy_pred[dial_id][uttr][frame_id]
						intent = intent_pred[dial_id][uttr][frame_id]
						for k in slots_needed:
							if float(slots_needed[k]) > 0.85:
								if k not in slot_values and k in sys_slot_provide[frame_id]:
								# if k in sys_slot_provide[frame_id]:
									if sys_slot_provide[frame_id][k][2] > 9999:
										continue
									if sys_slot_provide[frame_id][k][1] != 'INFORM':
										slot_values[k] = sys_slot_provide[frame_id][k][0]
										if schemas[frame_id]['slots'][k]['is_categorical'] == True:
											print('1---', dial_id, uttr, k, sys_slot_provide[frame_id][k][0])
										if schemas[frame_id]['slots'][k]['is_categorical'] == False:
											print('2---', dial_id, uttr, k, sys_slot_provide[frame_id][k][0])
					except KeyError:
						1

					# for k in schemas[frame_id]['slots_needed']:
					# 	if k in slotNotCare_pred[dial_id][uttr]:
					# 		if float(slotNotCare_pred[dial_id][uttr][k]) > 0.995:
					# 			slot_values[k] = ['dontcare']


					for k in [k for k in slot_values]:
						if k not in schemas[frame_id]['slots_needed']:
							del slot_values[k]

					frame['state']['slot_values'] = slot_values
					frame['slots'] = slots

					pred_turn['frames'].append(frame)

					slot_values_history[frame['service']] = copy.deepcopy(frame['state']['slot_values'])
					for k in slot_values_history[frame_id]:
						frame_slots_history[frame_id][k] = copy.deepcopy(slot_values_history[frame_id][k])

			pred_x['turns'].append(pred_turn)
		data.append(pred_x)
		
	with open("formatted/"+file, "w") as dump_f:
		json.dump(data, dump_f, sort_keys=True, indent=4, separators=(',', ':'))


path = '../../dstc_data0913/test/'
file_list = os.listdir(path)
for file in file_list:
	if 'dialogue' in file:
		process_file(path, file)