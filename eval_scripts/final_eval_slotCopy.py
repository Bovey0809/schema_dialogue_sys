import json
import pickle as pkl
import os
import nltk
import re
from pytorch_pretrained_bert.tokenization import BertTokenizer

tokenizer_bert = BertTokenizer.from_pretrained('/DSTC8-SGDST/pre_trained_models_base_cased', \
 do_lower_case=False, cache_dir='/DSTC8-SGDST/pre_trained_models_base_cased')


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
        schemas[item['service_name']] = resolv_schema(item, slots_categorical,
                                                      slots_api, intents_api)

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
                temp_turn['frames'][
                    frame['service']]['service'] = frame['service']

                temp_turn['frames'][frame['service']]['slots'] = {}
                for slot in frame['slots']:
                    temp_turn['frames'][frame['service']]['slots'][
                        slot['slot']] = slot

                temp_turn['frames'][frame['service']]['actions'] = {}
                for act in frame['actions']:
                    temp_turn['frames'][frame['service']]['actions'][
                        act['slot']] = act

                slots_sys_his[frame['service']] = set([key for key in temp_turn['frames'] \
                           [frame['service']]['actions']])

            else:
                temp_turn['frames'][frame['service']] = {}
                temp_turn['frames'][frame['service']]['state'] = frame['state']
                temp_turn['frames'][
                    frame['service']]['service'] = frame['service']
                temp_turn['frames'][frame['service']]['slots'] = {}

                for slot in frame['slots']:
                    temp_turn['frames'][frame['service']]['slots'][
                        slot['slot']] = slot

                slots_usr = set([key for key in frame['state']['slot_values']])
                slots_usr_new = set([
                    key
                    for key in temp_turn['frames'][frame['service']]['slots']
                ])
                slots_usr_temp = slots_usr - slots_usr_new - slots_usr_his[
                    frame['service']]

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
                uttr_inp = uttr_inp[-96:]
                uttr_inp = ' '.join(uttr_inp)

                _dial['inpUttr2orUttr'][uttr_inp] = last_sys_turn[
                    'utterance'] + ' # ' + turn['utterance']
                _dial[turn['utterance']] = {}
                _dial[turn['utterance']]['uttr_with_sys'] = uttr_with_sys

    return dials


def process_slotCopy_pred(slots_categorical):

    fn_samples = []
    tp_samples = []
    fp_samples = []
    tn_samples = []

    slots_tag_results = {}
    for i in range(len(total_eval[0])):

        slot_name = total_eval[5][i]

        if total_eval[2][i] == 'SYSTEM':
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
        slot = total_eval[5][i]
        related_api_in_train = []
        related_files_in_train = 0
        if slot in slots_api_train:
            related_api_in_train = slots_api_train[slot]
            for api in related_api_in_train:
                if api in api2dial_idx_train:
                    related_files_in_train += len(api2dial_idx_train[api])

        print('#tp samples: ', 'in train: %s'%str(total_eval[5][i] in slots_api_train), \
          related_api_in_train, related_files_in_train, total_eval[6][i], total_eval[7][i], \
           total_eval[8][i], total_eval[9][i], '\n', total_eval[0][i], total_eval[1][i], total_eval[2][i],\
           total_eval[3][i], total_eval[5][i], total_eval[-2][i], total_eval[-1][i], '%.4f'%(total_eval[-3][i]), '\n')

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

    print('#' * 50)

    for i in fn_samples:
        slot = total_eval[5][i]
        related_api_in_train = []
        related_files_in_train = 0
        if slot in slots_api_train:
            related_api_in_train = slots_api_train[slot]
            for api in related_api_in_train:
                if api in api2dial_idx_train:
                    related_files_in_train += len(api2dial_idx_train[api])

        print('#fn samples: ', 'in train: %s'%str(total_eval[5][i] in slots_api_train), \
          related_api_in_train, related_files_in_train, total_eval[6][i], total_eval[7][i], \
           total_eval[8][i], total_eval[9][i], '\n', total_eval[0][i], total_eval[1][i], total_eval[2][i],\
           total_eval[3][i], total_eval[5][i], total_eval[-2][i], total_eval[-1][i], '%.4f'%(total_eval[-3][i]), '\n')

    for i in fp_samples:
        slot = total_eval[5][i]
        related_api_in_train = []
        related_files_in_train = 0
        if slot in slots_api_train:
            related_api_in_train = slots_api_train[slot]
            for api in related_api_in_train:
                if api in api2dial_idx_train:
                    related_files_in_train += len(api2dial_idx_train[api])

        print('#fp samples: ', 'in train: %s'%str(total_eval[5][i] in slots_api_train), \
          related_api_in_train, related_files_in_train, total_eval[6][i], total_eval[7][i], \
           total_eval[8][i], total_eval[9][i], '\n', total_eval[0][i], total_eval[1][i], total_eval[2][i],\
           total_eval[3][i], total_eval[5][i], total_eval[-2][i], total_eval[-1][i], '%.4f'%(total_eval[-3][i]), '\n')

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

    precision_micro = TP / (TP + FP + 0.0000001)
    recall_micro = TP / (TP + FN + 0.0000001)
    F1_micro = 2 * precision_micro * recall_micro / (precision_micro +
                                                     recall_micro + 0.00000001)
    print('#Micro TP:%d, FN:%d, TN:%d, FP:%d' % (TP, FN, TN, FP))
    print('#Micro precision: %.4f, recall: %.4f, F1: %.4f' %
          (precision_micro, recall_micro, F1_micro))

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
    precision_macro = precision_macro / num_slots_consider
    recall_macro = recall_macro / num_slots_consider
    F1_macro = 2 * precision_macro * recall_macro / (precision_macro +
                                                     recall_macro + 0.0000001)
    print('#Macro precision: %.4f, recall: %.4f, F1: %.4f' %
          (precision_macro, recall_macro, F1_macro))
    '''
	save formatted predicted results as json file
	'''
    results = {}
    for i in range(len(total_eval[3])):
        slot = total_eval[5][i]
        dial_id = total_eval[0][i]
        api = total_eval[1][i]
        uttr = total_eval[3][i]

        if '[SEP]' in total_eval[4][i]:
            idx = total_eval[4][i].index('[SEP]')
            uttr_inp = ' '.join(total_eval[4][i][idx + 1:])
        else:
            uttr_inp = ' '.join(total_eval[4][i])

        if uttr_inp not in dev_dials[dial_id]['inpUttr2orUttr']:
            print(dial_id, uttr_inp)
            print(dev_dials[dial_id]['inpUttr2orUttr'])
            exit()
        uttr_inp = dev_dials[dial_id]['inpUttr2orUttr'][uttr_inp]

        # if total_eval[-1][i] == 1:
        # 	continue

        if dial_id not in results:
            results[dial_id] = {}

        if uttr_inp not in results[dial_id]:
            results[dial_id][uttr_inp] = {}

        if api not in results[dial_id][uttr_inp]:
            results[dial_id][uttr_inp][api] = {}

        results[dial_id][uttr_inp][api][slot] = '%.4f' % (total_eval[-3][i])

    with open("slotCopy_pred_0p99sampleDev_testGround_onTest.json",
              "w") as dump_f:
        json.dump(results,
                  dump_f,
                  sort_keys=True,
                  indent=4,
                  separators=(',', ':'))


schemas_test, slots_categorical_test, slots_api_test, intents_api_test = \
       load_shemas('../../dstc8-schema-guided-dialogue/test/schema.json')

schemas_train, slots_categorical_train, slots_api_train, intents_api_train = \
       load_shemas('../../dstc8-schema-guided-dialogue/train/schema.json')

api2dial_idx_train = resolv_train_dial(
    '../../dstc8-schema-guided-dialogue/train/')
print(api2dial_idx_train)

dev_dials = load_dev_dial('../../dstc8-schema-guided-dialogue/test/')

total_eval = list(
    pkl.load(
        open('../detailed_results/2019-10-11_20_56_19_total_eval_results.pkl',
             'rb')))

process_slotCopy_pred(slots_categorical_test)
# process_Cate_cls(slots_categorical_test)
#
