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


def charIdx2wordIdx_v1(utterance):
    uttr_words = []
    char_to_word_offset = []
    word2char = {}
    prev_is_whitespace = True
    for idx, c in enumerate(utterance):
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                uttr_words.append(c)
            else:
                uttr_words[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(uttr_words) - 1)
        if len(uttr_words) - 1 not in word2char:
            word2char[len(uttr_words) - 1] = []
        if not is_whitespace(c):
            word2char[len(uttr_words) - 1].append(idx)
    '''
	word tokenization
	'''
    word_to_wordTokenized = {}
    wordTokenized2char = {}
    uttr_words_tokenized = []
    tag_list_tokenized = []
    for idx, w in enumerate(uttr_words):
        assert len(w) == len(word2char[idx])

        m = re.search(r'[a-zA-Z]([.!?,])[^\s]', w)
        if m:
            w = w.replace(m.group(1), m.group(1) + ' ')

        words = nltk.word_tokenize(w)
        if len(words) == 1:
            wordTokenized2char[len(wordTokenized2char)] = word2char[idx]
        if len(words) > 1:
            offset = 0
            for _w in words:
                wordTokenized2char[len(
                    wordTokenized2char)] = word2char[idx][offset:offset +
                                                          len(_w)]
                offset += len(_w)

        word_to_wordTokenized[idx] = list(range(len(uttr_words_tokenized), \
         len(uttr_words_tokenized)+len(words)))
        uttr_words_tokenized += words
    '''
	subword tokenization (BERT-base-lowercase)
	'''
    word_to_BertTokenized = {}
    BertTokenized2char = {}
    uttr_words_BertTokenized = []
    tag_list_BertTokenized = []
    for idx, w in enumerate(uttr_words_tokenized):
        words = tokenizer_bert.tokenize(w)
        for _ in words:
            BertTokenized2char[len(
                BertTokenized2char)] = wordTokenized2char[idx]
        word_to_BertTokenized[idx] = list(range(len(uttr_words_BertTokenized), \
         len(uttr_words_BertTokenized)+len(words)))
        uttr_words_BertTokenized += words

    assert len(BertTokenized2char) == len(uttr_words_BertTokenized)

    return BertTokenized2char


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
                uttr_inp = default_tokenize(uttr_with_sys)[-(96 - 1):] + [
                    'null'
                ]
                uttr_inp = ' '.join(uttr_inp)
                _dial['inpUttr2orUttr'][uttr_inp] = last_sys_turn[
                    'utterance'] + ' # ' + turn['utterance']
                _dial[turn['utterance']] = {}
                _dial[turn['utterance']]['uttr_with_sys'] = uttr_with_sys
                _dial[turn['utterance']][
                    'tokenIdx2charIdx'] = charIdx2wordIdx_v1(uttr_with_sys)
                _dial[turn['utterance']]['offset'] = len(
                    'sys : ' + last_sys_turn['utterance'] + ' usr : ')

    return dials


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


# print(slots_categorical[total_eval[3][0]])
'''
def inverse_tag(tokenized_uttr, span_label):
	if span_label[0] > span_label[1]:
		print(span_label)
	assert span_label[0] <= span_label[1]
	assert span_label[1] < len(tokenized_uttr)
	return ' '.join(tokenized_uttr[span_label[0]:span_label[1]+1])
'''


def inverse_tag(tokenized_uttr, uttr_tag, span_label):
    uttr_new = []
    word_idx = []
    assert span_label[0] <= span_label[1]
    i = 0
    while i < len(tokenized_uttr):
        word = ''
        if uttr_tag[i] != 2:
            word = tokenized_uttr[i]
            word_idx.append(i)

            i += 1
            while i < len(tokenized_uttr) and uttr_tag[i] == 2:
                word += tokenized_uttr[i].strip('##')
                i += 1
                # print(word)
            uttr_new.append(word)
        else:
            i += 1

    i = 0
    while i < len(word_idx):
        if word_idx[i] == span_label[0]:
            start = i
            break
        if word_idx[i] < span_label[0] and span_label[0] < word_idx[i + 1]:
            start = i
            break
        i += 1

    while i < len(word_idx):
        if word_idx[i] == span_label[1]:
            end = i
            break
        if word_idx[i] < span_label[1] and span_label[1] < word_idx[i + 1]:
            end = i
            break
        i += 1

    assert start <= end

    return ' '.join(uttr_new[start:end + 1])


def process_nonCate_span(slots_categorical):

    print('#' * 20)
    print('derive results for slot tagging')
    print('#' * 20)

    fn_samples = []
    fp_samples = []
    tp_samples = []

    cateSlots_used_as_nonCate = 0

    slots_tag_results = {}
    uttr_tag = total_eval.pop(5)
    for i in range(len(total_eval[3])):

        slot_name = total_eval[5][i]

        if total_eval[2][i] == 'SYSTEM':
            continue

        if slots_categorical[slot_name]:
            continue

        if slot_name not in slots_tag_results:
            slots_tag_results[slot_name] = {}
            slots_tag_results[slot_name]['tp'] = 0
            slots_tag_results[slot_name]['fn'] = 0
            slots_tag_results[slot_name]['tn'] = 0
            slots_tag_results[slot_name]['fp'] = 0
            slots_tag_results[slot_name]['p'] = 0
            slots_tag_results[slot_name]['r'] = 0
            slots_tag_results[slot_name]['f1'] = 0

        if total_eval[-8][i] == 0:
            if total_eval[-6][i][0] == total_eval[-5][i][0] and total_eval[-6][
                    i][1] == total_eval[-5][i][1]:
                slots_tag_results[slot_name]['tp'] += 1
                tp_samples.append(i)
            else:
                slots_tag_results[slot_name]['fn'] += 1
                fn_samples.append(i)
        else:
            if total_eval[-6][i][0] == total_eval[-5][i][0] and total_eval[-6][
                    i][1] == total_eval[-5][i][1]:
                slots_tag_results[slot_name]['tn'] += 1
            else:
                slots_tag_results[slot_name]['fp'] += 1
                fp_samples.append(i)

    print('#use cateslots as nonCate: ', cateSlots_used_as_nonCate)

    for i in tp_samples:
        slot = total_eval[5][i]
        related_api_in_train = []
        related_files_in_train = 0
        if slot in slots_api_train:
            related_api_in_train = slots_api_train[slot]
            for api in related_api_in_train:
                if api in api2dial_idx_train:
                    related_files_in_train += len(api2dial_idx_train[api])

        tag_truth = inverse_tag(total_eval[4][i], uttr_tag[i],
                                total_eval[-6][i])
        tag_pred = inverse_tag(total_eval[4][i], uttr_tag[i],
                               total_eval[-5][i])
        print('#tp samples: ', 'in train: %s'%str(total_eval[5][i] in slots_api_train), \
          related_api_in_train, related_files_in_train, total_eval[7][i], total_eval[8][i], \
           total_eval[9][i], '\n', total_eval[0][i], total_eval[1][i], total_eval[2][i],\
           total_eval[3][i], total_eval[5][i], [tag_truth], [tag_pred], '%.4f'%(total_eval[-10][i]), '\n')

    for i in fn_samples:
        slot = total_eval[5][i]
        related_api_in_train = []
        related_files_in_train = 0
        if slot in slots_api_train:
            related_api_in_train = slots_api_train[slot]
            for api in related_api_in_train:
                if api in api2dial_idx_train:
                    related_files_in_train += len(api2dial_idx_train[api])

        tag_truth = inverse_tag(total_eval[4][i], uttr_tag[i],
                                total_eval[-6][i])
        tag_pred = inverse_tag(total_eval[4][i], uttr_tag[i],
                               total_eval[-5][i])
        print('#fn samples: ', 'in train: %s'%str(total_eval[5][i] in slots_api_train), \
          related_api_in_train, related_files_in_train, total_eval[7][i], total_eval[8][i], \
           total_eval[9][i], '\n', total_eval[0][i], total_eval[1][i], total_eval[2][i],\
           total_eval[3][i], total_eval[5][i], [tag_truth], [tag_pred], '%.4f'%(total_eval[-10][i]), '\n')

    for i in fp_samples:
        slot = total_eval[5][i]
        related_api_in_train = []
        related_files_in_train = 0
        if slot in slots_api_train:
            related_api_in_train = slots_api_train[slot]
            for api in related_api_in_train:
                if api in api2dial_idx_train:
                    related_files_in_train += len(api2dial_idx_train[api])

        tag_truth = inverse_tag(total_eval[4][i], uttr_tag[i],
                                total_eval[-6][i])
        tag_pred = inverse_tag(total_eval[4][i], uttr_tag[i],
                               total_eval[-5][i])
        print('#fp samples: ', 'in train: %s'%str(total_eval[5][i] in slots_api_train), \
          related_api_in_train, related_files_in_train, total_eval[7][i], total_eval[8][i], \
           total_eval[9][i], '\n', total_eval[0][i], total_eval[1][i], total_eval[2][i],\
           total_eval[3][i], total_eval[5][i], [tag_truth], [tag_pred], '%.4f'%(total_eval[-10][i]), '\n')

    TP, FN, TN, FP = 0, 0, 0, 0
    for k in slots_tag_results:
        TP += slots_tag_results[k]['tp']
        FN += slots_tag_results[k]['fn']
        TN += slots_tag_results[k]['tn']
        FP += slots_tag_results[k]['fp']

        if slots_tag_results[k]['tp'] + slots_tag_results[k]['fp'] > 0:
            slots_tag_results[k]['p'] = slots_tag_results[k]['tp']/(slots_tag_results[k]['tp'] \
                       + slots_tag_results[k]['fp'])

        if slots_tag_results[k]['tp'] + slots_tag_results[k]['fn'] > 0:
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

    precision_micro = TP / (TP + FP)
    recall_micro = TP / (TP + FN + 1e-05)
    F1_micro = 2 * precision_micro * recall_micro / (precision_micro +
                                                     recall_micro + 1e-05)
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
                                                     recall_macro + 1e-05)
    print('#Macro precision: %.4f, recall: %.4f, F1: %.4f' %
          (precision_macro, recall_macro, F1_macro))
    '''
	save formatted predicted results as json file
	'''
    results = {}
    exception = {}
    for i in range(len(total_eval[3])):
        slot_name = total_eval[5][i]

        if total_eval[2][i] == 'SYSTEM':
            continue
        if slots_categorical[slot_name]:
            continue

        slot = total_eval[5][i]
        dial_id = total_eval[0][i]
        api = total_eval[1][i]
        uttr = total_eval[3][i]

        tag_pred = inverse_tag(total_eval[4][i], uttr_tag[i],
                               total_eval[-5][i])

        token_offset = len(
            dev_dials[dial_id][uttr]['tokenIdx2charIdx']) + 1 - len(
                total_eval[4][i])

        if total_eval[-5][i][0] + token_offset == len(
                dev_dials[dial_id][uttr]['tokenIdx2charIdx']):
            assert total_eval[4][i][total_eval[-5][i][0]] == 'null'
            continue

        if total_eval[-5][i][1] + token_offset == len(
                dev_dials[dial_id][uttr]['tokenIdx2charIdx']):
            assert total_eval[4][i][total_eval[-5][i][1]] == 'null'
            continue

        p1 = dev_dials[dial_id][uttr]['tokenIdx2charIdx'][total_eval[-5][i][0]
                                                          + token_offset][0]
        p2 = dev_dials[dial_id][uttr]['tokenIdx2charIdx'][total_eval[-5][i][1]
                                                          + token_offset][-1]

        p1 = p1 - dev_dials[dial_id][uttr]['offset']
        p2 = p2 - dev_dials[dial_id][uttr]['offset']

        if p1 < 0 or p2 < 0:
            continue

        str_ori = total_eval[3][i][p1:p2 + 1]

        if tokenizer_bert.tokenize(str_ori) != tokenizer_bert.tokenize(
                tag_pred):
            print(tag_pred, ' # ', str_ori)
            print(dial_id, uttr)
            print(p1, p2, len(uttr))

        # if str_ori != tag_pred:
        # 	print(tag_pred, str_ori)

        uttr_inp = ' '.join(total_eval[4][i])
        if uttr_inp not in dev_dials[dial_id]['inpUttr2orUttr']:
            print(dial_id, uttr_inp)
            print(dev_dials[dial_id]['inpUttr2orUttr'])
            exit()
        uttr_inp = dev_dials[dial_id]['inpUttr2orUttr'][uttr_inp]

        if dial_id not in results:
            results[dial_id] = {}
            exception[dial_id] = {}
        if uttr_inp not in results[dial_id]:
            results[dial_id][uttr_inp] = {}
            exception[dial_id][uttr_inp] = {}
        if api not in results[dial_id][uttr_inp]:
            results[dial_id][uttr_inp][api] = {}

        if str_ori not in exception[dial_id][uttr_inp]:
            results[dial_id][uttr_inp][api][slot] = [
                str_ori, (p1, p2 + 1),
                '%.4f' % (total_eval[-10][i])
            ]
            exception[dial_id][uttr_inp][str_ori] = [
                api, slot, total_eval[-10][i]
            ]
        else:
            if total_eval[-10][i] > exception[dial_id][uttr_inp][str_ori][2]:
                results[dial_id][uttr_inp][api][slot] = [
                    str_ori, (p1, p2 + 1),
                    '%.4f' % (total_eval[-10][i])
                ]

                del_info = exception[dial_id][uttr_inp][str_ori]
                del results[dial_id][uttr_inp][del_info[0]][del_info[1]]

                exception[dial_id][uttr_inp][str_ori] = [
                    api, slot, total_eval[-10][i]
                ]

    with open("nonCate_pred_0p99sampleDev_onTest.json", "w") as dump_f:
        json.dump(results,
                  dump_f,
                  sort_keys=True,
                  indent=4,
                  separators=(',', ':'))


def process_request(slots_categorical):

    print('#' * 20)
    print('derive results for slot tagging')
    print('#' * 20)

    fn_samples = []
    fp_samples = []

    tp_samples = []

    cateSlots_used_as_nonCate = 0

    slots_tag_results = {}
    uttr_tag = total_eval.pop(5)
    for i in range(len(total_eval[3])):

        slot_name = total_eval[5][i]

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
            if total_eval[-2][i] == 1:
                slots_tag_results[slot_name]['tn'] += 1
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
          related_api_in_train, related_files_in_train, total_eval[7][i], total_eval[8][i], \
           total_eval[9][i], '\n', total_eval[0][i], total_eval[1][i], total_eval[2][i],\
           total_eval[3][i], total_eval[5][i], total_eval[-2][i], total_eval[-1][i], '%.4f'%(total_eval[-11][i]), '\n')

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
          related_api_in_train, related_files_in_train, total_eval[7][i], total_eval[8][i], \
           total_eval[9][i], '\n', total_eval[0][i], total_eval[1][i], total_eval[2][i],\
           total_eval[3][i], total_eval[5][i], total_eval[-2][i], total_eval[-1][i], '%.4f'%(total_eval[-11][i]), '\n')

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
          related_api_in_train, related_files_in_train, total_eval[7][i], total_eval[8][i], \
           total_eval[9][i], '\n', total_eval[0][i], total_eval[1][i], total_eval[2][i],\
           total_eval[3][i], total_eval[5][i], total_eval[-2][i], total_eval[-1][i], '%.4f'%(total_eval[-11][i]), '\n')

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

    precision_micro = TP / (TP + FP + 1e-05)
    recall_micro = TP / (TP + FN + 1e-05)
    F1_micro = 2 * precision_micro * recall_micro / (precision_micro +
                                                     recall_micro + 1e-05)
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
    precision_macro = precision_macro / (num_slots_consider + 1e-05)
    recall_macro = recall_macro / (num_slots_consider + +1e-05)
    F1_macro = 2 * precision_macro * recall_macro / (precision_macro +
                                                     recall_macro + 1e-05)
    print('#Macro precision: %.4f, recall: %.4f, F1: %.4f' %
          (precision_macro, recall_macro, F1_macro))
    '''
	save formatted predicted results as json file
	'''
    results = {}
    for i in range(len(total_eval[3])):

        if total_eval[2][i] == 'SYSTEM':
            continue

        slot = total_eval[5][i]
        dial_id = total_eval[0][i]
        api = total_eval[1][i]
        # uttr = total_eval[3][i]
        uttr_inp = ' '.join(total_eval[4][i])
        if uttr_inp not in dev_dials[dial_id]['inpUttr2orUttr']:
            print(dial_id, uttr_inp)
            print(dev_dials[dial_id]['inpUttr2orUttr'])
            exit()

        uttr = dev_dials[dial_id]['inpUttr2orUttr'][uttr_inp]

        pred = total_eval[-1][i]
        if pred == 0:
            if dial_id not in results:
                results[dial_id] = {}
            if uttr not in results[dial_id]:
                results[dial_id][uttr] = {}
            if api not in results[dial_id][uttr]:
                results[dial_id][uttr][api] = {}
            results[dial_id][uttr][api][slot] = '%.4f' % (total_eval[-11][i])

    # with open("request_pred_0p99Dev_onTest.json","w") as dump_f:
    # 	json.dump(results, dump_f, sort_keys=True, indent=4, separators=(',', ':'))


def process_Cate(slots_categorical):

    print('#' * 20)
    print('derive results for slot tagging')
    print('#' * 20)

    tp_samples = []
    fn_samples = []
    fp_samples = []

    cateSlots_used_as_nonCate = 0

    slots_tag_results = {}
    uttr_tag = total_eval.pop(5)
    for i in range(len(total_eval[3])):

        slot_name = total_eval[5][i]

        if total_eval[2][i] == 'SYSTEM':
            continue

        if not slots_categorical[slot_name]:
            continue

        if slot_name not in slots_tag_results:
            slots_tag_results[slot_name] = {}
            slots_tag_results[slot_name]['tp'] = 0
            slots_tag_results[slot_name]['fn'] = 0
            slots_tag_results[slot_name]['tn'] = 0
            slots_tag_results[slot_name]['fp'] = 0
            slots_tag_results[slot_name]['p'] = 0
            slots_tag_results[slot_name]['r'] = 0
            slots_tag_results[slot_name]['f1'] = 0

        if total_eval[-8][i] == 1:
            if total_eval[-4][i] == total_eval[-3][i]:
                tp_samples.append(i)
                slots_tag_results[slot_name]['tp'] += 1
            else:
                slots_tag_results[slot_name]['fn'] += 1
                fn_samples.append(i)
        else:
            if total_eval[-4][i] == total_eval[-3][i]:
                slots_tag_results[slot_name]['tn'] += 1
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

        val_truth = ' '.join(total_eval[6][i][total_eval[-4][i]])
        val_pred = ' '.join(total_eval[6][i][total_eval[-3][i]])

        print('#tp samples: ', 'in train: %s'%str(total_eval[5][i] in slots_api_train), \
          related_api_in_train, related_files_in_train, total_eval[7][i], total_eval[8][i], \
           total_eval[9][i], '\n', total_eval[0][i], total_eval[1][i], total_eval[2][i],\
           total_eval[3][i], total_eval[5][i], [val_truth], [val_pred], '%.4f'%(total_eval[-9][i]), '\n')

    for i in fn_samples:
        slot = total_eval[5][i]
        related_api_in_train = []
        related_files_in_train = 0
        if slot in slots_api_train:
            related_api_in_train = slots_api_train[slot]
            for api in related_api_in_train:
                if api in api2dial_idx_train:
                    related_files_in_train += len(api2dial_idx_train[api])

        val_truth = ' '.join(total_eval[6][i][total_eval[-4][i]])
        val_pred = ' '.join(total_eval[6][i][total_eval[-3][i]])

        print('#fn samples: ', 'in train: %s'%str(total_eval[5][i] in slots_api_train), \
          related_api_in_train, related_files_in_train, total_eval[7][i], total_eval[8][i], \
           total_eval[9][i], '\n', total_eval[0][i], total_eval[1][i], total_eval[2][i],\
           total_eval[3][i], total_eval[5][i], [val_truth], [val_pred], '%.4f'%(total_eval[-9][i]), '\n')

    for i in fp_samples:
        slot = total_eval[5][i]
        related_api_in_train = []
        related_files_in_train = 0
        if slot in slots_api_train:
            related_api_in_train = slots_api_train[slot]
            for api in related_api_in_train:
                if api in api2dial_idx_train:
                    related_files_in_train += len(api2dial_idx_train[api])

        val_truth = ' '.join(total_eval[6][i][total_eval[-4][i]])
        val_pred = ' '.join(total_eval[6][i][total_eval[-3][i]])

        print('#fp samples: ', 'in train: %s'%str(total_eval[5][i] in slots_api_train), \
          related_api_in_train, related_files_in_train, total_eval[7][i], total_eval[8][i], \
           total_eval[9][i], '\n', total_eval[0][i], total_eval[1][i], total_eval[2][i],\
           total_eval[3][i], total_eval[5][i], [val_truth], [val_pred], '%.4f'%(total_eval[-9][i]), '\n')

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

    precision_micro = TP / (TP + FP)
    recall_micro = TP / (TP + FN + 1e-05)
    F1_micro = 2 * precision_micro * recall_micro / (precision_micro +
                                                     recall_micro + 1e-05)
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
                                                     recall_macro + 1e-05)
    print('#Macro precision: %.4f, recall: %.4f, F1: %.4f' %
          (precision_macro, recall_macro, F1_macro))
    '''
	save formatted predicted results as json file
	'''
    results = {}
    for i in range(len(total_eval[3])):
        slot_name = total_eval[5][i]
        if total_eval[2][i] == 'SYSTEM':
            continue

        if not slots_categorical[slot_name]:
            continue

        slot = total_eval[5][i]
        dial_id = total_eval[0][i]
        api = total_eval[1][i]
        uttr = total_eval[3][i]

        uttr_inp = ' '.join(total_eval[4][i])
        if uttr_inp not in dev_dials[dial_id]['inpUttr2orUttr']:
            print(dial_id, uttr_inp)
            print(dev_dials[dial_id]['inpUttr2orUttr'])
            exit()
        uttr_inp = dev_dials[dial_id]['inpUttr2orUttr'][uttr_inp]

        poss_vals = schemas_test[api]['slots'][slot]['possible_values']
        poss_vals = [[x] for x in poss_vals]
        total_eval[6][i][2:2 + len(poss_vals)] = poss_vals
        val_pred = ' '.join(total_eval[6][i][total_eval[-3][i]])

        if val_pred == 'do not care':
            val_pred = 'dontcare'

        if val_pred != "null" and val_pred != '[PAD]':
            # continue
            if dial_id not in results:
                results[dial_id] = {}
            if uttr_inp not in results[dial_id]:
                results[dial_id][uttr_inp] = {}
            if api not in results[dial_id][uttr_inp]:
                results[dial_id][uttr_inp][api] = {}
            results[dial_id][uttr_inp][api][slot] = [
                val_pred, '%.4f' % (total_eval[-9][i])
            ]
    # with open("cate_pred_0p99sampleDev_onTest.json","w") as dump_f:
    # 	json.dump(results, dump_f, sort_keys=True, indent=4, separators=(',', ':'))



schemas_test, slots_categorical_test, slots_api_test, intents_api_test = \
       load_shemas('../../dstc8-schema-guided-dialogue/dev0920/schema.json')

schemas_train, slots_categorical_train, slots_api_train, intents_api_train = \
       load_shemas('../../dstc8-schema-guided-dialogue/train/schema.json')

api2dial_idx_train = resolv_train_dial(
    '../../dstc8-schema-guided-dialogue/train/')

dev_dials = load_dev_dial('../../dstc8-schema-guided-dialogue/dev0920/')

total_eval = list(
    pkl.load(
        open('../detailed_results/2019-10-11_13_04_02_total_eval_results.pkl',
             'rb')))

# process_nonCate_span(slots_categorical_test)
process_request(slots_categorical_test)

# process_Cate(slots_categorical_test)
