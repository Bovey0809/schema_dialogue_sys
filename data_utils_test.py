import json
import nltk
import copy
import random
import os
from pytorch_pretrained_bert.tokenization import BertTokenizer

# tokenizer_bert = BertTokenizer.from_pretrained('../../../CommonsenseQA/scripts/pre_trained_models_base/', do_lower_case=True, cache_dir='../../../CommonsenseQA/scripts/pre_trained_models_base/')

# tokenizer_bert = BertTokenizer.from_pretrained('pre_trained_models_base_cased/', \
# 	do_lower_case=False, cache_dir='pre_trained_models_base_cased/')


class Corpus(object):

    def __init__(self,
                 args,
                 prepare_data_manner=None,
                 data_path='../dstc8-schema-guided-dialogue/',
                 max_uttr_len=64):

        self.data_path = data_path
        self.bert_model = args.bert_model
        self.max_uttr_len = max_uttr_len
        self.tokenizer_bert = BertTokenizer.from_pretrained(
            self.bert_model,
            do_lower_case=False,
            cache_dir=args.load_model_dir)

        self.max_numVals_of_slot = 0
        self.max_numIntents = 0

        self.train_schemas = self.load_shemas(self.data_path +
                                              'test/schema.json')
        self.dev_schemas = self.load_shemas(self.data_path +
                                            'test/schema.json')

        self.train_dials = self.load_dialogues(self.data_path + 'test/',
                                               self.train_schemas)
        self.dev_dials = self.load_dialogues(self.data_path + 'test/',
                                             self.dev_schemas)

    def mix_train_dev(self):
        for k in self.dev_schemas:
            if k not in self.train_schemas:
                self.train_schemas[k] = self.dev_schemas[k]

        for k in [k for k in self.dev_dials]:
            if int(k.split('_')[0]) > 7:
                self.train_dials[k + '_dev'] = copy.deepcopy(self.dev_dials[k])
                del self.dev_dials[k]

    def sample_dev(self, p=0.9):
        for k in self.dev_schemas:
            if k not in self.train_schemas:
                self.train_schemas[k] = self.dev_schemas[k]

        dev_keys = [k for k in self.dev_dials]
        random.shuffle(dev_keys)

        for k in dev_keys[:int(len(dev_keys) * p)]:
            self.train_dials[k + '_dev'] = copy.deepcopy(self.dev_dials[k])
            del self.dev_dials[k]

    def check_length(self, data, maxlen=64, idx=-4):
        small = 0
        big = 0
        for x in data:
            # if x[2] == 'SYSTEM':
            # 	continue
            if len(x[idx]) <= maxlen:
                small += 1
            else:
                big += 1

        print("<=%d: %.4f" % (maxlen, 1. * small / (small + big + 0.000001)))

    def get_all_set(self):
        train_data = self.prepate_data(self.train_dials, self.train_schemas)
        dev_data = self.prepate_data(self.dev_dials, self.dev_schemas)

        train = self.merge_dial(train_data)
        dev = self.merge_dial(dev_data)
        test = self.merge_filt(train_data, copy.deepcopy(dev_data))

        self.check_nonGoal(train, -5)
        self.check_nonGoal(dev, -5)
        self.check_nonGoal(test, -5)

        self.check_prior(train, 6)
        self.check_prior(dev, 6)
        self.check_prior(test, 6)

        return train, dev, test

    def get_slotCopy_set(self):
        train_data = self.prepare_copy_slot(self.train_dials,
                                            self.train_schemas)
        dev_data = self.prepare_copy_slot(self.dev_dials, self.dev_schemas)

        train = self.merge_dial(train_data)
        dev = self.merge_dial(dev_data)
        test = self.merge_filt(train_data, copy.deepcopy(dev_data))

        self.check_nonGoal(train, -2)
        self.check_nonGoal(dev, -2)
        self.check_nonGoal(test, -2)

        self.check_prior(train, -1)
        self.check_prior(dev, -1)
        self.check_prior(test, -1)

        return train, dev, test

    def get_slotCross_set(self):
        train_data = self.prepare_slot_cross_v1(self.train_dials,
                                                self.train_schemas)
        dev_data = self.prepare_slot_cross_v1(self.dev_dials, self.dev_schemas)

        train = self.merge_dial(train_data)
        dev = self.merge_dial(dev_data)
        test = self.merge_filt(train_data, copy.deepcopy(dev_data))

        self.check_nonGoal(train, -1)
        self.check_nonGoal(dev, -1)
        self.check_nonGoal(test, -1)

        self.check_prior(train, -2)
        self.check_prior(dev, -2)
        self.check_prior(test, -2)

        return train, dev, test

    def get_uttrCopy_set(self):
        train_data = self.prepare_copy_uttr(self.train_dials,
                                            self.train_schemas)
        dev_data = self.prepare_copy_uttr(self.dev_dials, self.dev_schemas)

        train = self.merge_dial(train_data)
        dev = self.merge_dial(dev_data)
        test = self.merge_filt(train_data, copy.deepcopy(dev_data))

        self.check_nonGoal(train, -1)
        self.check_nonGoal(dev, -1)
        self.check_nonGoal(test, -1)

        return train, dev, test

    def get_slotNotCare_set(self):
        train_data = self.prepare_slots_notCare(self.train_dials,
                                                self.train_schemas)
        dev_data = self.prepare_slots_notCare(self.dev_dials, self.dev_schemas)

        train = self.merge_dial(train_data)
        dev = self.merge_dial(dev_data)
        test = self.merge_filt(train_data, copy.deepcopy(dev_data))

        self.check_nonGoal(train, -1)
        self.check_nonGoal(dev, -1)
        self.check_nonGoal(test, -1)

        return train, dev, test

    def get_slotInIntent_set(self):
        train_data = self.prepare_slotsInIntent(self.train_dials,
                                                self.train_schemas)
        dev_data = self.prepare_slotsInIntent(self.dev_dials, self.dev_schemas)

        train = self.merge_dial(train_data)
        dev = self.merge_dial(dev_data)
        test = self.merge_filt(train_data, copy.deepcopy(dev_data))

        self.check_nonGoal(train, -1)
        self.check_nonGoal(dev, -1)
        self.check_nonGoal(test, -1)

        return train, dev, test

    def get_nonCate_set(self):
        train_data = self.prepare_nonCate_slot(self.train_dials,
                                               self.train_schemas)
        dev_data = self.prepare_nonCate_slot(self.dev_dials, self.dev_schemas)

        train = self.merge_dial(train_data)
        dev = self.merge_dial(dev_data)
        test = self.merge_filt(train_data, copy.deepcopy(dev_data))

        self.check_nonGoal_nonCate(train)
        self.check_nonGoal_nonCate(dev)
        self.check_nonGoal_nonCate(test)

        return train, dev, test

    def get_Cate_set(self):
        train_data = self.prepare_Cate_slot(self.train_dials,
                                            self.train_schemas)
        dev_data = self.prepare_Cate_slot(self.dev_dials, self.dev_schemas)

        train = self.merge_dial(train_data)
        dev = self.merge_dial(dev_data)
        test = self.merge_filt(train_data, copy.deepcopy(dev_data))

        self.check_nonGoal_cate(train)
        self.check_nonGoal_cate(dev)
        self.check_nonGoal_cate(test)

        return train, dev, test

    def get_request_set(self):
        train_data = self.prepare_request_slot(self.train_dials,
                                               self.train_schemas)
        dev_data = self.prepare_request_slot(self.dev_dials, self.dev_schemas)

        train = self.merge_dial(train_data)
        dev = self.merge_dial(dev_data)
        test = self.merge_filt(train_data, copy.deepcopy(dev_data))

        self.check_nonGoal_request(train)
        self.check_nonGoal_request(dev)
        self.check_nonGoal_request(test)

        return train, dev, test

    def get_intent_set(self):
        train_data = self.prepare_intents(self.train_dials, self.train_schemas)
        dev_data = self.prepare_intents(self.dev_dials, self.dev_schemas)

        train = self.merge_dial(train_data)
        dev = self.merge_dial(dev_data)
        test = self.merge_filt(train_data, copy.deepcopy(dev_data))

        return train, dev, test

    def check_prior(self, data, i):
        counts = {}
        for x in data:
            if str(x[i]) not in counts:
                counts[str(x[i])] = 1
            else:
                counts[str(x[i])] += 1
        for k in counts:
            counts[k] /= len(data)
        print(counts)

    def check_nonGoal(self, data, i):
        counts = {}
        for x in data:
            # if x[-3] not in [0,1,2,3]:
            # 	print('error', x[-3])
            if str(x[i]) not in counts:
                counts[str(x[i])] = 1
            else:
                counts[str(x[i])] += 1
        for k in counts:
            counts[k] /= len(data)
        print(counts)

    def check_nonGoal_cate(self, data):
        counts = {}
        for x in data:
            # if x[-3] not in [0,1,2,3]:
            # 	print('error', x[-3])
            if str(x[-3]) not in counts:
                counts[str(x[-3])] = 1
            else:
                counts[str(x[-3])] += 1
        for k in counts:
            counts[k] /= len(data)
        print(counts)

    def check_nonGoal_request(self, data):
        counts = {}
        for x in data:
            # if x[-3] not in [0,1,2,3]:
            # 	print('error', x[-3])
            if str(x[-2]) not in counts:
                counts[str(x[-2])] = 1
            else:
                counts[str(x[-2])] += 1
        for k in counts:
            counts[k] /= len(data)
        print(counts)

    def check_distribution(self, data):
        print('#' * 20)
        for key in data:
            print(key, len(data[key]))

    def resolv_schema(self, raw):
        resolved = {}
        resolved['service_name'] = raw['service_name']
        resolved['description'] = raw['description']
        resolved['slots'] = {}
        self.max_numIntents = max(len(raw['intents']), self.max_numIntents)
        for item in raw['slots']:
            item['possible_values_tokenized'] = [
                self.tokenizer_bert.tokenize(val)
                for val in item['possible_values']
            ]
            resolved['slots'][item['name']] = item
            resolved['slots'][item['name']][
                'description_tokenized'] = self.tokenizer_bert.tokenize(
                    resolved['slots'][item['name']]['description'])
            resolved['slots'][item['name']][
                'description_tokenized_simple'] = self.tokenizer_bert.tokenize(
                    item['name'].replace('_', ' '))

            numVals = len(item['possible_values'])
            self.max_numVals_of_slot = max(numVals, self.max_numVals_of_slot)

        resolved['intents'] = {}
        for item in raw['intents']:
            resolved['intents'][item['name']] = item
            resolved['intents'][item['name']][
                'description_tokenized'] = self.tokenizer_bert.tokenize(
                    resolved['intents'][item['name']]['description'])
            resolved['intents'][item['name']][
                'description_tokenized_simple'] = self.tokenizer_bert.tokenize(
                    item['name'])

        return resolved

    def load_shemas(self, path):

        fp = open(path, 'r')
        data = json.loads(fp.read())
        print('#number of schemas:', len(data))
        fp.close()

        schemas = {}
        for item in data:
            schemas[item['service_name']] = self.resolv_schema(item)

        return schemas

    def resolv_dial(self, raw):
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
                    temp_turn['frames'][
                        frame['service']]['state'] = frame['state']
                    temp_turn['frames'][
                        frame['service']]['service'] = frame['service']
                    temp_turn['frames'][frame['service']]['slots'] = {}

                    for slot in frame['slots']:
                        temp_turn['frames'][frame['service']]['slots'][
                            slot['slot']] = slot

                    slots_usr = set(
                        [key for key in frame['state']['slot_values']])
                    slots_usr_new = set([
                        key for key in temp_turn['frames'][frame['service']]
                        ['slots']
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

    def is_whitespace(self, c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def charIdx2wordIdx(self, utterance, p1, p2):
        utterance = list(utterance)
        utterance.insert(p2, ' ')
        utterance.insert(p1, ' ')
        utterance = ''.join(utterance)
        p1 += 1
        p2 += 1

        # utterance += ' '
        slot_string_ori = utterance[p1:p2]
        # utterance = nltk.word_tokenize(utterance)
        uttr_words = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in utterance:
            if self.is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    uttr_words.append(c)
                else:
                    uttr_words[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(uttr_words) - 1)

        if p2 >= len(char_to_word_offset):
            print(utterance, p2, len(char_to_word_offset), len(utterance))

        p1_word = char_to_word_offset[p1]
        p2_word = char_to_word_offset[p2]

        assert p1_word <= p2_word
        slot_string_temp = ' '.join(uttr_words[p1_word:p2_word + 1])
        '''
		word tokenization
		'''
        word_to_wordTokenized = {}
        uttr_words_tokenized = []
        tag_list_tokenized = []
        for idx, w in enumerate(uttr_words):
            words = nltk.word_tokenize(w)
            word_to_wordTokenized[idx] = list(range(len(uttr_words_tokenized), \
             len(uttr_words_tokenized)+len(words)))
            uttr_words_tokenized += words

            if idx in range(p1_word, p2_word + 1):
                tag_list_tokenized += [1] * len(words)
            else:
                tag_list_tokenized += [0] * len(words)

        p1_word_tokenized = word_to_wordTokenized[p1_word][0]
        p2_word_tokenized = word_to_wordTokenized[p2_word][-1]

        assert p1_word_tokenized <= p2_word_tokenized
        slot_string_tokenized = ' '.join(uttr_words_tokenized \
              [p1_word_tokenized:p2_word_tokenized+1])
        '''
		subword tokenization (BERT-base-lowercase)
		'''
        word_to_BertTokenized = {}
        uttr_words_BertTokenized = []
        tag_list_BertTokenized = []
        for idx, w in enumerate(uttr_words):
            words = self.tokenizer_bert.tokenize(w)
            word_to_BertTokenized[idx] = list(range(len(uttr_words_BertTokenized), \
             len(uttr_words_BertTokenized)+len(words)))
            uttr_words_BertTokenized += words

            if idx in range(p1_word, p2_word + 1):
                tag_list_BertTokenized += [1] * len(words)
            else:
                tag_list_BertTokenized += [0] * len(words)

        p1_word_BertTokenized = word_to_BertTokenized[p1_word][0]
        p2_word_BertTokenized = word_to_BertTokenized[p2_word][-1]

        assert p1_word_BertTokenized <= p2_word_BertTokenized
        slot_string_BertTokenized = ' '.join(uttr_words_BertTokenized \
         [p1_word_BertTokenized:p2_word_BertTokenized+1])

        return slot_string_ori, slot_string_temp, slot_string_tokenized, \
          slot_string_BertTokenized, tag_list_tokenized, \
          tag_list_BertTokenized

    # with seq_tag used in original BERT for Connl-2003-NER
    def charIdx2wordIdx_v1(self, utterance, p1, p2):
        utterance = list(utterance)
        utterance.insert(p2, ' ')
        utterance.insert(p1, ' ')
        utterance = ''.join(utterance)
        p1 += 1
        p2 += 1

        # utterance += ' '
        slot_string_ori = utterance[p1:p2]
        # utterance = nltk.word_tokenize(utterance)
        uttr_words = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in utterance:
            if self.is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    uttr_words.append(c)
                else:
                    uttr_words[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(uttr_words) - 1)

        if p2 >= len(char_to_word_offset):
            print(utterance, p2, len(char_to_word_offset), len(utterance))

        p1_word = char_to_word_offset[p1]
        p2_word = char_to_word_offset[p2]

        assert p1_word <= p2_word
        slot_string_temp = ' '.join(uttr_words[p1_word:p2_word + 1])
        '''
		word tokenization
		'''
        word_to_wordTokenized = {}
        uttr_words_tokenized = []
        tag_list_tokenized = []
        for idx, w in enumerate(uttr_words):
            words = nltk.word_tokenize(w)
            word_to_wordTokenized[idx] = list(range(len(uttr_words_tokenized), \
             len(uttr_words_tokenized)+len(words)))
            uttr_words_tokenized += words

            if idx in range(p1_word, p2_word + 1):
                tag_list_tokenized += [1] * len(words)
            else:
                tag_list_tokenized += [0] * len(words)

        p1_word_tokenized = word_to_wordTokenized[p1_word][0]
        p2_word_tokenized = word_to_wordTokenized[p2_word][-1]

        assert p1_word_tokenized <= p2_word_tokenized
        slot_string_tokenized = ' '.join(uttr_words_tokenized \
              [p1_word_tokenized:p2_word_tokenized+1])
        '''
		subword tokenization (BERT-base-lowercase)
		'''
        word_to_BertTokenized = {}
        uttr_words_BertTokenized = []
        tag_list_BertTokenized = []
        for idx, w in enumerate(uttr_words_tokenized):
            words = self.tokenizer_bert.tokenize(w)
            word_to_BertTokenized[idx] = list(range(len(uttr_words_BertTokenized), \
             len(uttr_words_BertTokenized)+len(words)))
            uttr_words_BertTokenized += words

            if idx in range(p1_word_tokenized, p2_word_tokenized + 1):
                tag_list_BertTokenized += [1] + [2] * (len(words) - 1)
            else:
                tag_list_BertTokenized += [0] + [2] * (len(words) - 1)

        p1_word_BertTokenized = word_to_BertTokenized[p1_word_tokenized][0]
        p2_word_BertTokenized = word_to_BertTokenized[p2_word_tokenized][-1]

        start = word_to_BertTokenized[p1_word_tokenized][0]
        end = word_to_BertTokenized[p2_word_tokenized][0]

        assert p1_word_BertTokenized <= p2_word_BertTokenized
        slot_string_BertTokenized = ' '.join(uttr_words_BertTokenized \
         [p1_word_BertTokenized:p2_word_BertTokenized+1])

        assert len(uttr_words_BertTokenized) == len(tag_list_BertTokenized)
        for t, l in zip(uttr_words_BertTokenized, tag_list_BertTokenized):
            if '##' in t:
                assert l == 2
            # else:
            # 	if l == 2:
            # 		print(uttr_words_tokenized)
            # 		print(uttr_words_BertTokenized)
            # 		print(tag_list_BertTokenized)
            # 	assert l != 2

        return slot_string_ori, slot_string_temp, slot_string_tokenized, \
          slot_string_BertTokenized, tag_list_tokenized, \
          tag_list_BertTokenized, uttr_words_BertTokenized, start, end

    def default_tags(self, utterance):
        uttr_words = []
        prev_is_whitespace = True
        for c in utterance:
            if self.is_whitespace(c):
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
        tag_list_BertTokenized = []
        for idx, w in enumerate(uttr_words_tokenized):
            words = self.tokenizer_bert.tokenize(w)
            word_to_BertTokenized[idx] = list(range(len(uttr_words_BertTokenized), \
             len(uttr_words_BertTokenized)+len(words)))
            uttr_words_BertTokenized += words

            tag_list_BertTokenized += [0] + [2] * (len(words) - 1)

        assert len(uttr_words_BertTokenized) == len(tag_list_BertTokenized)
        for t, l in zip(uttr_words_BertTokenized, tag_list_BertTokenized):
            if '##' in t:
                assert l == 2
            # else:
            # 	if l == 2:
            # 		print(uttr_words_tokenized)
            # 		print(uttr_words_BertTokenized)
            # 		print(tag_list_BertTokenized)
            # 	assert l != 2

        return tag_list_BertTokenized, uttr_words_BertTokenized

    def load_dialogues(self, path, schemas):
        total_slotsCopy = 0
        total_slotCross = 0
        total_slotCross_check = 0
        file_list = os.listdir(path)
        data = []
        for file in file_list:
            if 'dialogues' in file:
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
            dials[item['dialogue_id']] = self.resolv_dial(item)

        for dialogue_id in dials:
            _dial = dials[dialogue_id]

            # if _dial['services'][0] not in schemas:
            # 	print(_dial['services'][0], ' not in schemas! ', path, _dial['file_name'])
            # 	continue

            # _schema = schemas[_dial['services'][0]] # single_api

            # for check whethe a newer slot is categorical
            state_slots_history = {}
            frame_slots_history = {}
            for _service in _dial['services']:
                state_slots_history[_service] = {}
                frame_slots_history[_service] = {}

            # initialize last_sys_turn
            last_sys_turn = {}
            last_sys_slots = {}
            last_sys_turn['utterance'] = ''

            # initialize last_sys_turn
            sys_slot_provide = {}
            sys_request_slot = {}
            for _service in _dial['services']:
                sys_slot_provide[_service] = {}
                sys_request_slot[_service] = {}
            for turn in _dial['turns']:
                if turn['speaker'] == 'SYSTEM':
                    last_sys_turn = turn
                    turn['default_tags'], turn[
                        'uttr_tokenized'] = self.default_tags(
                            turn['utterance'])
                    for frame_id in turn['frames']:
                        _schema = schemas[frame_id]
                        sys_request_slot[frame_id] = {}
                        frame = turn['frames'][frame_id]

                        frame['slots_nonCate'] = copy.deepcopy(frame['slots'])
                        frame['slots_Cate'] = []

                        for act in frame['actions']:
                            if frame['actions'][act]['act'] in set(
                                ['OFFER', 'INFORM', 'CONFIRM']):
                                sys_slot_provide[frame_id][act] = (frame['actions'][act]['values'],\
                                     frame['actions'][act]['act'])
                                frame_slots_history[frame_id][act] = frame[
                                    'actions'][act]['values']
                                if act not in frame['slots_nonCate']:
                                    frame['slots_Cate'].append(act)
                            if frame['actions'][act]['act'] == 'REQUEST':
                                if len(frame['actions'][act]['values']) == 1:
                                    sys_slot_provide[frame_id][act] = (frame['actions'][act]['values'],\
                                        frame['actions'][act]['act'])
                                else:
                                    sys_request_slot[frame_id][act] = ''

                        frame['slots_request'] = copy.deepcopy(
                            sys_request_slot)

                        for slot_name in frame['slots_nonCate']:
                            slot = frame['slots_nonCate'][slot_name]
                            # if slot_name not in sys_slot_provide[frame_id]:
                            # 	print(_dial['file_name'])
                            # 	print(slot_name)
                            # 	print(turn['utterance'])
                            # slot_value_act = sys_slot_provide[frame_id][slot_name]
                            p1 = slot['start']
                            p2 = slot['exclusive_end']

                            slot_string_ori, \
                            slot_string_temp, \
                            slot_string_tokenized, \
                               slot_string_BertTokenized, \
                               tag_list_tokenized, \
                            tag_list_BertTokenized, \
                            uttr_words_BertTokenized, \
                            start, end = \
                              self.charIdx2wordIdx_v1(turn['utterance'], p1, p2)

                            # if slot_string_ori != slot_string_tokenized:
                            # 	print(slot_string_ori+' -> '+slot_string_temp+' -> '\
                            # 	 +slot_string_tokenized+' -> '+ slot_string_BertTokenized)

                            slot['slot_tags'] = tag_list_BertTokenized
                            slot['start'] = start
                            slot['end'] = end

                        frame['nonCate_slot_key_words'] = []
                        for key in frame['slots_nonCate']:
                            frame['nonCate_slot_key_words'] += key.split('_')
                        frame['nonCate_slot_key_words'] = set(
                            frame['nonCate_slot_key_words'])

                        frame['Cate_slot_key_words'] = []
                        for key in frame['slots_Cate']:
                            frame['Cate_slot_key_words'] += key.split('_')
                        frame['Cate_slot_key_words'] = set(
                            frame['Cate_slot_key_words'])

                        frame['request_slot_key_words'] = []
                        for key in frame['slots_request']:
                            frame['request_slot_key_words'] += key.split('_')
                        frame['request_slot_key_words'] = set(
                            frame['request_slot_key_words'])

                        frame['slot_key_words'] = frame['nonCate_slot_key_words'] | \
                           frame['Cate_slot_key_words'] | frame['request_slot_key_words']

                if turn['speaker'] == 'USER':
                    turn['uttr_with_sys'] = 'sys : ' +last_sys_turn['utterance'] + ' usr : ' \
                           + turn['utterance']

                    # turn['default_tags'], turn['uttr_tokenized'] = self.default_tags(turn['utterance'])
                    turn['default_tags'], turn[
                        'uttr_tokenized'] = self.default_tags(
                            turn['uttr_with_sys'])

                    turn['state_slots_history'] = copy.deepcopy(
                        state_slots_history)
                    turn['frame_slots_history'] = copy.deepcopy(
                        frame_slots_history)
                    turn['sys_request_slot'] = copy.deepcopy(sys_request_slot)
                    turn['sys_slot_provide'] = copy.deepcopy(sys_slot_provide)
                    for frame_id in turn['frames']:
                        _schema = schemas[frame_id]
                        frame = turn['frames'][frame_id]
                        frame['slots_nonCate'] = copy.deepcopy(frame['slots'])
                        # update slots_nonCate from self-user_turn
                        for slot_name in frame['slots_nonCate']:
                            slot = frame['slots_nonCate'][slot_name]
                            slot['from_sys'] = False

                        state_slots = set(
                            [key for key in frame['state']['slot_values']])
                        nonCate_slots = set([key for key in frame['slots']])
                        # slots_tmp = state_slots - state_slots_history[frame_id] - nonCate_slots

                        slots_tmp = []
                        slots_cross_api = {}

                        for k in list(state_slots - nonCate_slots):
                            if k not in frame_slots_history[
                                    frame_id] and not _schema['slots'][k][
                                        'is_categorical']:
                                copy_flag = False

                                for frame_id_pre in frame_slots_history:
                                    if frame_id_pre == frame_id:
                                        continue

                                    for k_pre in frame_slots_history[
                                            frame_id_pre]:
                                        if len(
                                                set(frame['state']
                                                    ['slot_values'][k])
                                                & set(frame_slots_history[
                                                    frame_id_pre][k_pre])) > 0:
                                            if k not in slots_cross_api:
                                                slots_cross_api[k] = {}
                                            slots_cross_api[k][frame_id_pre,
                                                               k_pre] = ''
                                            print(
                                                '### slot cross-copied from history!'
                                            )
                                            print(_dial['file_name'])
                                            print(k, frame_id_pre, k_pre)
                                            print(turn['utterance'])
                                            copy_flag = True
                                            total_slotCross += 1
                                            continue
                                    if copy_flag:
                                        continue
                                if copy_flag:
                                    continue

                            if k not in state_slots_history[frame_id]:
                                slots_tmp.append(k)
                                continue
                            if len(
                                    set(frame['state']['slot_values'][k])
                                    & set(state_slots_history[frame_id][k])
                            ) == 0:
                                slots_tmp.append(k)

                        # if len(state_slots_history[frame_id] - state_slots)>0:
                        # 	print('###history slots not in states!')
                        # 	print(_dial['file_name'])
                        # 	print(turn['utterance'])

                        cate_slots = {}
                        slots_notCare = {}
                        for key in slots_tmp:
                            slot_values = frame['state']['slot_values'][key]
                            if slot_values[0] == 'dontcare':
                                slots_notCare[key] = 'dontcare'
                                print('### slot is dontcare!')
                                print(_dial['file_name'])
                                print(key, slot_values)
                                print(turn['utterance'])

                            if _schema['slots'][key]['is_categorical']:
                                cate_slots[key] = {}
                                if key in sys_slot_provide[frame_id]:
                                    if len(set(slot_values) & set(sys_slot_provide[frame_id][key][0]))>0 \
                                       and sys_slot_provide[frame_id][key][1] != 'REQUEST':

                                        total_slotsCopy += 1
                                        cate_slots[key]['from_sys'] = True
                                        print(
                                            '### categorical slot from sys action!',
                                            sys_slot_provide[frame_id][key][1])
                                        print(_dial['file_name'])
                                        print(
                                            key,
                                            frame['state']['slot_values'][key])
                                        print(turn['utterance'])

                                    else:
                                        cate_slots[key]['from_sys'] = False
                                else:
                                    cate_slots[key]['from_sys'] = False

                            else:
                                if key in sys_slot_provide[frame_id]:
                                    if len(
                                            set(slot_values)
                                            & set(sys_slot_provide[frame_id]
                                                  [key][0])) > 0:
                                        total_slotsCopy += 1
                                        frame['slots_nonCate'][key] = {}
                                        frame['slots_nonCate'][key][
                                            'from_sys'] = True
                                        print(
                                            '### non_categorical slot from sys action!',
                                            sys_slot_provide[frame_id][key][1])
                                        print(_dial['file_name'])
                                        print(
                                            key,
                                            frame['state']['slot_values'][key])
                                        print(turn['utterance'])
                                else:
                                    print(
                                        '### missing non_categorical slot, not from usr or sys!'
                                    )
                                    print(_dial['file_name'])
                                    print(key,
                                          frame['state']['slot_values'][key])
                                    print(turn['utterance'])

                        state_slots_history[frame_id] = copy.deepcopy(
                            frame['state']['slot_values'])
                        for k in state_slots_history[frame_id]:
                            frame_slots_history[frame_id][k] = copy.deepcopy(
                                state_slots_history[frame_id][k])

                        turn['frames'][frame_id]['cate_slots'] = copy.deepcopy(
                            cate_slots)
                        turn['frames'][frame_id][
                            'slots_notCare'] = copy.deepcopy(slots_notCare)
                        turn['frames'][frame_id][
                            'slots_cross_api'] = copy.deepcopy(slots_cross_api)
                        for k in slots_cross_api:
                            total_slotCross_check += len(slots_cross_api[k])

                        for slot_name in frame['slots_nonCate']:
                            slot = frame['slots_nonCate'][slot_name]
                            # assert len(frame['state']['slot_values'][slot['slot']]) == 1
                            # if len(frame['state']['slot_values'][slot['slot']]) > 1:
                            # 	print(frame['state']['slot_values'][slot['slot']])
                            if slot['from_sys']:
                                continue

                            tagged_slot_not_in_state = False
                            if slot['slot'] not in frame['state'][
                                    'slot_values']:
                                print('###tagged_slot_not_in_state')
                                print(_dial['file_name'])
                                print(slot['slot'])
                                print(turn['utterance'])
                                tagged_slot_not_in_state = True

                            if not tagged_slot_not_in_state:
                                slot_value = frame['state']['slot_values'][
                                    slot['slot']]
                                slot_value = [
                                    key.lower() for key in slot_value
                                ]

                            pos_offset = len('sys : ' +
                                             last_sys_turn['utterance'] +
                                             ' usr : ')

                            p1 = slot['start'] + pos_offset
                            p2 = slot['exclusive_end'] + pos_offset

                            slot_string_ori, \
                            slot_string_temp, \
                            slot_string_tokenized, \
                               slot_string_BertTokenized, \
                               tag_list_tokenized, \
                            tag_list_BertTokenized, \
                            uttr_words_BertTokenized, \
                            start, end = \
                              self.charIdx2wordIdx_v1(turn['uttr_with_sys'], p1, p2)

                            tagged_slot_not_same_with_stateValue = False
                            if slot_string_ori.lower() not in slot_value and (
                                    not tagged_slot_not_in_state):
                                print(
                                    '###tagged_slot_not_same_with_stateValue')
                                print(_dial['file_name'])
                                print(slot_value, slot_string_ori)
                                print(turn['utterance'])
                                tagged_slot_not_same_with_stateValue = True

                            # if slot_string_ori != slot_string_tokenized:
                            # 	print(slot_string_ori+' -> '+slot_string_temp+' -> '\
                            # 	 +slot_string_tokenized+' -> '+ slot_string_BertTokenized)

                            slot['slot_tags'] = tag_list_BertTokenized
                            # print(turn['uttr_with_sys_tokenized'])
                            # print(tag_list_BertTokenized)
                            slot['start'] = start
                            slot['end'] = end

                        frame['nonCate_slot_key_words'] = []
                        for key in frame['slots_nonCate']:
                            frame['nonCate_slot_key_words'] += key.split('_')
                        frame['nonCate_slot_key_words'] = set(
                            frame['nonCate_slot_key_words'])

                        frame['Cate_slot_key_words'] = []
                        for key in frame['cate_slots']:
                            frame['Cate_slot_key_words'] += key.split('_')
                        frame['Cate_slot_key_words'] = set(
                            frame['Cate_slot_key_words'])

                        frame['request_slot_key_words'] = []
                        for key in frame['state']['requested_slots']:
                            frame['request_slot_key_words'] += key.split('_')
                        frame['request_slot_key_words'] = set(
                            frame['request_slot_key_words'])

                        frame['slot_key_words'] = frame['nonCate_slot_key_words'] | \
                         frame['Cate_slot_key_words'] | frame['request_slot_key_words']

        return dials

    def prepate_data(self, dials, schemas):
        data = {}
        for dialogue_id in dials:
            _dial = dials[dialogue_id]

            if _dial['services'][0] not in schemas:
                print(_dial['services'][0], ' not in schemas! ', path)
                continue

            # _schema = schemas[_dial['services'][0]] # single_api
            _data = []

            for _turn in _dial['turns']:
                if _turn['speaker'] == 'SYSTEM':
                    continue

                if _turn['speaker'] == 'USER':
                    uttr_tokenized = _turn['uttr_tokenized']
                    for frame_id in _turn['frames']:
                        _schema = schemas[frame_id]
                        _frame = _turn['frames'][frame_id]
                        for slot_name in _schema['slots']:

                            slot_prior = True
                            slot_info = _schema['slots'][slot_name]
                            # slot_desp_tokenized = slot_info['description_tokenized']
                            # slot_desp_tokenized = slot_info['description_tokenized_simple']
                            slot_desp_tokenized = slot_info['description_tokenized_simple'] + ['#']\
                                  + slot_info['description_tokenized']
                            slot_categorical = _schema['slots'][slot_name][
                                'is_categorical']

                            slot_in_sys_his = slot_name in _turn[
                                'sys_slot_provide'][frame_id]
                            # if slot_categorical:
                            # 	slot_in_sys_his = slot_name in _turn['sys_slot_provide'][frame_id]
                            # else:
                            # 	slot_in_sys_his = False

                            slot_requested_by_sys = slot_name in _turn[
                                'sys_request_slot'][frame_id]
                            slot_in_usr_his = None
                            cateVal_idx = 0
                            tag_list_tokenized = _turn['default_tags']
                            start = len(tag_list_tokenized)
                            end = len(tag_list_tokenized)
                            slot_type = 3

                            _data.append([dialogue_id, frame_id,  _turn['speaker'], _turn['utterance'], \
                              slot_name, slot_categorical, slot_prior,\
                              slot_in_usr_his, slot_in_sys_his, slot_requested_by_sys,\
                              uttr_tokenized, slot_desp_tokenized, slot_info['possible_values_tokenized'],\
                              slot_type, tag_list_tokenized, start, end, cateVal_idx])

            api_service = _dial['services'][0]
            if api_service not in data:
                data[api_service] = {}
            data[api_service][dialogue_id] = _data

        return data

    # intents
    def prepare_intents(self, dials, schemas):

        data = {}
        for dialogue_id in dials:
            _dial = dials[dialogue_id]

            # if _dial['services'][0] not in schemas:
            # 	print(_dial['services'][0], ' not in schemas! ', path)
            # 	continue

            # _schema = schemas[_dial['services'][0]] # single_api
            _data = []
            last_intent = {}
            last_frames = {}
            for frame_id in _dial['services']:
                last_intent[frame_id] = ''

            intents = {}
            intents_desp = {}
            for frame_id in _dial['services']:
                _schema = schemas[frame_id]
                intents[frame_id] = []
                intents_desp[frame_id] = []
                for intent in _schema['intents']:
                    intents[frame_id].append(intent)
                    intents_desp[frame_id].append(_schema['intents'][intent]['description_tokenized'] \
                     + ['#'] + _schema['intents'][intent]['description_tokenized_simple'])
                intents_desp[frame_id].append(['none'])
                intents[frame_id].append('none')

            turn_id = 0
            for _turn in _dial['turns']:
                if _turn['speaker'] == 'SYSTEM':
                    continue

                if _turn['speaker'] == 'USER':
                    frames = {}
                    for frame_id in _turn['frames']:
                        frames[frame_id] = ''
                    if frames == last_frames:
                        uttr_tokenized = ['keep', '[SEP]'
                                          ] + _turn['uttr_tokenized']
                    else:
                        uttr_tokenized = ['jump', '[SEP]'
                                          ] + _turn['uttr_tokenized']
                    last_frames = copy.deepcopy(frames)

                    for frame_id in _turn['frames']:
                        _schema = schemas[frame_id]
                        frames[frame_id] = ''

                        _frame = _turn['frames'][frame_id]

                        last_intent_tag = [0] * len(intents[frame_id])
                        intent_idx = 0

                        # active_intent = _frame['state']['active_intent']
                        # if active_intent == 'NONE':
                        # 	active_intent = 'none'
                        # intent_idx = intents[frame_id].index(active_intent)

                        # last_intent_tag = [0]*len(intents[frame_id])
                        # if last_intent[frame_id] in intents[frame_id]:
                        # 	last_intent_tag[intents[frame_id].index(last_intent[frame_id])] = 1
                        # last_intent[frame_id] = active_intent

                        _data.append([dialogue_id, frame_id, turn_id, _turn['utterance'], uttr_tokenized,\
                          intents[frame_id], intents_desp[frame_id], intent_idx, last_intent_tag])

                        turn_id += 1

            api_service = _dial['services'][0]
            if api_service not in data:
                data[api_service] = {}
            data[api_service][dialogue_id] = _data

        return data

    # slots from sys
    def prepare_copy_slot(self, dials, schemas):

        data = {}
        total_slotsCopy = 0

        for dialogue_id in dials:
            _dial = dials[dialogue_id]

            _data = []

            for _turn in _dial['turns']:
                if _turn['speaker'] == 'SYSTEM':
                    continue

                if _turn['speaker'] == 'USER':

                    # uttr_tokenized = self.tokenizer_bert.tokenize(_turn['utterance'])
                    for frame_id in _turn['frames']:
                        _schema = schemas[frame_id]
                        _frame = _turn['frames'][frame_id]

                        try:
                            intent_info = _schema['intents'][_frame['state']
                                                             ['active_intent']]
                        except KeyError:
                            intent_info = {
                                'required_slots': {},
                                'optional_slots': {},
                                'description_tokenized': []
                            }

                        api_desp = self.tokenizer_bert.tokenize(
                            'api: ' + _schema['description'].lower())
                        uttr_tokenized = api_desp + [
                            '[SEP]'
                        ] + _turn['uttr_tokenized']

                        for slot_name in _schema['slots']:

                            if slot_name in _turn['sys_slot_provide'][
                                    frame_id]:
                                if _turn['sys_slot_provide'][frame_id][
                                        slot_name][1] == 'INFORM':
                                    continue

                            if slot_name not in _turn['sys_slot_provide'][
                                    frame_id]:
                                continue

                            slot_required = slot_name in intent_info[
                                'required_slots']
                            slot_optional = slot_name in intent_info[
                                'optional_slots']
                            slot_in_sys_his = slot_name in _turn[
                                'sys_slot_provide'][frame_id]
                            slot_in_usr_his = slot_name in _turn[
                                'state_slots_history'][frame_id]

                            if len(
                                    set(slot_name.split('_'))
                                    & _frame['slot_key_words']) > 2:
                                slot_prior = True
                            else:
                                slot_prior = False

                            slot_info = _schema['slots'][slot_name]
                            # slot_desp_tokenized = slot_info['description_tokenized_simple'] + ['[SEP]']\
                            # 						+ slot_info['description_tokenized']
                            slot_desp_tokenized = slot_info[
                                'description_tokenized']

                            if slot_name in _frame['cate_slots']:

                                if _frame['cate_slots'][slot_name]['from_sys']:
                                    if _turn['sys_slot_provide'][frame_id][
                                            slot_name][1] == 'INFORM':
                                        continue
                                    slot_type = 0
                                    assert slot_in_sys_his == True
                                    total_slotsCopy += 1
                                else:
                                    # assert slot_in_usr_his == True
                                    slot_type = 1
                            elif slot_name in _frame['slots_nonCate']:
                                if _frame['slots_nonCate'][slot_name][
                                        'from_sys']:
                                    if _turn['sys_slot_provide'][frame_id][
                                            slot_name][1] == 'INFORM':
                                        continue
                                    slot_type = 0
                                    assert slot_in_sys_his == True
                                    total_slotsCopy += 1
                                else:
                                    # assert slot_in_usr_his == True
                                    slot_type = 1
                            else:
                                # assert slot_in_usr_his == True
                                slot_type = 1

                            _data.append([dialogue_id, frame_id,  _turn['speaker'], _turn['utterance'], \
                               slot_name, slot_in_sys_his, slot_in_usr_his, \
                               slot_required, slot_optional,\
                               uttr_tokenized, slot_desp_tokenized,\
                               slot_type, slot_prior])

            api_service = _dial['services'][0]
            if api_service not in data:
                data[api_service] = {}
            data[api_service][dialogue_id] = _data

        print(total_slotsCopy)

        return data

    # slots from sys
    def prepare_slot_cross(self, dials, schemas):

        data = {}
        total_slotsCopy = 0
        total_slotsCopy_check = 0
        for dialogue_id in dials:
            _dial = dials[dialogue_id]
            apis = _dial['services']
            _data = []

            for _turn in _dial['turns']:
                if _turn['speaker'] == 'SYSTEM':
                    continue

                if _turn['speaker'] == 'USER':

                    # uttr_tokenized = self.tokenizer_bert.tokenize(_turn['utterance'])
                    for frame_id in _turn['frames']:
                        _schema = schemas[frame_id]
                        _frame = _turn['frames'][frame_id]
                        total_slotsCopy_check += len(_frame['slots_cross_api'])

                        try:
                            intent_info = _schema['intents'][_frame['state']
                                                             ['active_intent']]
                        except KeyError:
                            intent_info = {
                                'required_slots': {},
                                'optional_slots': {},
                                'description_tokenized': []
                            }

                        api_desp = self.tokenizer_bert.tokenize(
                            'api: ' + _schema['description'].lower())
                        uttr_tokenized = _turn['uttr_tokenized']

                        for slot_name in _schema['slots']:
                            slot_info = _schema['slots'][slot_name]
                            slot_desp_tokenized = slot_info[
                                'description_tokenized']
                            slot_required = slot_name in intent_info[
                                'required_slots']
                            slot_optional = slot_name in intent_info[
                                'optional_slots']
                            slot_in_frame_his = slot_name in _turn[
                                'frame_slots_history'][frame_id]
                            sample_tag = 1
                            if len(
                                    set(slot_name.split('_'))
                                    & _frame['slot_key_words']) > 2:
                                slot_prior = True
                            else:
                                slot_prior = False

                            for cross_frame_id in apis:
                                if cross_frame_id == frame_id:
                                    continue
                                sample_tag = 1
                                for slot_cross in schemas[cross_frame_id][
                                        'slots']:
                                    if slot_cross not in _turn[
                                            'frame_slots_history'][
                                                cross_frame_id]:
                                        continue
                                    sample_tag = 1
                                    slot_in_cross_frame_his = slot_cross in _turn[
                                        'frame_slots_history'][cross_frame_id]
                                    slot_cross_desp_tokenized = schemas[
                                        cross_frame_id]['slots'][slot_cross][
                                            'description_tokenized']
                                    if slot_name in _frame['slots_cross_api']:
                                        # print(_frame['slots_cross_api'][slot_name])
                                        # exit()
                                        if (cross_frame_id, slot_cross
                                            ) in _frame['slots_cross_api'][
                                                slot_name]:
                                            sample_tag = 0
                                            total_slotsCopy += 1
                                            # print(dialogue_id, _turn['utterance'], slot_name, cross_frame_id, slot_cross)

                                    _data.append([dialogue_id, frame_id, cross_frame_id, _turn['speaker'], _turn['utterance'], \
                                       slot_name, slot_cross, \
                                       slot_in_cross_frame_his, slot_in_frame_his, \
                                       slot_required, slot_optional,\
                                       uttr_tokenized, slot_desp_tokenized, slot_cross_desp_tokenized,\
                                       slot_prior, sample_tag])

            api_service = _dial['services'][0]
            if api_service not in data:
                data[api_service] = {}
            data[api_service][dialogue_id] = _data

        print(total_slotsCopy, total_slotsCopy_check)

        return data

    # slots from sys
    def prepare_slot_cross_v1(self, dials, schemas):

        data = {}
        total_slotsCopy = 0
        total_slotsCopy_check = 0
        for dialogue_id in dials:
            _dial = dials[dialogue_id]
            apis = _dial['services']
            _data = []

            last_frames = {}
            frame_success = {}
            for api in apis:
                frame_success[api] = False

            for _turn in _dial['turns']:
                if _turn['speaker'] == 'SYSTEM':
                    for frame_id in _turn['frames']:
                        _frame = _turn['frames'][frame_id]
                        if 'NOTIFY_SUCCESS' in [
                                _frame['actions'][act]['act']
                                for act in _frame['actions']
                        ]:
                            frame_success[frame_id] = True

                    continue

                if _turn['speaker'] == 'USER':

                    # uttr_tokenized = self.tokenizer_bert.tokenize(_turn['utterance'])
                    for frame_id in _turn['frames']:
                        if frame_id in last_frames:
                            frame_continue = True
                        else:
                            frame_continue = False
                        _schema = schemas[frame_id]
                        _frame = _turn['frames'][frame_id]
                        total_slotsCopy_check += len(_frame['slots_cross_api'])

                        try:
                            intent_info = _schema['intents'][_frame['state']
                                                             ['active_intent']]
                        except KeyError:
                            intent_info = {
                                'required_slots': {},
                                'optional_slots': {},
                                'description_tokenized': []
                            }

                        api_desp = self.tokenizer_bert.tokenize(
                            frame_id.split('_')[0].lower())
                        uttr_tokenized = _turn['uttr_tokenized']

                        for slot_name in _schema['slots']:
                            slot_info = _schema['slots'][slot_name]
                            slot_desp_tokenized = slot_info[
                                'description_tokenized_simple'] + [
                                    '#'
                                ] + slot_info['description_tokenized']
                            # slot_desp_tokenized = slot_info['description_tokenized_simple'] + ['#'] + api_desp
                            slot_required = slot_name in intent_info[
                                'required_slots']
                            slot_optional = slot_name in intent_info[
                                'optional_slots']
                            slot_in_frame_his = slot_name in _turn[
                                'frame_slots_history'][frame_id]
                            sample_tag = 1

                            slot_prior = False
                            if len(
                                    set(slot_name.split('_'))
                                    & _frame['slot_key_words']) > 1:
                                slot_prior = True
                            # else:
                            # 	slot_prior = False

                            for cross_frame_id in apis:
                                if cross_frame_id == frame_id:
                                    continue
                                cross_api_desp = self.tokenizer_bert.tokenize(
                                    cross_frame_id.split('_')[0].lower())
                                sample_tag = 1
                                for slot_cross in schemas[cross_frame_id][
                                        'slots']:
                                    if slot_cross not in _turn[
                                            'frame_slots_history'][
                                                cross_frame_id]:
                                        continue
                                    sample_tag = 1
                                    slot_in_cross_frame_his = slot_cross in _turn[
                                        'frame_slots_history'][cross_frame_id]
                                    slot_cross_desp_tokenized = \
                                      schemas[cross_frame_id]['slots'][slot_cross]['description_tokenized_simple'] \
                                      + ['#'] + schemas[cross_frame_id]['slots'][slot_cross]['description_tokenized']

                                    slot_cross_prior = False

                                    if slot_name in _frame['slots_cross_api']:
                                        key_words = []
                                        for x in _frame['slots_cross_api'][
                                                slot_name]:
                                            key_words += x[1].split('_')

                                        if len(
                                                set(slot_cross.split('_'))
                                                & set(key_words)) > 0:
                                            slot_cross_prior = True

                                        if (cross_frame_id, slot_cross
                                            ) in _frame['slots_cross_api'][
                                                slot_name]:
                                            sample_tag = 0
                                            total_slotsCopy += 1
                                            # print(dialogue_id, _turn['utterance'], slot_name, cross_frame_id, slot_cross)

                                    _data.append([dialogue_id, frame_id, cross_frame_id, _turn['speaker'], _turn['utterance'], \
                                       slot_name, slot_cross, \
                                       slot_in_cross_frame_his, slot_in_frame_his, \
                                       slot_required, slot_optional, frame_success[cross_frame_id], \
                                       uttr_tokenized, slot_desp_tokenized, slot_cross_desp_tokenized,\
                                       slot_prior or slot_cross_prior, sample_tag])

                    last_frames = {}
                    for frame_id in _turn['frames']:
                        last_frames[frame_id] = ''

            api_service = _dial['services'][0]
            if api_service not in data:
                data[api_service] = {}
            data[api_service][dialogue_id] = _data

        print(total_slotsCopy, total_slotsCopy_check)

        return data

    # uttr that can copy slots from sys
    def prepare_copy_uttr(self, dials, schemas):

        data = {}

        for dialogue_id in dials:
            _dial = dials[dialogue_id]

            if _dial['services'][0] not in schemas:
                print(_dial['services'][0], ' not in schemas! ', path)
                continue

            api = _dial['services'][0]  # single_api
            api_desp = self.tokenizer_bert.tokenize(api.lower().replace(
                '_', ' '))
            _data = []

            for _turn in _dial['turns']:
                if _turn['speaker'] == 'SYSTEM':
                    continue

                if _turn['speaker'] == 'USER':
                    uttr_tokenized = api_desp + ['#'] + _turn['uttr_tokenized']
                    # uttr_tokenized = self.tokenizer_bert.tokenize(_turn['utterance'])
                    for frame_id in _turn['frames']:
                        _schema = schemas[frame_id]
                        _frame = _turn['frames'][frame_id]
                        state_slots = dict(list(_frame['cate_slots'].items()) + \
                         list(_frame['slots_nonCate'].items()))

                        uttr_label = 1
                        for slot_name in state_slots:
                            if state_slots[slot_name]['from_sys']:
                                uttr_label = 0
                                break

                        _data.append([dialogue_id,  _turn['speaker'], _turn['utterance'], \
                           uttr_tokenized, uttr_label])

            api_service = _dial['services'][0]
            if api_service not in data:
                data[api_service] = {}
            data[api_service][dialogue_id] = _data

        return data

    # uttr that can copy slots from sys
    def prepare_slots_notCare(self, dials, schemas):

        data = {}

        for dialogue_id in dials:
            _dial = dials[dialogue_id]

            if _dial['services'][0] not in schemas:
                print(_dial['services'][0], ' not in schemas! ', path)
                continue

            _data = []

            for _turn in _dial['turns']:
                if _turn['speaker'] == 'SYSTEM':
                    continue

                if _turn['speaker'] == 'USER':
                    uttr_tokenized = _turn['uttr_tokenized']
                    # uttr_tokenized = self.tokenizer_bert.tokenize(_turn['utterance'])
                    for frame_id in _turn['frames']:
                        _schema = schemas[frame_id]
                        _frame = _turn['frames'][frame_id]

                        for slot_name in _schema['slots']:
                            if slot_name in _frame['slots_notCare']:
                                slot_label = 0
                            else:
                                slot_label = 1

                            slot_info = _schema['slots'][slot_name]
                            slot_desp_tokenized = slot_info['description_tokenized_simple'] + ['[SEP]']\
                                  + slot_info['description_tokenized']


                            _data.append([dialogue_id, _turn['speaker'], _turn['utterance'], slot_name, \
                              uttr_tokenized, slot_desp_tokenized, slot_label])

            api_service = _dial['services'][0]
            if api_service not in data:
                data[api_service] = {}
            data[api_service][dialogue_id] = _data

        return data

    # slots should appear in Intent
    def prepare_slotsInIntent(self, dials, schemas):

        data = {}

        for dialogue_id in dials:
            _dial = dials[dialogue_id]

            if _dial['services'][0] not in schemas:
                print(_dial['services'][0], ' not in schemas! ', path)
                continue

            _schema = schemas[_dial['services'][0]]  # single_api
            intent2slot = {}
            for api in _dial['services']:
                _schema = schemas[api]
                intent2slot[api] = {}
                for intent in _schema['intents']:
                    intent2slot[api][intent] = []

            _data = []

            for _turn in _dial['turns']:
                if _turn['speaker'] == 'SYSTEM':
                    continue

                if _turn['speaker'] == 'USER':
                    # uttr_tokenized = self.tokenizer_bert.tokenize(_turn['utterance'])
                    for frame_id in _turn['frames']:
                        _schema = schemas[frame_id]
                        _frame = _turn['frames'][frame_id]
                        intent = _frame['state']['active_intent']
                        if intent == 'NONE':
                            continue
                        slots = [k for k in _frame['state']['slot_values']]
                        if len(slots) < len(intent2slot[frame_id][intent]):
                            print('###Abnormal: ', dialogue_id,
                                  _turn['utterance'])
                        intent2slot[frame_id][intent] = slots

            for api in intent2slot:
                _schema = schemas[api]

                api_desp = self.tokenizer_bert.tokenize(api.lower().replace(
                    '_', ' '))
                for intent in intent2slot[api]:
                    try:
                        intent_info = _schema['intents'][intent]
                    except KeyError:
                        intent_info = {
                            'required_slots': {},
                            'optional_slots': {},
                            'description_tokenized': []
                        }

                    intent_desp = api_desp + [
                        '#'
                    ] + intent_info['description_tokenized']
                    for slot in _schema['slots']:
                        slot_desp = _schema['slots'][slot]['description_tokenized_simple'] + \
                         ['#'] + _schema['slots'][slot]['description_tokenized']

                        if slot in intent2slot[api][intent]:
                            slot_label = 0
                        else:
                            slot_label = 1

                        slot_required = slot in intent_info['required_slots']
                        slot_optional = slot in intent_info['optional_slots']

                        _data.append([dialogue_id, api, intent_desp, slot_desp, intent, slot,\
                            slot_required, slot_optional, slot_label])

            api_service = _dial['services'][0]
            if api_service not in data:
                data[api_service] = {}
            data[api_service][dialogue_id] = _data

        return data

    def split_by_dialID(self, data):
        merge = []
        for api_service in data:
            for dialogue_id in data[api_service]:
                merge.append(data[api_service][dialogue_id])

        # print('#total dials:', len(merge))

        random.shuffle(merge)
        train = []
        dev = []
        for i, dial in enumerate(merge):
            if i < 0.8 * len(merge):
                train += dial
            else:
                dev += dial

        return train, dev

    def merge_dial(self, data):
        merge = []
        for api_service in data:
            for dialogue_id in data[api_service]:
                merge += data[api_service][dialogue_id]
        random.shuffle(merge)

        return merge

    def merge_filt(self, data_out, data):
        #TODO what's the function doing?
        apis = {}
        for api_service in data_out:
            apis[api_service] = 1

        merge = []
        for api_service in data:
            if api_service in apis:
                continue
            for dialogue_id in data[api_service]:
                merge += data[api_service][dialogue_id]
        random.shuffle(merge)

        return merge

    def split_by_domain(self, data):
        train = []
        dev = []
        for api_service in data:
            if 'movies' not in api_service.lower():
                for dialogue_id in data[api_service]:
                    train += data[api_service][dialogue_id]
            else:
                for dialogue_id in data[api_service]:
                    dev += data[api_service][dialogue_id]

        return train, dev


if __name__ == "__main__":

    corpus = Corpus()
    print(corpus.max_numVals_of_slot)
    corpus.check_distribution(corpus.train_data)
    corpus.check_distribution(corpus.test_data)
