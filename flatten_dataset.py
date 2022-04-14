import collections
import json
from datasets import load_from_disk
import pandas as pd


def flatten_turns(dataframe):
    speakers = []
    for speaker in dataframe['turns.speaker']:
        speakers.extend(speaker)
    utts = []
    for utt in dataframe['turns.utterance']:
        utts.extend(utt)
    frames = []
    for frame in dataframe['turns.frames']:
        frames.extend(frame)
    dataframe = pd.DataFrame({
        'speaker': speakers,
        'utterance': utts,
        'frame': frames
    })
    return dataframe


def gen_model_input_utt(dataframe):
    model_input_utt = []
    for i in range(0, len(dataframe), 2):
        if i == 0:
            utt = "sys: ," + " usr: " + dataframe.utterance[i]
        else:
            utt = "sys: " + dataframe.utterance[
                i - 1] + " usr: " + dataframe.utterance[i]

        model_input_utt.append(utt)
        model_input_utt.append("")  # system response
    dataframe['model_input_utt'] = pd.Series(model_input_utt)
    return dataframe


def expand_dataframe(dataframe, name):
    _tmp = pd.DataFrame(dataframe[name].to_list())
    dataframe = pd.concat([dataframe, _tmp], axis=1)
    dataframe = dataframe.drop(name, axis=1)
    return dataframe


def gen_keep_or_jump(dataframe):
    last_services = []
    res = []
    for cur_services in dataframe.service:
        if len(cur_services) != len(last_services):
            last_services = cur_services
            res.append('jump')
        else:
            cur_services.sort()
            last_services.sort()
            for c, l in zip(cur_services, last_services):
                if c != l:
                    res.append("jump")
                    last_services = cur_services
            res.append("keep")
    dataframe['jump_keep'] = pd.Series(res)
    return dataframe


def schema_dict(scheme: list):
    res = collections.defaultdict(list)
    for service_dict in scheme:
        service_name = service_dict['service_name']
        assert service_name not in res, f"{service_name} is duplicated."
        res[service_name] = service_dict
    return res


def gen_intent_description(dataframe, schema):

    def _helper(x):
        intents = schema[x['service'][0]]['intents']
        for intent in intents:
            if intent['name'] == x['active_intent']:
                return intent['description']
        return 'none'

    dataframe['intent_description'] = dataframe.apply(_helper, axis=1)
    return dataframe


if __name__ == "__main__":

    with open("/dstc8-schema-guided-dialogue/train/schema.json", 'r') as f:
        train_schemas = json.load(f)

    with open("/dstc8-schema-guided-dialogue/test/schema.json", 'r') as f:
        test_schemas = json.load(f)

    with open("/dstc8-schema-guided-dialogue/dev/schema.json", 'r') as f:
        dev_schemas = json.load(f)

    test_schemas = schema_dict(test_schemas)
    train_schemas = schema_dict(train_schemas)
    dev_schemas = schema_dict(dev_schemas)

    schemas = dict(train=train_schemas,
                   validation=dev_schemas,
                   test=test_schemas)

    dataset = load_from_disk('schema_data')
    # convert dataset to pandas
    dataset = dataset.flatten()
    dataset.set_format('pandas')

    for split in ['train', 'validation', 'test']:
        df = dataset[split][:]
        df = flatten_turns(df)
        df = gen_model_input_utt(df)
        df = expand_dataframe(df, 'frame')

        df['state'] = df.state.apply(lambda x: x[0])
        df = expand_dataframe(df, 'state')

        df = gen_keep_or_jump(df)
        df = gen_intent_description(df, schemas[split])
        print(df.sample())
        df.to_parquet(f"{split}.parquet")