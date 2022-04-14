from transformers import AutoTokenizer

tokens = [
    '[CLS]', 'keep', '[SEP]', 's', '##ys', ':', 'In', 'which', 'city',
    'should', 'I', 'search', '?', 'us', '##r', ':', 'Please', 'look', 'in',
    'the', 'city', 'of', 'Phil', '##ly', '.', '[PAD]', '[PAD]', '[PAD]',
    '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
    '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
    '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
    '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
    '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
    '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
    '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
    '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
    '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[SEP]', 'Find', 'cultural',
    'events', '-', 'concerts', 'and', 'plays', '-', 'happening', 'in', 'a',
    'city', '#', 'Find', '##E', '##vent', '##s', '[SEP]', 'Buy', 'tickets',
    'for', 'a', 'cultural', 'event', 'and', 'date', 'in', 'a', 'given', 'city',
    '#', 'Buy', '##E', '##vent', '##T', '##ick', '##ets', '[SEP]', 'none',
    '[SEP]', '[PAD]', '[SEP]', '[PAD]', '[SEP]', '[PAD]', '[SEP]', '[PAD]',
    '[SEP]', '[PAD]', '[SEP]', '[PAD]', '[SEP]', '[PAD]', '[SEP]', '[PAD]',
    '[SEP]'
]
tokens2 = ['[CLS]', 'keep', '[SEP]', 's', '##ys', ':', 'In', 'which', 'city', 'should', 'I', 'search', '?', 'us', '##r', ':', 'Please', 'look', 'in', 'the', 'city', 'of', 'Phil', '##ly', '.', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[SEP]', 'Find', 'cultural', 'events', '-', 'concerts', 'and', 'plays', '-', 'happening', 'in', 'a', 'city', '#', 'Find', '##E', '##vent', '##s', '[SEP]', 'Buy', 'tickets', 'for', 'a', 'cultural', 'event', 'and', 'date', 'in', 'a', 'given', 'city', '#', 'Buy', '##E', '##vent', '##T', '##ick', '##ets', '[SEP]', 'none', '[SEP]', '[PAD]', '[SEP]', '[PAD]', '[SEP]', '[PAD]', '[SEP]', '[PAD]', '[SEP]', '[PAD]', '[SEP]', '[PAD]', '[SEP]', '[PAD]', '[SEP]', '[PAD]', '[SEP]']

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
print(tokenizer.convert_tokens_to_string(tokens2))