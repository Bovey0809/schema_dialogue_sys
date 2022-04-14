import datasets

if __name__ == '__main__':
    # load datasets
    new_dataset = datasets.load_dataset('parquet',
                                        data_files={
                                            'train': 'train.parquet',
                                            'validation': 'validation.parquet',
                                            'test': 'test.parquet'
                                        })

    # remove columns
    new_dataset = new_dataset.remove_columns([
        "speaker", "utterance", "service", 'slots', 'actions',
        'service_results', 'service_call', 'requested_slots', 'slot_values'
    ])
