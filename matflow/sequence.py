"""`matflow.sequence.py`

Module containing functionality to combine options for multiple task executions.

"""

import copy

from matflow.utils import nest_lists, combine_list_of_dicts


def combine_base_sequence(sequences, base=None):

    print(f'\ncombine_base_sequence.')
    print(f'----------------------.')

    for i_idx, i in enumerate(sequences):
        print(
            f'input\n'
            f'\ti_idx: {i_idx}; i.keys: {i.keys()}; name: {i["name"]}; '
            f'nest_idx: {i["nest_idx"]} '
            f'num vals: {len(i["vals"])}'
        )
        for j_idx, j in enumerate(i['vals']):
            print(
                f'\t\tval_idx: {j_idx}; val.keys: {j.keys()}'
            )

    if base is None:
        base = {}

    # Merge parallel sequences:
    merged_seqs = []
    skip_idx = []
    for seq_idx, seq in enumerate(sequences):

        if seq_idx in skip_idx:
            continue

        merged_seqs.append(seq)
        merged_seqs[-1]['name'] = [merged_seqs[-1]['name']]

        for next_idx in range(seq_idx + 1, len(sequences)):
            # print(f'next_idx: {next_idx}')

            if sequences[next_idx]['nest_idx'] == seq['nest_idx']:

                # Merge values:
                for val_idx in range(len(merged_seqs[-1]['vals'])):
                    merged_seqs[-1]['vals'][val_idx].update(
                        sequences[next_idx]['vals'][val_idx]
                    )
                # Merge names:
                name_old = merged_seqs[-1]['name']
                merged_seqs[-1]['name'] += [sequences[next_idx]['name']]

                skip_idx.append(next_idx)

    # Nest nested sequences:
    seq_vals = [i['vals'] for i in merged_seqs]
    nested_seqs = nest_lists(seq_vals)

    nested_seq_all = []
    for seq in nested_seqs:
        nested_seq_all.append(combine_list_of_dicts([base] + seq))

    # for i_idx, i in enumerate(nested_seq_all):
    #     print(
    #         f'combined\n'
    #         f'\ti_idx: {i_idx}; i.keys: {i.keys()}; '
    #     )

    return nested_seq_all
