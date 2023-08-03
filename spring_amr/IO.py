import glob
from typing import List, Union, Iterable
from pathlib import Path
from spring_amr.penman import load as pm_load
from copy import deepcopy


def read_raw_amr_data(
        paths: List[Union[str, Path]],
        use_recategorization=False,
        dereify=True,
        remove_wiki=False,
):
    assert paths

    if not isinstance(paths, Iterable):
        paths = [paths]

    graphs = []
    for path_ in paths:
        for path in glob.glob(str(path_)):
            path = Path(path)    
            graphs.extend(pm_load(path, dereify=dereify, remove_wiki=remove_wiki))

    assert graphs
    
    if use_recategorization:
        for g in graphs:
            metadata = g.metadata
            metadata['snt_orig'] = metadata['snt']
            tokens = eval(metadata['tokens'])
            metadata['snt'] = ' '.join([t for t in tokens if not ((t.startswith('-L') or t.startswith('-R')) and t.endswith('-'))])

    return graphs


# Read sentence pos tag from feature files
def read_feature_file(
        paths: List[Union[str, Path]],
):
    assert paths

    if not isinstance(paths, Iterable):
        paths = [paths]

    features_map = dict()
    feature_tuple = dict()
    for path_ in paths:
        for path in glob.glob(str(path_)):
            file = open(path)
            for line in file:
                # # ::id bolt12_64556_5627.1 ::date
                if line.startswith('# ::id '):
                    if '::' in line[len('# ::id '):]:
                        feature_tuple['id'] = line[len('# ::id '): line[len('# ::id '):].index(' ::') + len('# ::id ')]
                    else:
                        feature_tuple['id'] = line[len('# ::id '):].strip()
                elif line.startswith('# ::tokens '):                                                # amr 2.0
                    feature_tuple['token'] = eval(line.rstrip()[len('# ::tokens '):])
                elif line.startswith('# ::sentence '):                                              # amr 3.0
                    feature_tuple['token'] = line.rstrip()[len('# ::sentence '):].split()
                elif line.startswith('# ::clause_index '):
                    feature_tuple['index'] = eval(line.rstrip()[len('# ::clause_index '):])
                elif line.startswith('# ::pos_tags '):
                    feature_tuple['pos'] = eval(line.rstrip()[len('# ::pos_tags '):])
                elif line.startswith('# ::ner_tags '):
                    feature_tuple['ner'] = eval(line.rstrip()[len('# ::ner_tags '):])
                elif line.startswith('# ::dfs_clauses '):
                    feature_tuple['dfs_clauses'] = eval(line.rstrip()[len('# ::dfs_clauses '):])
                elif line.strip() == '':
                    if feature_tuple['id'] not in features_map.keys():
                        features_map[feature_tuple['id']] = feature_tuple
                    feature_tuple = dict()

            file.close()

    return features_map
