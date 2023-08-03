import logging
import random
import torch
from cached_property import cached_property
from torch.utils.data import Dataset
from spring_amr.IO import read_raw_amr_data, read_feature_file


def reverse_direction(x, y, pad_token_id=1):
    input_ids = torch.cat([y['decoder_input_ids'], y['lm_labels'][:, -1:]], 1)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[input_ids == pad_token_id] = 0
    decoder_input_ids = x['input_ids'][:,:-1]
    lm_labels = x['input_ids'][:,1:]
    x = {'input_ids': input_ids, 'attention_mask': attention_mask}
    y = {'decoder_input_ids': decoder_input_ids, 'lm_labels': lm_labels}
    return x, y


class AMRDataset(Dataset):
    
    def __init__(
        self,
        paths,
        tokenizer,
        device=torch.device('cpu'),
        use_recategorization=False,
        remove_longer_than=None,
        remove_wiki=False,
        dereify=True,
        feature_paths=None,
    ):
        self.paths = paths
        self.tokenizer = tokenizer
        self.device = device
        graphs = read_raw_amr_data(paths, use_recategorization, remove_wiki=remove_wiki, dereify=dereify)
        self.graphs = []
        self.sentences = []
        self.linearized = []
        self.linearized_extra = []
        self.remove_longer_than = remove_longer_than
        self.feature_paths = feature_paths

        features_map = read_feature_file(feature_paths)

        count = 0
        for g in graphs:
            l, e = self.tokenizer.linearize(g)

            # Fanyunlong Redundant Check Preprocessing
            # try:
            #     # self.tokenizer.batch_encode_sentences([g.metadata['snt']])
            #     self.tokenizer.batch_encode_sentences([' '.join(token_list[count])], [pos_list[count]], [ner_list[count]])
            # except Exception as exception:
            #     print(count)
            #     print(exception)
            #     logging.warning('Invalid sentence!')
            #     continue

            if remove_longer_than and len(l) > remove_longer_than:
                continue
            if len(l) > 1024:
                logging.warning('Sequence longer than 1024 included. BART does not support it!')

            self.sentences.append(g.metadata['snt'])
            if g.metadata['id'] in features_map.keys():
                g.metadata['feature_tuple'] = features_map[g.metadata['id']]
            else:
                token_count = len(g.metadata['snt'].strip().split())
                feature_tuple = dict()
                feature_tuple['id'] = g.metadata['id']
                feature_tuple['token'] = g.metadata['snt'].strip().split()
                feature_tuple['index'] = token_count * ['0']
                feature_tuple['keyword'] = token_count * ['O']
                feature_tuple['pos'] = token_count * ['O']
                feature_tuple['ner'] = token_count * ['O']
                feature_tuple['dfs_clauses'] = ["<Snt>", "0", "</Snt>"]

                g.metadata['feature_tuple'] = feature_tuple            # simple sentence without HCA annotations
            self.graphs.append(g)
            self.linearized.append(l)
            self.linearized_extra.append(e)
            count += 1

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sample = {}
        sample['id'] = idx
        sample['sentences'] = self.sentences[idx]
        if self.linearized is not None:
            sample['linearized_graphs_ids'] = self.linearized[idx]
            sample.update(self.linearized_extra[idx])            
        return sample
    
    def size(self, sample):
        return len(sample['linearized_graphs_ids'])
    
    def collate_fn(self, samples, device=torch.device('cpu')):
        # Fanyunlong
        x_token = []
        features_list = []

        for s in samples:
            # x.append(s['sentences'])
            features_list.append(s['graphs'].metadata['feature_tuple'])
            x_token.append(' '.join(s['graphs'].metadata['feature_tuple']['token']))
        # x = [ for s in samples]
        x, extra = self.tokenizer.batch_encode_sentences(x_token, features_list=features_list, device=device)
        if 'linearized_graphs_ids' in samples[0]:
            y = [s['linearized_graphs_ids'] for s in samples]
            y, extra_y = self.tokenizer.batch_encode_graphs_from_linearized(y, samples, device=device)
            extra.update(extra_y)
        else:
            y = None
        extra['ids'] = [s['id'] for s in samples]
        return x, y, extra


class AMRDatasetTokenBatcherAndLoader:

    def __init__(self, dataset, batch_size=800, device=torch.device('cpu'), shuffle=False, sort=False):
        assert not (shuffle and sort)
        self.batch_size = batch_size
        self.tokenizer = dataset.tokenizer
        self.dataset = dataset
        self.device = device
        self.shuffle = shuffle
        self.sort = sort

    def __iter__(self):
        it = self.sampler()
        it = ([[self.dataset[s] for s in b] for b in it])
        it = (self.dataset.collate_fn(b, device=self.device) for b in it)
        return it

    @cached_property
    def sort_ids(self):
        lengths = [len(s.split()) for s in self.dataset.sentences]
        ids, _ = zip(*sorted(enumerate(lengths), reverse=True))
        ids = list(ids)
        return ids

    def sampler(self):
        ids = list(range(len(self.dataset)))[::-1]
        
        if self.shuffle:
            random.shuffle(ids)
        if self.sort:
            ids = self.sort_ids.copy()

        batch_longest = 0
        batch_nexamps = 0
        batch_ntokens = 0
        batch_ids = []

        def discharge():
            nonlocal batch_longest
            nonlocal batch_nexamps
            nonlocal batch_ntokens
            ret = batch_ids.copy()
            batch_longest *= 0
            batch_nexamps *= 0
            batch_ntokens *= 0
            batch_ids[:] = []
            return ret

        while ids:
            idx = ids.pop()
            size = self.dataset.size(self.dataset[idx])
            cand_batch_ntokens = max(size, batch_longest) * (batch_nexamps + 1)
            if cand_batch_ntokens > self.batch_size and batch_ids:
                yield discharge()
            batch_longest = max(batch_longest, size)
            batch_nexamps += 1
            batch_ntokens = batch_longest * batch_nexamps
            batch_ids.append(idx)

            if len(batch_ids) == 1 and batch_ntokens > self.batch_size:
                yield discharge()

        if batch_ids:
            yield discharge()
