import copy
import sys
from pathlib import Path

import penman
import regex as re
import torch
from transformers import BartTokenizer

from spring_amr import ROOT, postprocessing
from spring_amr.linearization import AMRTokens, AMRLinearizer
from spring_amr.penman import encode
from spring_amr.tokenization_visibility_matrix import VisibilityMatrixConstructor
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from transformers.tokenization_utils import BatchEncoding
import itertools

# Define type aliases and NamedTuples
TextInput = str
PreTokenizedInput = List[str]
EncodedInput = List[int]
TextInputPair = Tuple[str, str]
PreTokenizedInputPair = Tuple[List[str], List[str]]
EncodedInputPair = Tuple[List[int], List[int]]


class AMRBartTokenizer(BartTokenizer):

    INIT = 'Ġ'

    ADDITIONAL = [
        AMRTokens.PNTR_N,
        AMRTokens.STOP_N,
        AMRTokens.LIT_START,
        AMRTokens.LIT_END,
        AMRTokens.BACKR_SRC_N,
        AMRTokens.BACKR_TRG_N,]

    def __init__(self, *args, use_pointer_tokens=False, collapse_name_ops=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.patterns = re.compile(
            r""" ?<[a-z]+:?\d*>| ?:[^\s]+|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.linearizer = AMRLinearizer(use_pointer_tokens=use_pointer_tokens, collapse_name_ops=collapse_name_ops)
        self.use_pointer_tokens = use_pointer_tokens
        self.collapse_name_ops = collapse_name_ops
        self.recategorizations = set()
        self.modified = 0

    @classmethod
    def from_pretrained(cls, pretrained_model_path, pred_min=5, *args, **kwargs):
        inst = super().from_pretrained(pretrained_model_path, *args, **kwargs)
        inst.init_amr_vocabulary(pred_min=pred_min)
        return inst

    def init_amr_vocabulary(self, pred_min=5):
        for tok in [self.bos_token, self.eos_token, self.pad_token, '<mask>', '<unk>']:
            ntok = self.INIT + tok
            i = self.encoder[tok]
            self.decoder[i] = ntok
            del self.encoder[tok]
            self.encoder[ntok] = i

        tokens = []
        for line in Path(ROOT/'data/vocab/predicates.txt').read_text().strip().splitlines():
            tok, count = line.split()
            if int(count) >= pred_min:
                tokens.append(tok)
                
        for tok in Path(ROOT/'data/vocab/additions.txt').read_text().strip().splitlines():
            tokens.append(tok)

        for tok in Path(ROOT/'data/vocab/recategorizations.txt').read_text().strip().splitlines():
            if not tok.startswith('_'):
                self.recategorizations.add(tok)
            tokens.append(tok)

        if self.use_pointer_tokens:
            for cnt in range(512):
                tokens.append(f"<pointer:{cnt}>")

        tokens += self.ADDITIONAL
        tokens = [self.INIT + t if t[0] not in ('_', '-') else t for t in tokens]
        tokens = [t for t in tokens if t not in self.encoder]
        self.old_enc_size = old_enc_size = len(self.encoder)
        for i, t in enumerate(tokens, start= old_enc_size):
            self.encoder[t] = i

        self.encoder = {k: i for i, (k,v) in enumerate(sorted(self.encoder.items(), key=lambda x: x[1]))}
        self.decoder = {v: k for k, v in sorted(self.encoder.items(), key=lambda x: x[1])}
        self.modified = len(tokens)
        
        self.bos_token = self.INIT + '<s>'
        self.pad_token = self.INIT + '<pad>'
        self.eos_token = self.INIT + '</s>'
        self.unk_token = self.INIT + '<unk>'

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        if token_ids_1 is None:
            return output
        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def _tokenize(self, text, features_list=None):
        """ Tokenize a string. Modified in order to handle sentences with recategorization pointers"""
        bpe_tokens = []

        # Fanyunlong
        # for clause_adjacency matrix and token_visibility_matrix
        clause_id_stack = []
        clause_id = 0
        clause_adjacency_matrix = None
        sub_token_to_clause_id = dict()
        sub_token_to_pos_dict = dict()
        sub_token_to_ner_dict = dict()
        sub_token_id = 0
        current_clause_id = -1
        token_id = 0

        for tok_span in text.lstrip().split(' '):
            try:
                token_pos = features_list[token_id]['pos']
                token_ner = features_list[token_id]['ner']
                token_id += 1
            except:
                print('error')

            # Fanyunlong
            if tok_span in VisibilityMatrixConstructor.CLAUSE_TAG:
                if 'MulSnt' in tok_span:  # visibility of every two sentences equals 0 (invisible)
                    continue
                current_clause_tag = tok_span
                # Start Clause TAG
                if '/' not in current_clause_tag:
                    if current_clause_tag not in VisibilityMatrixConstructor.VIRTUAL_CLAUSE_TAG:
                        current_clause_id = clause_id
                        if clause_adjacency_matrix is None:
                            clause_adjacency_matrix = [[2]]  # visibility of the same clause equals 2 (same clause)
                        else:
                            for x in range(len(clause_adjacency_matrix)):
                                clause_adjacency_matrix[x].append(0)
                            temp = [0] * current_clause_id
                            temp.append(2)  # visibility of the same clause equals 2 (same clause)
                            clause_adjacency_matrix.append(temp)
                        if len(clause_id_stack) > 0 and clause_id_stack[-1][1] in {'<AND>', '<OR>', '<BUT>'}:
                            clause_id_stack[-1][2].append(current_clause_id)

                        clause_id += 1
                    else:
                        current_clause_id = -1
                    clause_id_stack.append((current_clause_id, current_clause_tag, []))
                # Stop Clause TAG
                else:
                    if len(clause_id_stack) > 1:
                        clause_id_stack[-2][2].extend(clause_id_stack[-1][2])  # ``OP'' type Children of node 'AND', 'OR' or 'BUT' should be dilivered upward
                    if clause_id_stack[-1][0] == -1:  # Stack top node is 'AND', 'OR' or 'BUT', then set visibilities between its children to 1
                        for i in range(len(clause_id_stack[-1][2])):
                            for j in range(len(clause_id_stack[-1][2])):
                                if clause_id_stack[-1][2][i] != clause_id_stack[-1][2][j]:
                                    clause_adjacency_matrix[clause_id_stack[-1][2][i]][clause_id_stack[-1][2][j]] = 1
                                    clause_adjacency_matrix[clause_id_stack[-1][2][j]][clause_id_stack[-1][2][i]] = 1
                    temp_stack_top_id = clause_id_stack[-1][0]
                    temp_stack_top_chidren = clause_id_stack[-1][2]
                    clause_id_stack.pop(-1)
                    if len(clause_id_stack) > 0 and clause_id_stack[-1][0] != -1 and temp_stack_top_id != -1:
                        clause_adjacency_matrix[clause_id_stack[-1][0]][temp_stack_top_id] = 1
                        clause_adjacency_matrix[temp_stack_top_id][clause_id_stack[-1][0]] = 1
                        for child_id in temp_stack_top_chidren:
                            if child_id != clause_id_stack[-1][0]:
                                clause_adjacency_matrix[clause_id_stack[-1][0]][child_id] = 1
                                clause_adjacency_matrix[child_id][clause_id_stack[-1][0]] = 1
                continue

            tok_span = tok_span.strip()
            recats = tok_span.rsplit('_', 1)
            new_added_sub_tokens = []
            if len(recats) == 2 and recats[0] in self.recategorizations and ('_' + recats[1]) in self.encoder:
                new_added_sub_tokens = [self.INIT + recats[0], '_' + recats[1]]
                bpe_tokens.extend(new_added_sub_tokens)
            else:
                for token in re.findall(self.pat, ' ' + tok_span):
                    token = "".join(
                        self.byte_encoder[b] for b in token.encode("utf-8")
                    )  # Maps all our bytes to unicode strings, avoiding controle tokens of the BPE (spaces in our case)
                    new_added_sub_tokens.extend(self.bpe(token).split(" "))
                    bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))

            for sub_t in new_added_sub_tokens:
                sub_token_to_clause_id[sub_token_id] = current_clause_id
                sub_token_to_pos_dict[sub_token_id] = token_pos
                sub_token_to_ner_dict[sub_token_id] = token_ner
                sub_token_id += 1
        
        token_visibility_matrix = VisibilityMatrixConstructor.construct_visibility_matrix(clause_adjacent_matrix=clause_adjacency_matrix, sub_token_to_clause_id=sub_token_to_clause_id, visibility_rate_dict=self.visibility_rate_dict, token_id_to_pos_dict=sub_token_to_pos_dict, token_id_to_ner_dict=sub_token_to_ner_dict)

        return bpe_tokens, token_visibility_matrix

    def _tok_bpe(self, token, add_space=True):
        # if add_space:
        #     token = ' ' + token.lstrip()
        tokk = []
        tok = token.strip()
        recats = tok.rsplit('_', 1)
        if len(recats) == 2 and recats[0] in self.recategorizations and ('_' + recats[1]) in self.encoder:
            tokk.extend([self.INIT + recats[0], '_' + recats[1]])
        else:
            for tok in self.patterns.findall(' ' + token):
                tok = "".join(
                    self.byte_encoder[b] for b in tok.encode("utf-8"))
                toks = self.bpe(tok).split(' ')
                tokk.extend(toks)
        return tokk

    def _get_nodes_and_backreferences(self, graph):
        lin = self.linearizer.linearize(graph)
        linearized_nodes, backreferences = lin.nodes, lin.backreferences
        return linearized_nodes, backreferences

    def tokenize_amr(self, graph):
        linearized_nodes, backreferences = self._get_nodes_and_backreferences(graph)

        bpe_tokens = []
        bpe_backreferences = []
        counter = 0
        
        for i, (backr, tokk) in enumerate(zip(backreferences, linearized_nodes)):
            is_in_enc = self.INIT + tokk in self.encoder
            is_rel = tokk.startswith(':') and len(tokk) > 1
            is_spc = tokk.startswith('<') and tokk.endswith('>')
            is_of = tokk.startswith(':') and tokk.endswith('-of')
            is_frame = re.match(r'.+-\d\d', tokk) is not None

            if tokk.startswith('"') and tokk.endswith('"'):
                tokk = tokk[1:-1].replace('_', ' ')
                bpe_toks = [self.INIT + AMRTokens.LIT_START]
                bpe_toks += self._tok_bpe(tokk, add_space=True)
                bpe_toks.append(self.INIT + AMRTokens.LIT_END)

            elif (is_rel or is_spc or is_frame or is_of):
                if is_in_enc:
                    bpe_toks = [self.INIT + tokk]
                elif is_frame:
                    bpe_toks = self._tok_bpe(tokk[:-3], add_space=True) + [tokk[-3:]]
                elif is_of:
                    rel = tokk[:-3]
                    if self.INIT + rel in self.encoder:
                        bpe_toks = [self.INIT + rel, '-of']
                    else:
                        bpe_toks = [self.INIT + ':'] + self._tok_bpe(rel[1:], add_space=True) + ['-of']
                elif is_rel:
                    bpe_toks = [self.INIT + ':'] + self._tok_bpe(tokk[1:], add_space=True)
                else:
                    raise

            else:
                if is_in_enc:
                    bpe_toks = [self.INIT + tokk]
                else:
                    bpe_toks = self._tok_bpe(tokk, add_space=True)

            bpe_tokens.append(bpe_toks)

            if i == backr:
                bpe_backr = list(range(counter, counter + len(bpe_toks)))
                counter += len(bpe_toks)
                bpe_backreferences.append(bpe_backr)
            else:
                bpe_backreferences.append(bpe_backreferences[backr][0:1])
                counter += 1               
        bpe_tokens = [b for bb in bpe_tokens for b in bb]
        bpe_token_ids = [self.encoder.get(b, self.unk_token_id) for b in bpe_tokens]
        bpe_backreferences = [b for bb in bpe_backreferences for b in bb]
        return bpe_tokens, bpe_token_ids, bpe_backreferences

    def batch_encode_sentences(self, sentences, features_list=None, ner_list=None, device=torch.device('cpu')):
        sentences = [s for s in sentences]
        extra = {'sentences': sentences}
        batch = self.batch_encode_plus(sentences, features_list=features_list, return_tensors='pt', pad_to_max_length=True)
        batch = {k: v.to(device) for k, v in batch.items()}
        return batch, extra
    
    def linearize(self, graph):
        shift = len(self.encoder)
        tokens, token_ids, backreferences = self.tokenize_amr(graph)
        extra = {'linearized_graphs': tokens, 'graphs': graph}
        token_uni_ids = \
            [idx if i == b else b + shift for i, (idx, b) in enumerate(zip(token_ids, backreferences))]
        if token_uni_ids[-1] != (self.INIT + AMRTokens.EOS_N):
            tokens.append(self.INIT + AMRTokens.EOS_N)
            token_ids.append(self.eos_token_id)
            token_uni_ids.append(self.eos_token_id)
            backreferences.append(len(backreferences))
        return token_uni_ids, extra
        
    def batch_encode_graphs(self, graphs, device=torch.device('cpu')):
        linearized, extras = zip(*[self.linearize(g) for g in graphs])
        return self.batch_encode_graphs_from_linearized(linearized, extras, device=device)
    
    def batch_encode_graphs_from_linearized(self, linearized, extras=None, device=torch.device('cpu')):
        if extras is not None:
            batch_extra = {'linearized_graphs': [], 'graphs': []}
            for extra in extras:
                batch_extra['graphs'].append(extra['graphs'])
                batch_extra['linearized_graphs'].append(extra['linearized_graphs'])
        else:
            batch_extra = {}
        maxlen = 0
        batch = []
        for token_uni_ids in linearized:
            maxlen = max(len(token_uni_ids), maxlen)
            batch.append(token_uni_ids)
        batch = [x + [self.pad_token_id] * (maxlen - len(x)) for x in batch]
        batch = torch.tensor(batch).to(device)
        batch = {'decoder_input_ids': batch[:, :-1], 'lm_labels': batch[:, 1:]}
        return batch, batch_extra

    def decode_amr(self, tokens, restore_name_ops=False):
        try:
            nodes, backreferences = postprocessing.decode_into_node_and_backreferences(tokens, self)
        except Exception as e:
            print('Decoding failure:', file=sys.stderr)
            print(e, file=sys.stderr)
            return postprocessing.BACKOFF, postprocessing.ParsedStatus.BACKOFF, (None, None)
        if self.use_pointer_tokens:
            nodes, backreferences = postprocessing.restore_backreferences_from_pointers(nodes)
        try:
            graph_ = graph = postprocessing.build_graph(nodes, backreferences, restore_name_ops=restore_name_ops)
        except Exception as e:
            print('Building failure:', file=sys.stderr)
            print(nodes, file=sys.stderr)
            print(backreferences, file=sys.stderr)
            print(e, file=sys.stderr)
            return postprocessing.BACKOFF, postprocessing.ParsedStatus.BACKOFF, (None, None)
        try:
            graph, status = postprocessing.connect_graph_if_not_connected(graph)
            if status == postprocessing.ParsedStatus.BACKOFF:
                print('Reconnection 1 failure:')
                print(nodes, file=sys.stderr)
                print(backreferences, file=sys.stderr)
                print(graph_, file=sys.stderr)
            return graph, status, (nodes, backreferences)
        except Exception as e:
            print('Reconnction 2 failure:', file=sys.stderr)
            print(e, file=sys.stderr)
            print(nodes, file=sys.stderr)
            print(backreferences, file=sys.stderr)
            print(graph_, file=sys.stderr)
            return postprocessing.BACKOFF, postprocessing.ParsedStatus.BACKOFF, (nodes, backreferences)

    def batch_encode_plus(
            self,
            batch_text_or_text_pairs: Union[
                List[TextInput],
                List[TextInputPair],
                List[PreTokenizedInput],
                List[PreTokenizedInputPair],
                List[EncodedInput],
                List[EncodedInputPair],
            ],
            add_special_tokens: bool = True,
            max_length: Optional[int] = None,
            stride: int = 0,
            truncation_strategy: str = "longest_first",
            pad_to_max_length: bool = False,
            is_pretokenized: bool = False,
            return_tensors: Optional[str] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_masks: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_masks: bool = False,
            return_offsets_mapping: bool = False,
            return_lengths: bool = False,
            features_list=None,
            **kwargs
    ) -> BatchEncoding:

        def get_input_ids(text, features_list=None):
            if isinstance(text, str):
                tokens, token_visibility_matrx = self.tokenize(text, add_special_tokens=add_special_tokens, features_list=features_list, **kwargs)
                return self.convert_tokens_to_ids(tokens), token_visibility_matrx
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        # Throw an error if we can pad because there is no padding token
        if pad_to_max_length and self.pad_token_id is None:
            raise ValueError(
                "Unable to set proper padding strategy as the tokenizer does not have a padding token. In this case please set the `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` or add a new pad token via the function add_special_tokens if you want to use a padding strategy"
            )

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers."
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        input_ids = []
        for i in range(len(batch_text_or_text_pairs)):
            ids_or_pair_ids = batch_text_or_text_pairs[i]
            if isinstance(ids_or_pair_ids, (list, tuple)) and len(ids_or_pair_ids) == 2 and not is_pretokenized:
                ids, pair_ids = ids_or_pair_ids
            else:
                ids, pair_ids = ids_or_pair_ids, None

            first_ids, token_visibility_matrix = get_input_ids(ids, features_list=features_list[i])
            second_ids = get_input_ids(pair_ids) if pair_ids is not None else None
            input_ids.append((first_ids, second_ids, token_visibility_matrix))

        if max_length is None and pad_to_max_length:

            def total_sequence_length(input_pairs):
                first_ids, second_ids, token_visibility_matrix = input_pairs
                return len(first_ids) + (
                    self.num_special_tokens_to_add()
                    if second_ids is None
                    else (len(second_ids) + self.num_special_tokens_to_add(pair=True))
                )

            max_length = max([total_sequence_length(ids) for ids in input_ids])

        batch_outputs = {}
        for first_ids, second_ids, token_visibility_matrix in input_ids:
            # Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by
            # the model. It adds special tokens, truncates sequences if overflowing while taking into account
            # the special tokens and manages a window stride for overflowing tokens
            outputs = self.prepare_for_model(
                first_ids,
                pair_ids=second_ids,
                max_length=max_length,
                pad_to_max_length=pad_to_max_length,
                add_special_tokens=add_special_tokens,
                stride=stride,
                truncation_strategy=truncation_strategy,
                return_attention_mask=return_attention_masks,
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_masks,
                return_lengths=return_lengths,
                return_tensors=None,  # We will convert the whole batch to tensors at the end
            )

            # Fanyunlong
            modified_token_visibility_matrix = VisibilityMatrixConstructor.reconstruct_visibility_matrix_by_max_length(token_visibility_matrix, max_length)

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)
            if 'token_visibility_matrix' not in batch_outputs.keys():
                batch_outputs['token_visibility_matrix'] = []
            batch_outputs['token_visibility_matrix'].append(modified_token_visibility_matrix)

        if return_tensors is not None:

            self.convert_to_tensors_(batch_outputs, return_tensors)
        return BatchEncoding(batch_outputs)

    def tokenize(self, text: TextInput, features_list=None, **kwargs):
        """ Converts a string in a sequence of tokens (string), using the tokenizer.
            Split in words for word-based vocabulary or sub-words for sub-word-based
            vocabularies (BPE/SentencePieces/WordPieces).

            Take care of added tokens.

            Args:
                text (:obj:`string`): The sequence to be encoded.
                features_list (:obj:`List`): The features list of the sequence.
                **kwargs (:obj: `dict`): Arguments passed to the model-specific `prepare_for_tokenization` preprocessing method.
        """
        all_special_tokens = self.all_special_tokens
        text = self.prepare_for_tokenization(text, **kwargs)

        # TODO: should this be in the base class?
        def lowercase_text(t):
            # convert non-special tokens to lowercase
            escaped_special_toks = [re.escape(s_tok) for s_tok in all_special_tokens]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            return re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), t)

        if self.init_kwargs.get("do_lower_case", False):
            text = lowercase_text(text)

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                sub_text = sub_text.rstrip()
                if i == 0 and not sub_text:
                    result += [tok]
                elif i == len(split_text) - 1:
                    if sub_text:
                        result += [sub_text]
                    else:
                        pass
                else:
                    if sub_text:
                        result += [sub_text]
                    result += [tok]
            return result

        def split_on_tokens(tok_list, text, features_list=None):
            if not text.strip():
                return []
            if not tok_list:
                return self._tokenize(text, features_list)

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_added_tokens_encoder:
                        tokenized_text += split_on_token(tok, sub_text)
                    else:
                        tokenized_text += [sub_text]
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token, features_list) if token not in self.unique_added_tokens_encoder else [token]
                        for token in tokenized_text
                    )
                )
            )

        added_tokens = self.unique_added_tokens_encoder
        # Fanyunlong
        # tokenized_text = split_on_tokens(added_tokens, text, pos_list, ner_list)
        tokenized_text, token_visibility_matrx = split_on_tokens(added_tokens, text, features_list)
        return tokenized_text, token_visibility_matrx


class PENMANBartTokenizer(AMRBartTokenizer):

    def __init__(self, *args, raw_graph=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.linearizer = None
        self.remove_pars = False
        self.raw_graph = raw_graph

    def _tokenize_encoded_graph(self, encoded):
        linearized = re.sub(r"(\".+?\")", r' \1 ', encoded)
        pieces = []
        for piece in linearized.split():
            if piece.startswith('"') and piece.endswith('"'):
                pieces.append(piece)
            else:
                piece = piece.replace('(', ' ( ')
                piece = piece.replace(')', ' ) ')
                piece = piece.replace(':', ' :')
                piece = piece.replace('/', ' / ')
                piece = piece.strip()
                pieces.append(piece)
        linearized = re.sub(r'\s+', ' ', ' '.join(pieces)).strip()
        linearized_nodes = [AMRTokens.BOS_N] + linearized.split(' ')
        return linearized_nodes

    def tokenize_amr(self, graph):
        if self.raw_graph:
            graph_ = copy.deepcopy(graph)
            graph_.metadata = {}
            linearized = penman.encode(graph_)
            linearized = re.sub(r"\s+", ' ', linearized)
            bpe_tokens = [self.bos_token] + self._tokenize(linearized)[:1022][0]
            bpe_token_ids = [self.encoder.get(b, self.unk_token_id) for b in bpe_tokens]
            bpe_backreferences = list(range(len(bpe_token_ids)))
            return bpe_tokens, bpe_token_ids, bpe_backreferences
        else:
            return super().tokenize_amr(graph)

    def _get_nodes_and_backreferences(self, graph):
        graph_ = copy.deepcopy(graph)
        graph_.metadata = {}
        linearized = penman.encode(graph_)
        linearized_nodes = self._tokenize_encoded_graph(linearized)

        if self.use_pointer_tokens:
            remap = {}
            for i in range(1, len(linearized_nodes)):
                nxt = linearized_nodes[i]
                lst = linearized_nodes[i-1]
                if nxt == '/':
                    remap[lst] = f'<pointer:{len(remap)}>'
            i = 1
            linearized_nodes_ = [linearized_nodes[0]]
            while i < (len(linearized_nodes)):
                nxt = linearized_nodes[i]
                lst = linearized_nodes_[-1]
                if nxt in remap:
                    if lst == '(' and linearized_nodes[i+1] == '/':
                        nxt = remap[nxt]
                        i += 1
                    elif lst.startswith(':'):
                        nxt = remap[nxt]
                linearized_nodes_.append(nxt)
                i += 1
            linearized_nodes = linearized_nodes_
            if self.remove_pars:
                linearized_nodes = [n for n in linearized_nodes if n != '(']
        backreferences = list(range(len(linearized_nodes)))
        return linearized_nodes, backreferences

    def _classify(self, node):
        if not isinstance(node, str):
            return "CONST"
        elif node == 'i':
            return "I"
        elif re.match(r'^[a-z]\d*$', node) is not None:
            return "VAR"
        elif node[0].isdigit():
            return "CONST"
        elif node.startswith('"') and node.endswith('"'):
            return "CONST"
        elif node in ('+', '-'):
            return "CONST"
        elif node == ':mode':
            return 'MODE'
        elif node.startswith(':'):
            return "EDGE"
        elif node in ['/', '(', ')']:
            return node
        elif node[0].isalpha():
            for char in (',', ':', '/', '(', ')', '.', '!', '?', '\\'):
                if char in node:
                    return "CONST"
            return "INST"
        else:
            return 'CONST'

    def _fix_and_make_graph(self, nodes):

        nodes_ = []
        for n in nodes:
            if isinstance(n, str):
                if n.startswith('<') and n.endswith('>') and (not n.startswith('<pointer:')):
                    pass
                else:
                    nodes_.append(n)
            else:
                nodes_.append(n)
        nodes = nodes_

        if self.use_pointer_tokens:

            i = 0
            nodes_ = []
            while i < len(nodes):
                nxt = nodes[i]
                pst = None
                if isinstance(nxt, str) and nxt.startswith('<pointer:'):
                    e = nxt.find('>')
                    if e != len(nxt) -1:
                        pst = nxt[e+1:]
                        nxt = nxt[:e+1]
                    nodes_.append(nxt)
                    if pst is not None:
                        nodes_.append(pst)
                else:
                    nodes_.append(nxt)
                i += 1
            nodes = nodes_

            i = 1
            nodes_ = [nodes[0]]
            while i < len(nodes):
                nxt = nodes[i]
                if isinstance(nxt, str) and nxt.startswith('<pointer:'):
                    nxt = 'z' + nxt[9:-1]
                    fol = nodes[i+1]
                    # is not expansion
                    if isinstance(fol, str) and (fol.startswith(':') or (fol == ')')):
                        nodes_.append(nxt)
                    else:
                        if self.remove_pars:
                            nodes_.append('(')
                        else:
                            if nodes_[-1] != '(':
                                nodes_.append('(')
                                #pass
                        nodes_.append(nxt)
                        nodes_.append('/')
                else:
                    nodes_.append(nxt)
                i += 1
            nodes = nodes_

        i = 0
        nodes_ = []
        while i < (len(nodes) - 1):
            if nodes[i] == ':':
                nodes_.append(nodes[i] + nodes[i+1])
                i += 2
                last = False
            else:
                nodes_.append(nodes[i])
                i += 1
                last = True
        if last:
            nodes_.append(nodes[-1])
        nodes = nodes_

        i = 0
        nodes_ = []
        while i < (len(nodes)):
            if i < 2:
                nodes_.append(nodes[i])
                i += 1
            elif nodes_[-2] == '/' and nodes[i] == '/':
                i += 2
            else:
                nodes_.append(nodes[i])
                i += 1
        nodes = nodes_

        i = 0
        newvars = 0
        variables = set()
        remap = {}
        nodes_ = []
        while i < (len(nodes)):

            next = nodes[i]

            if next == '/':
                last = nodes_[-1]
                if last in variables:
                    last_remap = f"z{newvars+1000}"
                    newvars += 1
                    nodes_[-1] = last_remap
                    remap[last] = last_remap
                variables.add(last)
                nodes_.append(next)

            elif self._classify(next) == 'VAR' and next in remap and (i < len(nodes) - 1) and nodes[i+1] != '/':
                next = remap[next]
                nodes_.append(next)

            else:
                nodes_.append(next)

            i += 1

        nodes = nodes_
        pieces_ = []
        open_cnt = 0
        closed_cnt = 0
        if nodes[0] != '(':
            pieces_.append('(')
            open_cnt += 1
        for p in nodes:
            if p == '(':
                open_cnt += 1
            elif p == ')':
                closed_cnt += 1
            pieces_.append(p)
            if open_cnt == closed_cnt:
                break
        nodes = pieces_ + [')'] * (open_cnt - closed_cnt)

        pieces = []
        for piece in nodes:
            if not pieces:
                pieces.append('(')
            else:
                piece = str(piece)
                if piece.startswith('"') or piece.startswith('"') or '"' in piece.strip('"'):
                    piece = '"' + piece.replace('"', '') + '"'

                prev = self._classify(pieces[-1])
                next = self._classify(piece)

                if next == 'CONST':
                    quote = False
                    for char in (',', ':', '/', '(', ')', '.', '!', '?', '\\', '_', '='):
                        if char in piece:
                            quote = True
                            break
                    if quote:
                        piece = '"' + piece.strip('"') + '"'

                if  prev == '(':
                    if next in ('VAR', 'I'):
                        pieces.append(piece)
                elif prev == ')':
                    if next in (')', 'EDGE', 'MODE'):
                        pieces.append(piece)
                elif prev == 'VAR':
                    if next in ('/', 'EDGE', 'MODE', ')'):
                        pieces.append(piece)
                elif prev == '/':
                    if next in ('INST', 'I'):
                        pieces.append(piece)
                elif prev == 'INST':
                    if next in (')', 'EDGE', 'MODE'):
                        pieces.append(piece)
                elif prev == 'I':
                    if next in ('/', ')', 'EDGE', 'MODE'):
                        pieces.append(piece)
                elif prev == 'EDGE':
                    if next in ('(', 'VAR', 'CONST', 'I'):
                        pieces.append(piece)
                    elif next == ')':
                        pieces[-1] = piece
                    elif next in ('EDGE', 'MODE'):
                        pieces[-1] = piece
                elif prev == 'MODE':
                    if next == 'INST':
                        pieces.append(piece)
                elif prev == 'CONST':
                    if next in (')', 'EDGE', 'MODE'):
                        pieces.append(piece)

        pieces_ = []
        open_cnt = 0
        closed_cnt = 0
        if pieces[0] != '(':
            pieces_.append('(')
            open_cnt += 1
        for p in pieces:
            if p == '(':
                open_cnt += 1
            elif p == ')':
                closed_cnt += 1
            pieces_.append(p)
            if open_cnt == closed_cnt:
                break
        pieces = pieces_ + [')'] * (open_cnt - closed_cnt)

        linearized = re.sub(r'\s+', ' ', ' '.join(pieces)).strip()

        """
        line = linearized
        # make sure parentheses match
        # copied from https://github.com/RikVN/AMR/blob/master/restoreAMR/restore_amr.py
        open_count = 0
        close_count = 0
        for i, c in enumerate(line):
            if c == '(':
                open_count += 1
            elif c == ')':
                close_count += 1
            if open_count == close_count and open_count > 0:
                line = line[:i].strip()
                break
        old_line = line
        while True:
            open_count = len(re.findall(r'\(', line))
            close_count = len(re.findall(r'\)', line))
            if open_count > close_count:
                line += ')' * (open_count - close_count)
            elif close_count > open_count:
                for i in range(close_count - open_count):
                    line = line.rstrip(')')
                    line = line.rstrip(' ')
            if old_line == line:
                break
            old_line = line
        """

        graph = penman.decode(linearized + ' ')
        triples = []
        newvars = 2000
        for triple in graph.triples:
            x, rel, y = triple
            if x is None:
                pass
            elif rel == ':instance' and y is None:
                triples.append(penman.Triple(x, rel, 'thing'))
            elif y is None:
                var = f'z{newvars}'
                newvars += 1
                triples.append(penman.Triple(x, rel, var))
                triples.append(penman.Triple(var, ':instance', 'thing'))
            else:
                triples.append(triple)
        graph = penman.Graph(triples)
        linearized = encode(graph)

        def fix_text(linearized=linearized):
            n = 0
            def _repl1(match):
                nonlocal n
                out = match.group(1) + match.group(2) + str(3000 + n) + ' / ' + match.group(2) + match.group(3)
                n += 1
                return out
            linearized = re.sub(r'(\(\s?)([a-z])([^\/:\)]+[:\)])', _repl1, linearized,
                                flags=re.IGNORECASE | re.MULTILINE)

            def _repl2(match):
                return match.group(1)
            linearized = re.sub(r'(\(\s*[a-z][\d+]\s*\/\s*[^\s\)\(:\/]+\s*)((?:/\s*[^\s\)\(:\/]+\s*)+)', _repl2,
                                linearized,
                                flags=re.IGNORECASE | re.MULTILINE)

            # adds a ':' to args w/o it
            linearized = re.sub(r'([^:])(ARG)', r'\1 :\2', linearized)

            # removes edges with no node
            # linearized = re.sub(r':[^\s\)\(:\/]+?\s*\)', ')', linearized, flags=re.MULTILINE)

            return linearized

        linearized = fix_text(linearized)

        g = penman.decode(linearized)
        return g

    def decode_amr(self, tokens, restore_name_ops=None):
        try:
            if self.raw_graph:
                nodes = self._tokenize_encoded_graph(self.decode(tokens))
                backreferences = list(range(len(nodes)))
            else:
                nodes, backreferences = postprocessing.decode_into_node_and_backreferences(tokens, self)
            nodes_ = nodes
        except Exception as e:
            print('Decoding failure:', file=sys.stderr)
            print(e, file=sys.stderr)
            return postprocessing.BACKOFF, postprocessing.ParsedStatus.BACKOFF, (None, None)
        try:
            graph_ = graph = self._fix_and_make_graph(nodes)
            if self.collapse_name_ops:
                graph_ = graph = postprocessing._split_name_ops(graph)
        except Exception as e:
            print('Building failure:', file=sys.stderr)
            print(nodes, file=sys.stderr)
            print(backreferences, file=sys.stderr)
            print(e, file=sys.stderr)
            return postprocessing.BACKOFF, postprocessing.ParsedStatus.BACKOFF, (None, None)
        try:
            graph, status = postprocessing.connect_graph_if_not_connected(graph)
            if status == postprocessing.ParsedStatus.BACKOFF:
                print('Reconnection 1 failure:')
                print(nodes, file=sys.stderr)
                print(backreferences, file=sys.stderr)
                print(graph_, file=sys.stderr)
            return graph, status, (nodes_, backreferences)
        except Exception as e:
            print('Reconnction 2 failure:', file=sys.stderr)
            print(e, file=sys.stderr)
            print(nodes, file=sys.stderr)
            print(backreferences, file=sys.stderr)
            print(graph_, file=sys.stderr)
            return postprocessing.BACKOFF, postprocessing.ParsedStatus.BACKOFF, (nodes_, backreferences)
