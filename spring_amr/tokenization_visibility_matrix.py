#!/usr/bin/env python 
# -*- coding:utf-8 -*-
__author__ = 'metrovancloud'

import copy

import torch
from torch import Tensor


class VisibilityMatrixConstructor(object):
	SAME_CLAUSE = 'attn_rate_same_clause'
	ADJACENT_CLAUSE = 'attn_rate_adjacent_clause'
	NONADJACENT_CLAUSE = 'attn_rate_nonadjacent_clause'
	KEYWORD_TOKEN = 'attn_rate_keyword_token'
	REENTRANT_TOKEN = 'attn_rate_reen_token'
	REENTRANT_POS_TYPE = {'PRP', 'PRP$'}
	REENTRANT_NER_TYPE = {'PERSON', 'COUNTRY', 'ORGANIZATION'}
	CLAUSE_TAG = {'<MulSnt>', '</MulSnt>', '<Snt>', '</Snt>', '<OP>', '</OP>', '<AND>', '</AND>', '<OR>', '</OR>', '<BUT>', '</BUT>', '<SUB>', '</SUB>', '<OBJ>', '</OBJ>', '<PRD>', '</PRD>', '<APP>', '</APP>', '<REL>', '</REL>', '<TME>', '</TME>', '<PLA>', '</PLA>', '<CND>', '</CND>', '<CSN>', '</CSN>', '<REA>', '</REA>', '<RES>', '</RES>', '<PUR>', '</PUR>', '<MAN>', '</MAN>', '<CMP>', '</CMP>'}
	VIRTUAL_CLAUSE_TAG = {'<MulSnt>', '</MulSnt>', '<AND>', '</AND>', '<OR>', '</OR>', '<BUT>', '</BUT>'}

	"""
	Args:
		clause_adjacent_matrix: Clause Adjacency Matrix: Element equals 1 when the two clauses are adjacent, otherwise 0
		sub_token_clause_index_list: Sub token index mapping clause id dict
		sub_tokens_num: Sub tokens Number of the input sequence after BPE
		visibility_rate_dict: Dict mapping Relation type between every two tokens to their visibility rate of different
		sub_token_pos_list: Dict mapping token id to its POS tag
		sub_token_ner_list: Dict mapping token id to its NER tag
	"""
	@staticmethod
	def construct_visibility_matrix2(clause_adjacent_matrix, sub_token_clause_index_list, visibility_rate_dict, sub_token_pos_list, sub_token_ner_list, clause_index_to_clause_id_dict, clause_token_inf_mask=False):
		if visibility_rate_dict is None:
			visibility_rate_dict = {VisibilityMatrixConstructor.SAME_CLAUSE: 1.5, VisibilityMatrixConstructor.ADJACENT_CLAUSE: 1.2, VisibilityMatrixConstructor.KEYWORD_TOKEN: 1.3, VisibilityMatrixConstructor.REENTRANT_TOKEN: 1.4, VisibilityMatrixConstructor.NONADJACENT_CLAUSE: -1}

		sub_tokens_num = len(sub_token_clause_index_list)
		assert len(sub_token_clause_index_list) > 0

		visibility_matrix = []
		for i in range(sub_tokens_num):
			visibility_matrix_row = []
			for j in range(sub_tokens_num):
				if i > j and clause_token_inf_mask:			# 当用-inf来做mask时，只赋值上三角矩阵
					visibility_matrix_row.append(0)
				else:
					clause_index_i = sub_token_clause_index_list[i]
					clause_index_j = sub_token_clause_index_list[j]
					try:
						adjacent_flag = clause_adjacent_matrix[clause_index_to_clause_id_dict[clause_index_i]][clause_index_to_clause_id_dict[clause_index_j]]
					except:
						print()
					if adjacent_flag == 2:
						visibility_matrix_row.append(visibility_rate_dict[VisibilityMatrixConstructor.SAME_CLAUSE])
					else:
						if sub_token_pos_list[i] in VisibilityMatrixConstructor.REENTRANT_POS_TYPE \
								or sub_token_pos_list[j] in VisibilityMatrixConstructor.REENTRANT_POS_TYPE \
								or sub_token_ner_list[i] in VisibilityMatrixConstructor.REENTRANT_NER_TYPE \
								or sub_token_ner_list[j] in VisibilityMatrixConstructor.REENTRANT_NER_TYPE:
							visibility_matrix_row.append(visibility_rate_dict[VisibilityMatrixConstructor.REENTRANT_TOKEN])
						elif adjacent_flag == 1:
							visibility_matrix_row.append(visibility_rate_dict[VisibilityMatrixConstructor.ADJACENT_CLAUSE])
						else:
							visibility_matrix_row.append(visibility_rate_dict[VisibilityMatrixConstructor.NONADJACENT_CLAUSE])

			visibility_matrix.append(visibility_matrix_row)

		# print(visibility_matrix)
		# assert list(map(list, zip(*visibility_matrix))) == visibility_matrix
		return visibility_matrix

	@staticmethod
	def reconstruct_visibility_matrix_by_max_length(visibility_matrix, max_length, clause_token_inf_mask=False):
		padding_size = max_length - len(visibility_matrix) - 2
		modified_matrix = []
		for i in range(len(visibility_matrix)):
			row = copy.copy(visibility_matrix[i])
			if clause_token_inf_mask:
				# mask -inf
				row.insert(0, 0)
				row.append(0)
			else:
				row.insert(0, 1.0)
				row.append(1.0)

			row.extend(padding_size * [0])		# There are padding masks for the padding token, it's Okay for any assigned value for them
			modified_matrix.append(row)

		if clause_token_inf_mask:
			temp = (len(visibility_matrix) + 2) * [0]
		else:
			temp = (len(visibility_matrix) + 2) * [1.0]

		temp.extend(padding_size * [0])			# There are padding masks for the padding token, it's Okay for any assigned value for them
		modified_matrix.insert(0, temp)
		modified_matrix.append(temp)
		modified_matrix.extend(padding_size * [(max_length * [0])])		# There are padding masks for the padding token, it's Okay for any assigned value for them

		return modified_matrix

	@staticmethod
	def construct_clause_adjacency_matrix(text):
		clause_id_stack = []
		clause_id = 0
		clause_visibility_matrix = None
		for tok_span in text.lstrip().split(' '):
			# Fanyunlong
			if tok_span in VisibilityMatrixConstructor.CLAUSE_TAG:
				if 'MulSnt' in tok_span:  # visibility of every two sentences equals 0 (invisible)
					continue
				current_clause_tag = tok_span
				# Start Clause TAG
				if '/' not in current_clause_tag:
					if current_clause_tag not in VisibilityMatrixConstructor.VIRTUAL_CLAUSE_TAG:
						current_clause_id = clause_id
						if clause_visibility_matrix is None:
							clause_visibility_matrix = [[2]]  # visibility of the same clause equals 2 (same clause)
						else:
							for x in range(len(clause_visibility_matrix)):
								clause_visibility_matrix[x].append(0)
							temp = [0] * current_clause_id
							temp.append(2)  # visibility of the same clause equals 2 (same clause)
							clause_visibility_matrix.append(temp)
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
									clause_visibility_matrix[clause_id_stack[-1][2][i]][clause_id_stack[-1][2][j]] = 1
									clause_visibility_matrix[clause_id_stack[-1][2][j]][clause_id_stack[-1][2][i]] = 1
					temp_stack_top_id = clause_id_stack[-1][0]
					temp_stack_top_chidren = clause_id_stack[-1][2]
					clause_id_stack.pop(-1)
					if len(clause_id_stack) > 0 and clause_id_stack[-1][0] != -1 and temp_stack_top_id != -1:
						clause_visibility_matrix[clause_id_stack[-1][0]][temp_stack_top_id] = 1
						clause_visibility_matrix[temp_stack_top_id][clause_id_stack[-1][0]] = 1
						for child_id in temp_stack_top_chidren:
							if child_id != clause_id_stack[-1][0]:
								clause_visibility_matrix[clause_id_stack[-1][0]][child_id] = 1
								clause_visibility_matrix[child_id][clause_id_stack[-1][0]] = 1
				continue

		print(clause_visibility_matrix)


if __name__ == '__main__':
	snt = "<MulSnt> <Snt> So the insurgents went to big 155 mm rounds . <REA> The first IEDs were mostly mortar rounds <REL> that failed against our armor . </REL> </REA> </Snt> <Snt> <AND> <OP> That 's a big , big shell </OP> <OP> it 's going to kill almost anything but an M1 . <TME> and when you trigger it at the right moment , </TME> </OP> </AND> </Snt> </MulSnt>"

	VisibilityMatrixConstructor.construct_clause_adjacency_matrix(snt)
	# vm = [[0.2, 0.2, 1.4, 0.2, 0.2, 1.5], [0.2, 0.2, 1.4, 0.2, 0.2, 1.5], [0.2, 0.2, 1.4, 0.2, 0.2, 1.5], [0.2, 0.2, 1.4, 0.2, 0.2, 1.5], [0.2, 0.2, 1.4, 0.2, 0.2, 1.5], [0.2, 0.2, 1.4, 0.2, 0.2, 1.5]]
	# VisibilityMatrixConstructor.reconstruct_visibility_matrix_by_max_length(vm, 24)
	# print(vm)

	# clau_adj_m = [[1, 0], [0, 1]]
	# subToken2ClauId = {0: 0, 1: 0, 2: 1, 3: 1}
	#
	# v_rate_dict = {'attn_rate_same_clause': 1.5, 'attn_rate_adjacent_clause': 1.2, 'attn_rate_nonadjacent_clause': 0.5,
	# 			   'attn_rate_keyword_token': 1.3, 'attn_rate_reen_token': 1.4}
	#
	# # vm = VisibilityMatrixConstructor.construct_visibility_matrix(clau_adj_m, subToken2ClauId, 4, v_rate_dict)
	#

