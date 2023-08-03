from pathlib import Path

import penman
import torch

from spring_amr import ROOT
from spring_amr.evaluation import predict_amrs, compute_smatch
from spring_amr.penman import encode
from spring_amr.tokenization_visibility_matrix import VisibilityMatrixConstructor
from spring_amr.utils import instantiate_loader, instantiate_model_and_tokenizer

if __name__ == '__main__':

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="Script to predict AMR graphs given sentences. LDC format as input.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--datasets', type=str, required=True, nargs='+',
        help="Required. One or more glob patterns to use to load amr files.")
    parser.add_argument('--checkpoint', type=str, required=True,
        help="Required. Checkpoint to restore.")
    parser.add_argument('--model', type=str, default='facebook/bart-large',
        help="Model config to use to load the model class.")
    parser.add_argument('--beam-size', type=int, default=1,
        help="Beam size.")
    parser.add_argument('--batch-size', type=int, default=1000,
        help="Batch size (as number of linearized graph tokens per batch).")
    parser.add_argument('--device', type=str, default='cuda',
        help="Device. 'cpu', 'cuda', 'cuda:<n>'.")
    parser.add_argument('--pred-path', type=Path, default=ROOT / 'data/tmp/inf-pred.txt',
        help="Where to write predictions.")
    parser.add_argument('--gold-path', type=Path, default=ROOT / 'data/tmp/inf-gold.txt',
        help="Where to write the gold file.")
    parser.add_argument('--use-recategorization', action='store_true',
        help="Predict using Zhang recategorization on top of our linearization (requires recategorized sentences in input).")
    parser.add_argument('--penman-linearization', action='store_true',
        help="Predict using PENMAN linearization instead of ours.")
    parser.add_argument('--use-pointer-tokens', action='store_true')
    parser.add_argument('--raw-graph', action='store_true')
    parser.add_argument('--restore-name-ops', action='store_true')
    parser.add_argument('--return-all', action='store_true')

    parser.add_argument('--feature_path', type=str)
    parser.add_argument('--clause_token_visibility', action='store_true', default=False)
    parser.add_argument('--clause_token_inf_mask', action='store_true', default=False)
    parser.add_argument('--attn_rate_same_clause', type=float)
    parser.add_argument('--attn_rate_adjacent_clause', type=float)
    parser.add_argument('--attn_rate_nonadjacent_clause', type=float)
    parser.add_argument('--attn_rate_keyword_token', type=float)
    parser.add_argument('--attn_rate_reen_token', type=float)
    # parser.add_argument('--attn_rate_same_clause', type=int)
    # parser.add_argument('--attn_rate_adjacent_clause', type=int)
    # parser.add_argument('--attn_rate_nonadjacent_clause', type=int)
    # parser.add_argument('--attn_rate_keyword_token', type=int)
    # parser.add_argument('--attn_rate_reen_token', type=int)
    parser.add_argument('--clause_attn_head_num', type=int)
    parser.add_argument('--clause_attn_layer_id', type=str)

    args = parser.parse_args()

    visibility_rate_dict = {VisibilityMatrixConstructor.SAME_CLAUSE: args.attn_rate_same_clause,
                            VisibilityMatrixConstructor.ADJACENT_CLAUSE: args.attn_rate_adjacent_clause,
                            VisibilityMatrixConstructor.NONADJACENT_CLAUSE: args.attn_rate_nonadjacent_clause,
                            VisibilityMatrixConstructor.KEYWORD_TOKEN: args.attn_rate_keyword_token,
                            VisibilityMatrixConstructor.REENTRANT_TOKEN: args.attn_rate_reen_token
                            }

    device = torch.device(args.device)
    model, tokenizer = instantiate_model_and_tokenizer(
        args.model,
        dropout=0.,
        attention_dropout=0.,
        penman_linearization=args.penman_linearization,
        use_pointer_tokens=args.use_pointer_tokens,
        raw_graph=args.raw_graph,
        clause_token_visibility=args.clause_token_visibility,
        clause_token_inf_mask=args.clause_token_inf_mask,
        visibility_rate_dict=visibility_rate_dict,
        clause_attn_head_num=args.clause_attn_head_num,
        clause_attn_layer_id=args.clause_attn_layer_id,
    )
    model.amr_mode = True
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model'])
    model.to(device)

    gold_path = args.gold_path
    pred_path = args.pred_path
    loader = instantiate_loader(
        args.datasets,
        tokenizer,
        batch_size=args.batch_size,
        evaluation=True, out=gold_path,
        use_recategorization=args.use_recategorization,
        input_features_path=args.feature_path,
    )
    loader.device = device

    graphs = predict_amrs(
        loader,
        model,
        tokenizer,
        beam_size=args.beam_size,
        restore_name_ops=args.restore_name_ops,
        return_all=args.return_all,
    )
    if args.return_all:
        graphs = [g for gg in graphs for g in gg]

    for g in graphs:
        g.metadata.pop('feature_tuple')

    pieces = [encode(g) for g in graphs]
    pred_path.write_text('\n\n'.join(pieces))

    if not args.return_all:
        score = compute_smatch(gold_path, pred_path)
        print(f'Smatch: {score:.3f}')
