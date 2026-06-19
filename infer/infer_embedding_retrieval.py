from __future__ import annotations
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.jsonl import read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='./outputs/embedding_model')
    parser.add_argument('--data-path', default='./data/embedding/pairs.jsonl')
    parser.add_argument('--query', default='How does fallback work in LLM serving?')
    parser.add_argument('--top-k', type=int, default=3)
    args = parser.parse_args()

    model = SentenceTransformer(args.model_path)
    rows = read_jsonl(args.data_path)
    docs = sorted(set([r['positive'] for r in rows] + [r.get('negative', '') for r in rows if r.get('negative')]))
    q_emb = model.encode([args.query], normalize_embeddings=True)
    d_emb = model.encode(docs, normalize_embeddings=True)
    scores = np.matmul(q_emb, d_emb.T)[0]
    order = np.argsort(scores)[::-1][:args.top_k]
    print('Query:', args.query)
    for idx in order:
        print(f'score={scores[idx]:.4f} | {docs[idx]}')


if __name__ == '__main__':
    main()
