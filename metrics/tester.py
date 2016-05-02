"""
Computes the BLEU, ROUGE
using the COCO metrics scripts
"""
from bleu.bleu import Bleu
from rouge.rouge import Rouge
import glob


def load_textfiles(references, hypothesis):
    hypo = {idx: [lines.strip()] for (idx, lines) in enumerate(hypothesis)}
    # take out newlines before creating dictionary
    raw_refs = [map(str.strip, r) for r in zip(references)]
    refs = {idx: rr for idx, rr in enumerate(raw_refs)}
    # sanity check that we have the same number of references as hypothesis
    if len(hypo) != len(refs):
        raise ValueError("There is a sentence number mismatch between the inputs")
    return refs, hypo


def score(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge(), "ROUGE_L"),
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

if __name__ == '__main__':
    # Feed in the directory where the hypothesis summary and true summary is stored
    hyp_file = glob.glob('hypothesis/*')
    ref_file = glob.glob('reference/*')

    BLEU_1 = 0.
    BLEU_2 = 0.
    BLEU_3 = 0.
    BLEU_4 = 0.
    ROUGE_L = 0.
    num_files = 0
    for reference_file, hypothesis_file in zip(ref_file, hyp_file):
        num_files += 1
        print reference_file, hypothesis_file

        with open(reference_file) as rf:
            reference = rf.readlines()

        with open(hypothesis_file) as hf:
            hypothesis = hf.readlines()

        print reference
        print hypothesis

        ref, hypo = load_textfiles(reference, hypothesis)
        score_map = score(ref, hypo)
        BLEU_1 += score_map['Bleu_1']
        BLEU_2 += score_map['Bleu_2']
        BLEU_3 += score_map['Bleu_3']
        BLEU_4 += score_map['Bleu_4']
        ROUGE_L += score_map['ROUGE_L']


    print 'Average Metric Score for All Review Summary Pairs:'
    print 'Bleu - 1gram:', BLEU_1/num_files
    print 'Bleu - 2gram:', BLEU_2/num_files
    print 'Bleu - 3gram:', BLEU_3/num_files
    print 'Bleu - 4gram:', BLEU_4/num_files
    print 'Rouge:', ROUGE_L/num_files
