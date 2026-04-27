# TODO: consistent reference coordinates across outputs

import os
import time
import random
import argparse
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor 

import numpy as np

#
from common import convert_fasta
from mafft import align

def hamming_ambig_partial_matches(bin_alignment):
    
    '''
    scoring matrix from: https://www.hiv.lanl.gov/content/sequence/RIP/RIPexplain.html
    '''
    
    dist_dict = {
        0B1000: {0B1000:0, 0B0001:1, 0B0010:1, 0B0100:1, 0B1001:0.50, 0B1010:0.50, 0B1100:0.50, 0B0011:1, 
              0B0101:1, 0B0110:1, 0B0111:1, 0B1110:0.66, 0B1101:0.66, 0B1011:0.66, 0B1111:0},
        0B0001: {0B1000:1, 0B0001:0, 0B0010:1, 0B0100:1, 0B1001:0.50, 0B1010:1, 0B1100:1, 0B0011:0.50, 
              0B0101:0.50, 0B0110:1, 0B0111:0.66, 0B1110:1, 0B1101:0.66, 0B1011:0.66, 0B1111:0},
        0B0010: {0B1000:1, 0B0001:1, 0B0010:0, 0B0100:1, 0B1001:1, 0B1010:0.50, 0B1100:1, 0B0011:0.50, 
              0B0101:1, 0B0110:0.50, 0B0111:0.66, 0B1110:0.66, 0B1101:1, 0B1011:0.66, 0B1111:0},
        0B0100: {0B1000:1, 0B0001:1, 0B0010:1, 0B0100:0, 0B1001:1, 0B1010:1, 0B1100:0.50, 0B0011:1, 
              0B0101:0.50, 0B0110:0.50, 0B0111:0.66, 0B1110:0.66, 0B1101:0.66, 0B1011:1, 0B1111:0},
        0B1001: {0B1000:0.50, 0B0001:0.50, 0B0010:1, 0B0100:1, 0B1001:0, 0B1010:0.50, 0B1100:0.50, 0B0011:0.50, 
              0B0101:0.50, 0B0110:1, 0B0111:0.66, 0B1110:0.66, 0B1101:0.33, 0B1011:0.33, 0B1111:0},
        0B1010: {0B1000:0.50, 0B0001:1, 0B0010:0.50, 0B0100:1, 0B1001:0.50, 0B1010:0, 0B1100:0.50, 0B0011:0.50, 
              0B0101:1, 0B0110:0.50, 0B0111:0.66, 0B1110:0.33, 0B1101:0.66, 0B1011:0.33, 0B1111:0},
        0B1100: {0B1000:0.50, 0B0001:1, 0B0010:1, 0B0100:0.50, 0B1001:0.50, 0B1010:0.50, 0B1100:0, 0B0011:1, 
              0B0101:0.50, 0B0110:0.50, 0B0111:0.66, 0B1110:0.33, 0B1101:0.33, 0B1011:0.66, 0B1111:0},
        0B0011: {0B1000:1, 0B0001:0.50, 0B0010:0.50, 0B0100:1, 0B1001:0.50, 0B1010:0.50, 0B1100:1, 0B0011:0, 
              0B0101:0.50, 0B0110:0.50, 0B0111:0.33, 0B1110:0.66, 0B1101:0.66, 0B1011:0.33, 0B1111:0},
        0B0101: {0B1000:1, 0B0001:0.50, 0B0010:1, 0B0100:0.50, 0B1001:0.50, 0B1010:1, 0B1100:0.50, 0B0011:0.50, 
              0B0101:0, 0B0110:0.50, 0B0111:0.33, 0B1110:0.66, 0B1101:0.33, 0B1011:0.66, 0B1111:0},
        0B0110: {0B1000:1, 0B0001:1, 0B0010:0.50, 0B0100:0.50, 0B1001:1, 0B1010:0.50, 0B1100:0.50, 0B0011:0.50, 
              0B0101:0.50, 0B0110:0, 0B0111:0.33, 0B1110:0.33, 0B1101:0.66, 0B1011:0.66, 0B1111:0},
        0B0111: {0B1000:1, 0B0001:0.66, 0B0010:0.66, 0B0100:0.66, 0B1001:0.66, 0B1010:0.66, 0B1100:0.66, 0B0011:0.33, 
              0B0101:0.33, 0B0110:0.33, 0B0111:0, 0B1110:0.33, 0B1101:0.33, 0B1011:0.33, 0B1111:0},
        0B1110: {0B1000:0.66, 0B0001:1, 0B0010:0.66, 0B0100:0.66, 0B1001:0.66, 0B1010:0.33, 0B1100:0.33, 0B0011:0.66, 
              0B0101:0.66, 0B0110:0.33, 0B0111:0.33, 0B1110:0, 0B1101:0.33, 0B1011:0.33, 0B1111:0},
        0B1101: {0B1000:0.66, 0B0001:0.66, 0B0010:1, 0B0100:0.66, 0B1001:0.33, 0B1010:0.66, 0B1100:0.33, 0B0011:0.66, 
              0B0101:0.33, 0B0110:0.66, 0B0111:0.33, 0B1110:0.33, 0B1101:0, 0B1011:0.33, 0B1111:0},
        0B1011: {0B1000:0.66, 0B0001:0.66, 0B0010:0.66, 0B0100:1, 0B1001:0.33, 0B1010:0.33, 0B1100:0.66, 0B0011:0.33, 
              0B0101:0.66, 0B0110:0.66, 0B0111:0.33, 0B1110:0.33, 0B1101:0.33, 0B1011:0, 0B1111:0},
        0B1111: {0B1000:0, 0B0001:0, 0B0010:0, 0B0100:0, 0B1001:0, 0B1010:0, 0B1100:0, 0B0011:0, 
              0B0101:0, 0B0110:0, 0B0111:0, 0B1110:0, 0B1101:0, 0B1011:0, 0B1111:0}
    }
    
    query = bin_alignment.pop('query')
    
    # Iterate over remaining sequences as references
    results = {}
    for h, s in bin_alignment.items():
        result = []
        for nt1, nt2 in zip(query, s):
            result.append(dist_dict[nt1][nt2])
        results.update({h: result})

    return results
    
def hamming(bin_alignment):
    """
    Convert list of lists into boolean outcomes (difference between query and reference)
    :param fasta: object returned by align() converted to bit strings
    :return: dictionary of boolean lists keyed by reference label
    """
    nts = [0B1000,0B0100,0B0010,0B0001]
    
    query = bin_alignment.pop('query')

    # Iterate over remaining sequences as references
    results = {}
    for h, s in bin_alignment.items():
        result = []
        for nt1, nt2 in zip(query, s):
            # append None if comparison of nt1 and nt2 are not evaluable, e.g. ambiguous characters or gaps
            
            
            # original rip has the option whether to 'Score multistate 
            # characters as partial matches', if so a matrix is used to score 
            # partial matches. 
            # Otherwise, it is not clearly stated how ambiguous 
            # characters are handled. Only that, if an NT and an ambiguous
            # character that also covers this NT are compared, it is scored 
            # as mismatch. It can be infered that if any NT is compared with
            # any ambiguous character, it is scored as mismatch.
            # But what happens if both compared characters are ambiguous?
            # Is it 'simply' an nt1 == nt2 comparison, irrespective of  
            # nt1 and/or nt2 being ambiguous characters?
        
        
            # In this implementation, if an NT and its anti-NT are compared
            # it is scored as mismatch.
            # If an NT and an ambiguous character 'containing' this NT co- 
            # occur, the respective position is not incorporated into the
            # distance calculation (None).
            # If two ambiguous characters co-occur, it is counted as mismatch
            # if these are mutually exclusive (e.g. M[A,C] and K[G,T]).
            # Otherwise, they are not incorporated into the distance 
            # calculation. 
            
            # bool(nt1 & nt2): if NT and its anti-NT co-occur: False
            #                  else: True
            #              -> if any set bit is shared between nt1 and nt2: True
            #                 used to catch ambiguous characters, 
            #                 NTs only share set bit if they are equal
            #              -> two different NTs always lead to False
            # 
            # (nt1 | nt2) not in nts: if same NTs co-occur: False
            #                         else: True
            #              -> used to catch case in which both nt1 and nt2 are 
            #                 NTs and are equal,
            #
            # ((not nt1) | (not nt2)): if nt1 or nt2 is a gap: True
            #                          else: False
            
            
            if bool(nt1 & nt2) & ((nt1 | nt2) not in nts) | ((not nt1) | (not nt2)):
                result.append(None)
                continue
            # return 1 if mismatch
            # return 0 is match
            # length corresponds to length of evaluable positions
            # used to calculate p-distance of sequences within window
            result.append(int(nt1!=nt2))
            
        results.update({h: result})

    return results

def create_con_of_cons(alignment):
    
    # tie breaker
    tie_break_order = 'AGTC-N'
    
    # create consensus of consensus for given consensus sequences
    con_of_cons = ''
    
    # retain only consensus sequences for con of cons
    alignment_only_cons = [tup for tup in alignment if 'CON' in tup[0]]
    
    # iterate over positions of alignment
    for i in range(len(alignment_only_cons[0][1])):
        
        # collect positions of consensus /reference sequences
        votes = [tup[1][i] for tup in alignment_only_cons]
        
        # sort vote counts 
        vote_counts = sorted([(c,votes.count(c)) for c in set(votes)],key=lambda x: x[1],reverse=True)
        
        # only keep votes with max value
        vote_counts_max = [vc for vc in vote_counts if vc[1] == vote_counts[0][1]]
        
        if len(vote_counts_max) > 1:
            # check for tie
            tie_break_ind = min([tie_break_order.index(vc[0]) for vc in  vote_counts_max])
            con_of_cons += tie_break_order[tie_break_ind]
        else:
            # there is a clear majority NT
            con_of_cons += vote_counts[0][0]

    return(con_of_cons)
    
def update_alignment(seq, reference):
    """
    Append query sequence <seq> to reference alignment and remove insertions relative to
    global consensus sequence.
    :param seq: the query sequence
    :param reference: the reference sequence
    :return: a list of [header, sequence] lists
    """
    
    # append query sequence to reference alignment   
    alignment = align(seq, reference)
        
    # original RIP performs gap stripping by default    
    # perform gap strip
    # iterate over columns of alignment
    gap_stripped_alignment = []
    skip_cols = set([])
    for i in range(len(alignment[0][1])):
        for h, s in alignment:
            if s[i] == '-':
                skip_cols.update([i])
    
    for h, s in alignment:
        s2 = [nt for i, nt in enumerate(s) if i not in skip_cols]
        gap_stripped_alignment.append([h, ''.join(s2)])
    #print('Done\n')
    
    return gap_stripped_alignment
        
def encode(fasta):
    """
    Encodes each nucleotide in a sequence using 4-bits
    :param fasta: the result of the alignment
    :return: the sequence as a bitstring where each nucleotide is encoded using a 4-bits
    """
    bin_fasta = dict(fasta)
    assert "query" in bin_fasta, "Argument <fasta> must contain 'query' entry"
    
    binary_nt = {'A': 0B1000, 'T': 0B0100, 'G': 0B0010, 'C': 0B0001,
                 'R': 0B1010, 'Y': 0B0101, 'S': 0B0011, 'W': 0B1100,
                 'K': 0B0110, 'M': 0B1001, 'B': 0B0111, 'D': 0B1110,
                 'H': 0B1101, 'V': 0B1011, 'N': 0B1111, '-': 0B0000}

    for h, s in bin_fasta.items():
        seq = []
        for nt in s:
            seq.append(binary_nt[nt])
        bin_fasta[h] = seq

    return bin_fasta

def riplike(inputs):    
    
    """
    :param seq:  query sequence
    :param reference: the alignment background
    :param window:  width of sliding window in nucleotides
    :param step:  step size of sliding window in nucleotides
    :param nrep:  number of replicates for nonparametric bootstrap sampling
    :return: list of result dictionaries in order of window position
    """
    seq, reference, window, step, nrep, partial_matches = inputs
    
    
    # set random seed for reproducibility
    random.seed(42)

    # align query and reference sequences
    # strip alignment from gaps
    alignment = update_alignment(seq, reference)
    
    query = dict(alignment)['query']  # aligned query
    seqlen = len(query)
    bin_alignment = encode(alignment)
    
    if partial_matches:
        # using partial matching of ambiguous characters
        # to calculate hamming distance
        ham = hamming_ambig_partial_matches(bin_alignment)
    else:
        # calculating hamming distance without partial
        # scoring of ambiguous characters
        ham = hamming(bin_alignment)
    
    results = []
    for centre in range(window // 2, seqlen - (window // 2), step):
        best_p, second_p = 1., 1.  # maximum p-distance
        best_ref, second_ref = None, None
        best_seq = []

        # iterate over reference genomes
        for h, s in ham.items():
            if h == 'query' or h == 'CON_OF_CONS':
                continue

            # slice window segment from reference
            s1 = s[centre - (window // 2): centre + (window // 2)]
            s2 = [x for x in s1 if x is not None]

            # calculate p-distance
            ndiff = sum(s2)
            denom = len(s2)
            
            if denom == 0:
                # reset parameters
                best_p, second_p = 1., 1.  # maximum p-distance
                best_ref, second_ref = None, None
                best_seq = []
                # no overlap!  TODO: require minimum overlap?
                continue
                
            pd = ndiff / denom

            if pd < best_p:
                # query is closer to this reference
                second_p = best_p
                second_ref = best_ref
                best_p = pd
                best_ref = h
                best_seq = s2
            elif pd < second_p:
                # replace second best
                second_p = pd
                second_ref = h
            
        result = {'centre': centre, 'best_ref': best_ref, 'best_p': best_p,
                  'second_ref': second_ref, 'second_p': None if second_ref is None else second_p}
        
        quant = 0
        if second_ref is not None and nrep > 0:
            # use nonparametric bootstrap to determine significance
            count = 0.
            n = len(best_seq)
            sample = random.choices(best_seq, k=n*nrep)
            for rep in range(nrep):
                boot = sample[rep: rep + n]
                if sum(boot) / n < second_p:
                    count += 1
            quant = count / nrep

        result.update({'quant': quant})
        results.append(result)

    return((results,alignment))

def create_report_dicts(results,query_seq,n_windows,window=400,conf_thresh=0.7,min_len=10):
    # combine windows with same best match
    
    # used to calculate coverage of match relative to query  
    query_seq_len = len(query_seq)
    
    # number of windows that have a 
    # confidence (bootstrap support) of > conf_thresh
    sum_valid_windows = 0    
    
    # track number of windows without any Ns
    windows_without_n = 0

    # initialize current report dict
    curr_report_dict = {'start': 0,
                        'end': 0,
                        'best_ref': '',
                        'avg_pdist': 0,
                        'avg_bs_support': 0,
                        'n_windows': 0,
                        }
    
    # fetch set of found subtypes
    found_subtypes = set([res['best_ref'] for res in results])
    # create dict 
    report_dicts = dict([(fs,[]) for fs in found_subtypes])
    
    # helper variable, to check whether last window(s) contained
    # Ns. If so the current report dict does not get updates.
    last_not_n = True
    
    for r_dict in results:
        # if current window contains an N, skip it
        if 'N' in query_seq[r_dict['centre']-(window//2):r_dict['centre']+(window//2)]:
            # window contains Ns
            last_not_n = False
            continue
        # count windows without Ns    
        windows_without_n += 1
        # if current and previous window match same reference
        if r_dict['best_ref'] == curr_report_dict['best_ref'] and last_not_n:
            # add p-distance
            curr_report_dict['avg_pdist'] += r_dict['best_p']
            # add avg. boot strap support
            curr_report_dict['avg_bs_support'] += r_dict['quant']
            # increase number of covered windows
            curr_report_dict['n_windows'] += 1
            # add current end of consecutive windows
            curr_report_dict['end'] = r_dict['centre']
        else:
            # calculate average p-dist  
            curr_report_dict['avg_pdist'] /= max(curr_report_dict['n_windows'],1)
            # calculate average bootstrap support 
            curr_report_dict['avg_bs_support'] /= max(curr_report_dict['n_windows'],1)
            # coverage (fraction of overall windows)
            curr_report_dict['window_coverage'] = (curr_report_dict['n_windows']/n_windows)*100
            # if window is > min_len and has bootstrap support > conf_thresh 
            # add it to report dicts
            if (conf_thresh < curr_report_dict['avg_bs_support']) and (min_len < curr_report_dict['n_windows']):
                 report_dicts[curr_report_dict['best_ref']].append(curr_report_dict)
                 # keep track of valid windows
                 sum_valid_windows += curr_report_dict['n_windows']
            
            # create new current report dict with current windows details
            curr_report_dict = {'start': r_dict['centre'],
                                'end': r_dict['centre'],
                                'best_ref': r_dict['best_ref'],
                                'avg_pdist': r_dict['best_p'],
                                'avg_bs_support': r_dict['quant'],
                                'n_windows': 1,
                                }
            # reset, if new subtype match was found 
            last_not_n = True
    
    # if last window        
    if curr_report_dict['n_windows'] > 0:
        # prepare last report dict
        # calculate average p-dist  
        curr_report_dict['avg_pdist'] /= curr_report_dict['n_windows']
        # calculate average bootstrap support 
        curr_report_dict['avg_bs_support'] /= curr_report_dict['n_windows']
        # coverage (fraction of overall windows)
        curr_report_dict['window_coverage'] = (curr_report_dict['n_windows']/n_windows)*100
    
        # append last report dict
        if (conf_thresh < curr_report_dict['avg_bs_support']) and (min_len < curr_report_dict['n_windows']):
            report_dicts[curr_report_dict['best_ref']].append(curr_report_dict)
            # keep track of valid windows
            sum_valid_windows += curr_report_dict['n_windows']

    return(report_dicts,sum_valid_windows,windows_without_n)
    
def get_best_matches(report_dicts,windows_without_n,pure_threshold=89):
    
    #
    ######
    # got error with best_match initialized as False 
    # if no subtype match was found
    # found_labels.append(','.join([bm]+om))
    # TypeError: sequence item 0: expected str instance, bool found
    best_match = ''
    ######
    #best_match = False
    best_match_coverage = 0
    best_match_alignment_coverage = 0
    best_match_avg_bs = 0
    
    other_matches = ['','','','','']
    other_matches_coverage = [0,0,0,0,0]
    other_match_alignment_coverage = [0,0,0,0,0]
    other_matches_bs = [0,0,0,0,0]
    
    for subtype,rep_dicts in report_dicts.items():
        if rep_dicts:
            #print('Subtype: ',subtype)
            #print('Windows: ')
            # initialize auxiliary variables
            current_coverage = 0
            current_alignment_coverage = 0
            current_bs = []
            
            # iterate over report windows
            for rep_dict in rep_dicts:
                # calculate window coverage, excluding windows containing Ns
                window_match = (rep_dict['n_windows']/windows_without_n)*100
                current_coverage += window_match
                #
                current_alignment_coverage += rep_dict['window_coverage']
                
                # track bs support of matched windows
                current_bs.append(rep_dict['avg_bs_support']) 
                
                print('Match subtype: ',rep_dict['best_ref'])
                print('Match start: ', rep_dict['start'])
                print('Match end: ', rep_dict['end'])
                print('Match alignment window coverage: ', rep_dict['window_coverage'])
                print('Match relative window coverage: ', window_match)
                print('Match # windows: ',rep_dict['n_windows'])
                print('Match BS support: ',rep_dict['avg_bs_support'])
                print('Match avg. p-dist: ', rep_dict['avg_pdist'],'\n')
                
            if round(current_coverage,0) >= pure_threshold:
                # works for LANL-HIV_CON_withoutP.fasta
                #best_match = subtype.split('_')[-1].split('(')[0]
                best_match = subtype.split('.')[0]
                # set variables for best match
                best_match_coverage = current_coverage
                best_match_avg_bs = sum(current_bs)/len(current_bs)
                #
                best_match_alignment_coverage = current_alignment_coverage 
            elif current_coverage > min(other_matches_coverage):
                # get subtype of other best match
                other_best_match = subtype.split('.')[0]
                # get index of minimum of other_matches_coverage
                min_ind = other_matches_coverage.index(min(other_matches_coverage))
                # replace minimum other match
                other_matches[min_ind] = other_best_match
                other_matches_coverage[min_ind] = current_coverage
                other_matches_bs[min_ind] = sum(current_bs)/len(current_bs)
                #
                other_match_alignment_coverage[min_ind] = current_alignment_coverage
    
    return(best_match,best_match_coverage,best_match_alignment_coverage,best_match_avg_bs,
           other_matches,other_matches_coverage,other_match_alignment_coverage,other_matches_bs)    
    
def create_report(query_name,results,alignment,window=400,step=1,conf_thresh=0.7,min_len=10,min_cov=1,min_bs=0.8):
    
    print('Parameters:')
    print(f'Bootstrap support threshold: {conf_thresh}')
    print(f'Min. consecutive window length: {min_len}')
    print(f'Window size: {window}')
    print(f'Step size: {step}\n')
    
    # number of windows that cover sequence
    n_windows = len(results)
    
    # query is fetched from gapstripped alignment
    query_seq = dict(alignment)['query']
    print('query_seq_len: ',len(query_seq))
    
    # aggregate best matching windows of same subtypes
    report_dicts,sum_valid_windows,windows_without_n = create_report_dicts(results,query_seq,n_windows,window=window,
                                                                           conf_thresh=conf_thresh,min_len=min_len)
    
    print(f'windows_without_n: {windows_without_n}')
    print(f'Valid windows vs. all windows: {sum_valid_windows}/{n_windows}\n')
    
    print(f'{query_name}' )
    print(f'Subtype: {query_name.split(".")[0]}')
    
    
    # get best match and other best matches
    bm,bm_cov,bm_al_cov,bm_bs,om,om_cov,om_al_cov,om_bs = get_best_matches(report_dicts,windows_without_n) 
    
    # sort other best matches by coverage
    sorted_indices = np.argsort(om_cov)[::-1]
    
    if bm:
        print(f'Best match: {bm}')
        print(f'Query window coverage: {bm_cov}')
        print(f'Alignment window coverage: {bm_al_cov}')
        print(f'Bootstrap support: {bm_bs}\n')
        
        print(f'Other best matches: {om}')
        print(f'Query window coverage: {om_cov}')
        print(f'Alignment window coverage: {om_al_cov}')
        print(f'Bootstrap support: {om_bs}\n')
        # iterate over indices sorted by window coverage
        for sort_ind in sorted_indices:
            # 
            if (om_cov[sort_ind]>min_cov) and (om_bs[sort_ind]>min_bs):
                # if second best match indicates recombinant, 
                # set best match variables to default
                
                #bm = False
                ######
                # got error with best_match initialized as False 
                # if no subtype match was found
                # found_labels.append(','.join([bm]+om))
                # TypeError: sequence item 0: expected str instance, bool found#
                bm = ''
                #####
                bm_cov = 0
                bm_bs = 0
                #
                bm_al_cov = 0
                #
                print('Recombinant!')
        # if no other match was found
        if bm:
            print('Pure!')
    else:
        print(f'Other best matches: {om}')
        print(f'Query window coverage: {om_cov}')
        print(f'Alignment window coverage: {om_al_cov}')
        print(f'Bootstrap support: {om_bs}\n')
        print('Recombinant!')
    print('---------------------------\n\n')

    # sort other matches
    om_cov = [om_cov[ind] for ind in sorted_indices if om[ind]]
    om_bs = [om_bs[ind] for ind in sorted_indices if om[ind]]
    #
    om_al_cov = [om_al_cov[ind] for ind in sorted_indices if om[ind]]
    #
    om = [om[ind] for ind in sorted_indices if om[ind]]
    
    return(bm,bm_cov,bm_al_cov,bm_bs,om,om_cov,om_al_cov,om_bs)         
            
def fetch_metadata(infile,metadata_filepath,ids):
    # fetch id for pangea data, test
    # collect data from metadata tsv file to 
    # write to updated/relabeled metadata file
    country = []
    partner = []
    status = []
    date = []
    old_labels = []
    corr_count = 0
    
    # define metadata file and path 
    metadata_file = f'{infile.split("_sequences")[0]}_metadata.tsv'
    # define path to metadata
    metadata_filepath = os.path.join(os.getcwd(),os.pardir,os.pardir,'Data',
                                     'PANGEA-HIV','metadata')
    with open(os.path.join(metadata_filepath,metadata_file),'r') as metafile:
        lines = metafile.readlines()
        for i,line in enumerate(lines[1:])
        #for i,line in enumerate(lines[1:11]):
            line_split = line.split('\t')
            country.append(line_split[1]) 
            partner.append(line_split[2])
            old_labels.append(line_split[3])
            status.append(line_split[4])
            date.append(line_split[5].replace('\n',''))
            if line_split[0] == ids[i]:
                ids[i] = f'{line_split[3]}.{ids[i]}'
                corr_count += 1
    print(f'corr_count:{corr_count}, sample_size:{len(ids)}' ) 
    return(country,partner,status,date,old_labels)   
    
def relabel(sequences,ids,alignments,results,window):
        
    # define pure subtypes
    pure_subtypes = ['A1','A2','A3','A4','A5','A6','A7','A8',
                     'B','C','D','F1','F2','G','H','J','K','L',
                     'N','O','P']
    
    
    # define auxiliary variables
    new_labels = []
    found_labels = []
    coverage = []
    bootstrap = []
    alignment_coverage = []
    seq_counter = 0
    start_report = time.time()
    
    # variables to keep track of 'true' 
    # and 'false' found subtypes
    true_subtype_counter = 0
    false_subtype_counter = 0
    true_recombinant_counter = 0
    false_recombinant_counter = 0
    
    # 
    for s,h,alignment,result in zip(sequences,ids,alignments,results):
        # get original subtype of sample
        sample_subtype = h.split(".")[0]
        # create report: get best
        bm,bm_cov,bm_al_cov,bm_bs,om,om_cov,om_al_cov,om_bs = create_report(h,result,alignment,window=window) 
        # if instance is labeled as recombinant
        # keep label -> RIPlike not suited for exact recombinant detection 
        if sample_subtype not in pure_subtypes:
            new_labels.append(sample_subtype)
            found_labels.append(','.join([bm]+om))
            #
            coverage.append(','.join([str(omc) for omc in om_cov]))
            bootstrap.append(','.join([str(ombs) for ombs in om_bs]))
            alignment_coverage.append(','.join([str(omac) for omac in om_al_cov]))
        else:
            # if major match was found, add new label
            if bm:
                new_labels.append(bm) 
                found_labels.append(','.join([bm]+om))   
                # add coverage and bootstrap support
                coverage.append(bm_cov)
                bootstrap.append(bm_bs)
                alignment_coverage.append(bm_al_cov)
            # else add found subtypes
            else:
                #
                #new_labels.append(','.join(om))
                ####
                new_labels.append('_'.join(om))
                ####
                found_labels.append(','.join(om))
                coverage.append(','.join([str(omc) for omc in om_cov]))
                bootstrap.append(','.join([str(ombs) for ombs in om_bs]))
                alignment_coverage.append(','.join([str(omac) for omac in om_al_cov]))
                
        if sample_subtype in pure_subtypes:
            if bm:
                print(f'Best match: {bm}, Subtype: {sample_subtype}')
                print('True subtype')
                true_subtype_counter += 1
            else:
                print(f'other_matches: {om}, Recombinant: {sample_subtype}')
                print('False subtype')
                false_subtype_counter += 1
        else:
            if bm:
                print(f'Best match: {bm}, Subtype: {sample_subtype}')
                print('False recombinant')
                false_recombinant_counter += 1
            else:
                print(f'other_matches: {om}, Recombinant: {sample_subtype}')
                print('True recombinant')
                true_recombinant_counter += 1
        seq_counter += 1
        print('====================================\n\n\n')  
        if seq_counter % 10 == 0:
            print('################################')
            print(f'True Subtypes  {true_subtype_counter} of {true_subtype_counter+false_subtype_counter}')
            print(f'True Recombinants  {true_recombinant_counter} of {true_recombinant_counter+false_recombinant_counter}')
            print(f'check: {true_subtype_counter+false_recombinant_counter+false_subtype_counter+true_recombinant_counter} / {seq_counter}')
            print('################################\n\n')     
        #######
    end_report = time.time()
    
    print('Report time: ',end_report-start_report)
    
    return(new_labels,found_labels,coverage,bootstrap,alignment_coverage)

def main():
    parser = argparse.ArgumentParser(
        description='An approximate implementation of the Recombinant Identification '
                    'program by the Los Alamos National Laboratory.'
    )
    parser.add_argument('infile', type=str,#argparse.FileType('r'),
                        help='<input> FASTA file containing sequences to process.')
    #parser.add_argument('outfile', type=str,#argparse.FileType('w'),
    #                    help='<output> file to write CSV results.')
    parser.add_argument('-window', type=int, default=250,#250,#250,#400
                        help='<optional, int> Window size for p-distances.')
    parser.add_argument('-step', type=int, default=1,
                        # step size 1 is default in LANL RIP 3.0
                        help='<optional, int> Window step size.')
    parser.add_argument('-nrep', type=int, default=100,
                        help='<optional, int> Number of bootstrap replicates.')
    parser.add_argument('-custombg', type=argparse.FileType('r'),
                        help='<optional> FASTA file to be used as the alignment background')

    args = parser.parse_args()

    if args.custombg:
        ref_seq = args.custombg.name
    else:
        ref_seq_filename = 'HIV1_RIP_2023_genome_DNA_Mgroup_CONSandRegularIfNoCon_withNOP_noRecombinants.fasta'
        ref_seq = os.path.join(os.getcwd(),'ref_genomes',ref_seq_filename)

    with open(ref_seq) as handle:
        reference = convert_fasta(handle)
    
    # reading fasta input sequences file
    ## change infile
    infile = os.path.join(os.getcwd(),os.pardir,os.pardir,'Data',
                                     'PANGEA-HIV','sequences',args.infile)
    handle = open(infile, 'r')
    
    fasta = convert_fasta(handle)
    # splitting sequences and IDs
    sequences = [tup[1] for tup in fasta]
    ids = [tup[0] for tup in fasta]
    
    ####
    #sequences = sequences[:10]
    #ids = ids[:10]
    ####
    
    # define metadata filepath
    metadata_filepath = os.path.join(os.getcwd(),os.pardir,os.pardir,'Data',
                                     'PANGEA-HIV','metadata')
    # get metadata for PANGEA sequences                    
    country,partner,status,date,old_labels = fetch_metadata(args.infile,metadata_filepath,ids)
    
    # set up tuple as input for RIPlike 
    inputs = [(s,reference,args.window,args.step,args.nrep,False) for s in sequences]
    
    # perform RIPlike in parallel
    start_par1 = time.time()
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(riplike, inputs))
    end_par1 = time.time()
    print('ProcessPoolExecutor: ', end_par1 -start_par1)
    
    # split results into alignments and 
    alignments = [tup[1] for tup in results]
    results = [tup[0] for tup in results]
    
    # create reports(best matches, coverages, bootstrap supports) 
    # and collate new labels for query sequences
    new_labels,found_labels,coverage,bootstrap,alignment_coverage = relabel(sequences,ids,alignments,results,args.window) 
    
    
    # write new metadata tsv file
    new_metadata_filename = f'{args.infile.split("_sequences")[0]}_metadata_relabeled.tsv'
    with open(os.path.join(metadata_filepath,new_metadata_filename), 'w') as outfile:
        outfile.write('sequence_id\tcountry\tpartner\tsubtype\tstatus\tdate\tfound_subtypes\tquery_coverage\talignment_coverage\tbootstrap\n')
        for i,nl in enumerate(new_labels):
            outstring = f'{ids[i].split(".")[1]}\t{country[i]}\t{partner[i]}\t{nl}\t{status[i]}\t{date[i]}\t{found_labels[i]}\t{coverage[i]}\t{alignment_coverage[i]}\t{bootstrap[i]}\n'
            outfile.write(outstring)
            

if __name__ == '__main__':
    main()
