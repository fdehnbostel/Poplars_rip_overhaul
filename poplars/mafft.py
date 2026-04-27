#!/usr/bin/env python3
"""
Wrapper script to ensure that MAFFT use system-specific binaries
MAFFT versions: 7.427 (Linux), 7.429 (Windows), 7.427 (Mac OS X)
"""

import sys
import os
import subprocess
import logging
import argparse
import tempfile

from common import convert_fasta, convert_clustal


def align(query, reference, clustal=False):
    """
    Python wrapper to MAFFT for pairwise/multiple sequence alignment
    :param query: The query sequence
    :param clustal: output Clustal format (default: FASTA)
    :param reference: Either a reference sequence (str) or a list from convert_fasta()
    """
    handle = tempfile.NamedTemporaryFile('w+', delete=False)
    if type(reference) == str:
        handle.write('>reference\n{}\n'.format(reference))
    elif type(reference) == list:
        for h, s in reference:
            handle.write('>{}\n{}\n'.format(h, s))

    handle.write('>query\n{}\n'.format(query))
    handle.close()

    # Path to the temporary query file for MAFFT
    raw_output = run_mafft(handle.name, clustal)

    output = raw_output.decode('utf-8')

    if not clustal:
        aln_output = convert_fasta(output.split('\n'))
    else:
        aln_output = convert_clustal(output.split('\n'))

    return aln_output


def run_mafft(file_path, clustal=False):
    """
    Runs MAFFT on Linux, Windows, or Mac OS X and produces FASTA output
    :param file_path: path to the input file
    :param clustal: output Clustal format (default: FASTA)
    :return output: the output of MAFFT
    """

    try:
        sys.platform.startswith("linux") or sys.platform.startswith("win") or sys.platform == "darwin"

    except OSError:
        print("OSError: {} is not supported".format(sys.platform))

    # Path to find MAFFT
    script_path = os.path.dirname(os.path.abspath(__file__))

    if sys.platform.startswith("linux"):
        bin_path = os.path.join(script_path, 'bin/mafft-linux64/mafft.bat')

    elif sys.platform.startswith("win"):
        bin_path = os.path.join(script_path, 'bin/mafft-win/mafft.bat')

    else:
        bin_path = os.path.join(script_path, 'bin/mafft-mac/mafft.bat')

    if not os.path.isfile(bin_path):
        logging.error("No file exists.")

    if not clustal:
        if sys.platform.startswith("win"):
            raw_output = subprocess.check_output([bin_path, '--quiet', file_path],
                                                 shell=False, stderr=subprocess.DEVNULL)
        else:
            raw_output = subprocess.check_output([bin_path, '--quiet', file_path],
                                                 shell=False, stderr=subprocess.STDOUT)

    else:
        if sys.platform.startswith("win"):
            raw_output = subprocess.check_output([bin_path, '--quiet', '--clustalout', file_path],
                                                 shell=False, stderr=subprocess.DEVNULL)
        else:
            raw_output = subprocess.check_output([bin_path, '--quiet', '--clustalout', file_path],
                                                 shell=False, stderr=subprocess.STDOUT)

    return raw_output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Wrapper to run MAFFT"
    )
    parser.add_argument("infile", help="The input file", type=argparse.FileType('r'))

    return parser.parse_args()


def main():
    args = parse_args()

    run_mafft(args.infile)


if __name__ == '__main__':
    main()
