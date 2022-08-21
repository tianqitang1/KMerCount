import numpy as np
from multiprocessing import Pool
from Bio import SeqIO
from itertools import product
from functools import partial

class KMerCount:
    def __init__(self, k, thread=1, split_len=5000) -> None:
        self.k = k

        self.thread = thread
        self.pool = Pool(self.thread)

        self.split_len = split_len

        nucleotides = ["A", "C", "G", "T"]
        self.kmer_to_index = {}
        for i, mer in enumerate(product(nucleotides, repeat=k)):
            self.kmer_to_index["".join(list(mer))] = i

    @staticmethod
    def chunkstring(string, length):
        return (string[0+i:length+i] for i in range(0, len(string), length))
    
    def count(self, filename, revcomp=False, fragmented=False):
        sparse_mode = self.k > 5

        fragments = []
        for record in SeqIO.parse(filename, 'fasta'):
            fragments.extend(list(self.chunkstring(str(record.seq).upper(), self.split_len)))
            if revcomp:
                fragments.extend(list(self.chunkstring(str(record.seq.reverse_complement()).upper(), self.split_len)))
        count_preset = partial(
            self.count_single_frag,
            k=self.k,
            kmer_to_index=self.kmer_to_index
        )
        kmer_count = self.pool.map(count_preset, fragments)
        kmer_count = np.vstack(kmer_count)
        if not fragmented:
            kmer_count = kmer_count.sum(axis=0)
        # kmer_count_all = kmer_count.sum(axis=0)
        # return kmer_count, kmer_count_all

        if not sparse_mode:
            pass

        return kmer_count

    @staticmethod
    def count_single_frag(frag, k, kmer_to_index):
        kmer_count = np.zeros(4 ** k, dtype=np.int)
        if k > len(frag):
            return kmer_count
        for i in range(len(frag)-4):
            kmer_count[kmer_to_index[frag[i:i+k]]] += 1
        return kmer_count
        