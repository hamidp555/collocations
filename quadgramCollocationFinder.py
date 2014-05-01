from nltk.collocations import AbstractCollocationFinder
from nltk.probability import FreqDist 
from nltk.util import ingrams
from nltk.collocations import TrigramCollocationFinder
from nltk.collocations import BigramCollocationFinder

class QuadgramCollocationFinder(AbstractCollocationFinder):
    """A tool for the finding and ranking of quadgram collocations or other
    association measures. It is often useful to use from_words() rather than
    constructing an instance directly.
    """ 
    def __init__(self, word_fd, bigram_fd, wildcard2_fd, trigram_fd, wildcard3_fd, quadgram_fd):
        """Construct a QuadgramCollocationFinder, given FreqDists for
        appearances of words, bigrams, trigrams, three words with any word between them,
        and quadgrans.
        """
        AbstractCollocationFinder.__init__(self, word_fd, quadgram_fd)
        self.wildcard3_fd = wildcard3_fd
        self.wildcard2_fd = wildcard2_fd
        self.trigram_fd = trigram_fd
        self.bigram_fd = bigram_fd
    
        
    @classmethod
    def from_words(cls, words):
        """Construct a QuadgramCollocationFinder for all quadgrams in the given
        sequence.
        """
        wfd = FreqDist()
        bfd = FreqDist()
        wild2fd = FreqDist()
        tfd = FreqDist()
        wild3fd = FreqDist()
        qfd = FreqDist()
        
        for w1, w2, w3, w4 in ingrams(words, 4, pad_right=True):
            wfd.inc(w1)
            if w2 is None:
                continue
            bfd.inc((w1, w2))
            if w3 is None:
                continue
            wild2fd.inc((w1, w3))  
            tfd.inc((w1, w2, w3))
            if w4 is None:
                continue
            wild2fd.inc((w1, w4))
            wild3fd.inc((w1, w2, w4))
            wild3fd.inc((w1, w3, w4))
            qfd.inc((w1, w2, w3, w4))
        return cls(wfd, bfd, wild2fd, tfd, wild3fd, qfd)            
      
    def bigram_finder(self):       
        return BigramCollocationFinder(self.word_fd, self.bigram_fd)
    
    def trigram_finder(self): 
        return TrigramCollocationFinder(self.word_fd, self.trigram_fd)
      
    def score_ngram(self, score_fn, w1, w2, w3, w4): 
        """Returns the score for a given quadgram using the given scoring
        function.
        """
        n_all = self.word_fd.N()
        n_iiii = self.ngram_fd[(w1, w2, w3, w4)]
        if not n_iiii:
            return
        
        n_iiix = self.trigram_fd[(w1, w2, w3)]
        n_iixi = self.wildcard3_fd[(w1, w2, w4)]
        n_ixii = self.wildcard3_fd[(w1, w3, w4)]
        n_xiii = self.trigram_fd[(w2, w3, w4)]
        
        n_iixx = self.bigram_fd[(w1, w2)]
        n_ixix = self.wildcard2_fd[(w1, w3)]
        n_ixxi = self.wildcard2_fd[(w1, w4)]
        n_xiix = self.bigram_fd[(w2, w3)]
        n_xixi = self.wildcard2_fd[(w2, w4)]
        n_xxii = self.bigram_fd[(w3, w4)]
        
        n_ixxx = self.word_fd[w1]
        n_xixx = self.word_fd[w2]
        n_xxix = self.word_fd[w3]
        n_xxxi = self.word_fd[w4]

        return score_fn(n_iiii, 
                     (n_iiix, n_iixi, n_ixii, n_xiii), 
                     (n_iixx, n_ixix, n_ixxi, n_xiix, n_xixi, n_xxii), 
                     (n_ixxx, n_xixx, n_xxix, n_xxxi), 
                     n_all)
        
        
        