from nltk.collocations import AbstractCollocationFinder
from nltk.probability import FreqDist 
from nltk.util import ingrams
from nltk.collocations import TrigramCollocationFinder
from nltk.collocations import BigramCollocationFinder
from com.base.quadgramCollocationFinder import QuadgramCollocationFinder

class PentagramCollocationFinder(AbstractCollocationFinder):
    """A tool for the finding and ranking of pentagram collocations or other
    association measures. It is often useful to use from_words() rather than
    constructing an instance directly.
    """ 
    def __init__(self, word_fd, bigram_fd, wildcard2_fd, trigram_fd, wildcard3_fd, quadgram_fd, wildcard4_fd, pentagram_fd):
        """Construct a PentagramCollocationFinder, given FreqDists for
        appearances of words, bigrams, trigrams, three words with any word between them, quadgrams, four words with any word between them
        and pentagram.
        """
        AbstractCollocationFinder.__init__(self, word_fd, pentagram_fd)
        self.wildcard4_fd = wildcard4_fd
        self.wildcard3_fd = wildcard3_fd
        self.wildcard2_fd = wildcard2_fd
        self.quadgram_fd = quadgram_fd
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
        wild4fd = FreqDist()
        pfd = FreqDist()
        
        for w1, w2, w3, w4, w5 in ingrams(words, 5, pad_right=True):
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
            if w5 is None:
                continue
            wild2fd.inc((w1, w5))
            wild3fd.inc((w1, w2, w5))
            wild3fd.inc((w1, w3, w5)) 
            wild3fd.inc((w1, w4, w5))   
            wild4fd.inc((w1, w3, w4, w5))
            wild4fd.inc((w1, w2, w4, w5))
            wild4fd.inc((w1, w2, w3, w5))
            pfd.inc((w1, w2, w3, w4, w5))  
        return cls(wfd, bfd, wild2fd, tfd, wild3fd, qfd, wild4fd, pfd)            
      
    def bigram_finder(self):       
        return BigramCollocationFinder(self.word_fd, self.bigram_fd)
    
    def trigram_finder(self): 
        return TrigramCollocationFinder(self.word_fd, self.trigram_fd)
    
    def quadgram_finder(self): 
        return QuadgramCollocationFinder(self.word_fd, self.quadgram_fd)
    
    def score_ngram(self, score_fn, w1, w2, w3, w4, w5): 
        """Returns the score for a given quadgram using the given scoring
        function.
        """
        n_all = self.word_fd.N()
        n_iiiii = self.ngram_fd[(w1, w2, w3, w4, w5)]
        if not n_iiiii:
            return
        
        n_xiiii = self.quadgram_fd[(w2, w3, w4, w5)]
        n_ixiii = self.wildcard4_fd[(w1, w3, w4, w5)]
        n_iixii = self.wildcard4_fd[(w1, w2, w4, w5)]
        n_iiixi = self.wildcard4_fd[(w1, w2, w3, w5)]
        n_iiiix = self.quadgram_fd[(w1, w2, w3, w4)]
        
        n_xxiii = self.trigram_fd[(w3, w4, w5)]
        n_ixxii = self.wildcard3_fd[(w1, w4, w5)]
        n_iixxi = self.wildcard3_fd[(w1, w2, w5)]
        n_iiixx = self.trigram_fd[(w1, w2, w3)]
        n_xixii = self.wildcard3_fd[(w2, w4, w5)]
        n_xiixi = self.wildcard3_fd[(w2, w3, w5)]
        n_xiiix = self.trigram_fd[(w2, w3, w4)]
        n_iixix = self.wildcard3_fd[(w1, w2, w4)]
        n_ixiix = self.wildcard3_fd[(w1, w3, w4)]
        n_ixixi = self.wildcard3_fd[(w1, w3, w5)]
         
        n_xxxii = self.bigram_fd[(w4, w5)]
        n_iixxx = self.bigram_fd[(w1, w2)]
        n_xixxi = self.wildcard2_fd[(w2, w5)]
        n_xiixx = self.bigram_fd[(w2, w3)]
        n_xixix = self.wildcard2_fd[(w2, w4)]
        n_ixxxi = self.wildcard2_fd[(w1, w5)]
        n_ixxix = self.wildcard2_fd[(w1, w4)]
        n_ixixx = self.wildcard2_fd[(w1, w3)]
        n_xxixi = self.wildcard2_fd[(w3, w5)]
        n_xxiix = self.bigram_fd[(w3, w4)]

        n_ixxxx = self.word_fd[w1]
        n_xxxxi = self.word_fd[w5]
        n_xxxix = self.word_fd[w4]
        n_xxixx = self.word_fd[w3]
        n_xixxx = self.word_fd[w2]

        return score_fn(n_iiiii,
                     (n_xiiii, n_ixiii, n_iixii, n_iiixi, n_iiiix),
                     (n_xxiii, n_ixxii, n_iixxi, n_iiixx, n_xixii, n_xiixi, n_xiiix, n_iixix, n_ixiix, n_ixixi),
                     (n_xxxii, n_iixxx, n_xixxi, n_xiixx, n_xixix, n_ixxxi, n_ixxix, n_ixixx, n_xxixi, n_xxiix),
                     (n_ixxxx, n_xxxxi, n_xxxix, n_xxixx, n_xixxx),
                     n_all)
        
        
        