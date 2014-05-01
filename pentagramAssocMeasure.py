from nltk.metrics import NgramAssocMeasures

class PentagramAssocMeasures(NgramAssocMeasures):
   
    _n = 5
    
    @staticmethod
    def _contingency(n_iiiii,
                     (n_xiiii, n_ixiii, n_iixii, n_iiixi, n_iiiix),
                     (n_xxiii, n_ixxii, n_iixxi, n_iiixx, n_xixii, n_xiixi, n_xiiix, n_iixix, n_ixiix, n_ixixi),
                     (n_xxxii, n_iixxx, n_xixxi, n_xiixx, n_xixix, n_ixxxi, n_ixxix, n_ixixx, n_xxixi, n_xxiix),
                     (n_ixxxx, n_xxxxi, n_xxxix, n_xxixx, n_xixxx),
                     n_xxxxx):
        
        n_oiiii= n_xiiii - n_iiiii
        n_ioiii= n_ixiii - n_iiiii
        n_iioii= n_iixii - n_iiiii
        n_iiioi= n_iiixi - n_iiiii 
        n_iiiio= n_iiiix - n_iiiii
        
        n_ooiii= n_xxiii - n_iiiii - n_oiiii - n_ioiii  
        n_iooii= n_ixxii - n_iiiii - n_iioii - n_ioiii 
        n_iiooi= n_iixxi - n_iiiii - n_iioii - n_iiioi 
        n_iiioo= n_iiixx - n_iiiii - n_iiioi - n_iiiio 
        n_oioii= n_xixii - n_iiiii - n_iioii - n_oiiii
        n_oiioi= n_xiixi - n_iiiii - n_iiioi - n_oiiii
        n_oiiio= n_xiiix - n_iiiii - n_iiiio - n_oiiii 
        n_iioio= n_iixix - n_iiiii - n_iioii - n_iiiio 
        n_ioiio= n_ixiix - n_iiiii - n_iiiio - n_ioiii 
        n_ioioi= n_ixixi - n_iiiii - n_iiioi - n_ioiii 
           
        n_oooii= n_xxxii - n_iiiii - n_iioii - n_oiiii - n_oioii - n_ioiii - n_iooii - n_ooiii 
        n_iiooo= n_iixxx - n_iiiii - n_iioii - n_iiioi - n_iiiio - n_iiooi - n_iioio - n_iiioo 
        n_oiooi= n_xixxi - n_iiiii - n_iioii - n_iiioi - n_iiooi - n_oiiii - n_oioii - n_oiioi 
        n_oiioo= n_xiixx - n_iiiii - n_iiioi - n_iiiio - n_iiioo - n_oiiii - n_oiioi - n_oiiio 
        n_oioio= n_xixix - n_iiiii - n_iioii - n_iiiio - n_iioio - n_oiiii - n_oioii - n_oiiio
        n_ioooi= n_ixxxi - n_iiiii - n_iioii - n_iiioi - n_iiooi - n_ioiii - n_iooii - n_ioioi 
        n_iooio= n_ixxix - n_iiiii - n_iioii - n_iiiio - n_iioio - n_ioiii - n_iooii - n_ioiio 
        n_ioioo= n_ixixx - n_iiiii - n_iiioi - n_iiiio - n_iiioo - n_ioiii - n_ioioi - n_ioiio 
        n_ooioi= n_xxixi - n_iiiii - n_iiioi - n_oiiii - n_oiioi - n_ioiii - n_ioioi - n_ooiii 
        n_ooiio= n_xxiix - n_iiiii - n_iiiio - n_oiiii - n_oiiio - n_ioiii - n_ioiio - n_ooiii 
        
        n_ioooo= n_ixxxx - n_iiiii - n_iioii - n_iiioi - n_iiiio - n_iiooi - n_iioio - n_iiioo - n_iiooo - n_ioiii - n_iooii - n_ioioi - n_ioiio - n_ioooi - n_iooio - n_ioioo
        n_ooooi= n_xxxxi - n_iiiii - n_oiiii - n_ioiii - n_ooiii - n_iioii - n_oioii - n_iooii - n_oooii - n_iiioi - n_oiioi - n_ioioi - n_ooioi - n_iiooi - n_oiooi - n_ioooi
        n_oooio= n_xxxix - n_iiiii - n_oiiii - n_ioiii - n_ooiii - n_iioii - n_oioii - n_iooii - n_oooii - n_iiiio - n_oiiio - n_ioiio - n_ooiio - n_iioio - n_oioio - n_iooio
        n_ooioo= n_xxixx - n_iiiii - n_oiiii - n_ioiii - n_ooiii - n_iiioi - n_oiioi - n_ioioi - n_ooioi - n_iiiio - n_oiiio - n_ioiio - n_ooiio - n_iiioo - n_oiioo - n_ioioo 
        n_oiooo= n_xixxx - n_iiiii - n_iioii - n_iiioi - n_iiiio - n_iiooi - n_iioio - n_iiioo - n_iiooo - n_oiiii - n_oioii - n_oiioi - n_oiiio - n_oiooi - n_oioio - n_oiioo 
       
        n_ooooo= n_xxxxx - n_iiiii - n_iioii - n_iiioi - n_iiiio - n_iiooi - n_iioio - n_iiioo - n_iiooo - n_oiiii - n_oioii - n_oiioi - n_oiiio - n_oiooi - n_oioio - n_oiioo - n_oiooo - n_ioiii - n_iooii - n_ioioi - n_ioiio - n_ioooi - n_iooio - n_ioioo - n_ioooo - n_ooiii - n_oooii - n_ooioi - n_ooiio - n_ooooi - n_oooio - n_ooioo
        
        return(n_iiiii,
               n_oiiii, n_ioiii, n_iioii,n_iiioi, n_iiiio,
               n_ooiii, n_iooii, n_iiooi, n_iiioo, n_oioii, n_oiioi, n_oiiio, n_iioio, n_ioiio, n_ioioi,
               n_oooii, n_iiooo, n_oiooi, n_oiioo, n_oioio, n_ioooi, n_iooio, n_ioioo, n_ooioi, n_ooiio,
               n_ioooo, n_ooooi, n_oooio, n_ooioo, n_oiooo, n_ooooo)
    
    @staticmethod
    def _marginals(*contingency):
        """Calculates values of contingency table marginals from its values."""
        n_iiiii, n_oiiii, n_ioiii, n_iioii,n_iiioi, n_iiiio, n_ooiii, n_iooii, n_iiooi, n_iiioo, n_oioii, n_oiioi, n_oiiio, n_iioio, n_ioiio, n_ioioi, n_oooii, n_iiooo, n_oiooi, n_oiioo, n_oioio, n_ioooi, n_iooio, n_ioioo, n_ooioi, n_ooiio, n_ioooo, n_ooooi, n_oooio, n_ooioo, n_oiooo, n_ooooo = contingency

        return(n_iiiii,
               (n_oiiii + n_iiiii,
                n_ioiii + n_iiiii,
                n_iioii + n_iiiii,
                n_iiioi + n_iiiii,
                n_iiiio + n_iiiii),
               (n_ooiii + n_iiiii + n_oiiii + n_ioiii,  
                n_iooii + n_iiiii + n_iioii + n_ioiii,
                n_iiooi + n_iiiii + n_iioii + n_iiioi, 
                n_iiioo + n_iiiii + n_iiioi + n_iiiio, 
                n_oioii + n_iiiii + n_iioii + n_oiiii,
                n_oiioi + n_iiiii + n_iiioi + n_oiiii,
                n_oiiio + n_iiiii + n_iiiio + n_oiiii,
                n_iioio + n_iiiii + n_iioii + n_iiiio, 
                n_ioiio + n_iiiii + n_iiiio + n_ioiii, 
                n_ioioi + n_iiiii + n_iiioi + n_ioiii ),
               (n_oooii + n_iiiii + n_iioii + n_oiiii + n_oioii + n_ioiii + n_iooii + n_ooiii, 
                n_iiooo + n_iiiii + n_iioii + n_iiioi + n_iiiio + n_iiooi + n_iioio + n_iiioo, 
                n_oiooi + n_iiiii + n_iioii + n_iiioi + n_iiooi + n_oiiii + n_oioii + n_oiioi, 
                n_oiioo + n_iiiii + n_iiioi + n_iiiio + n_iiioo + n_oiiii + n_oiioi + n_oiiio, 
                n_oioio + n_iiiii + n_iioii + n_iiiio + n_iioio + n_oiiii + n_oioii + n_oiiio,
                n_ioooi + n_iiiii + n_iioii + n_iiioi + n_iiooi + n_ioiii + n_iooii + n_ioioi, 
                n_iooio + n_iiiii + n_iioii + n_iiiio + n_iioio + n_ioiii + n_iooii + n_ioiio, 
                n_ioioo + n_iiiii + n_iiioi + n_iiiio + n_iiioo + n_ioiii + n_ioioi + n_ioiio, 
                n_ooioi + n_iiiii + n_iiioi + n_oiiii + n_oiioi + n_ioiii + n_ioioi + n_ooiii, 
                n_ooiio + n_iiiii + n_iiiio + n_oiiii + n_oiiio + n_ioiii + n_ioiio + n_ooiii),
               (n_ioooo + n_iiiii + n_iioii + n_iiioi + n_iiiio + n_iiooi + n_iioio + n_iiioo + n_iiooo + n_ioiii + n_iooii + n_ioioi + n_ioiio + n_ioooi + n_iooio + n_ioioo,
                n_ooooi + n_iiiii + n_oiiii + n_ioiii + n_ooiii + n_iioii + n_oioii + n_iooii + n_oooii + n_iiioi + n_oiioi + n_ioioi + n_ooioi + n_iiooi + n_oiooi + n_ioooi,
                n_oooio + n_iiiii + n_oiiii + n_ioiii + n_ooiii + n_iioii + n_oioii + n_iooii + n_oooii + n_iiiio + n_oiiio + n_ioiio + n_ooiio + n_iioio + n_oioio + n_iooio,
                n_ooioo + n_iiiii + n_oiiii + n_ioiii + n_ooiii + n_iiioi + n_oiioi + n_ioioi + n_ooioi + n_iiiio + n_oiiio + n_ioiio + n_ooiio + n_iiioo + n_oiioo + n_ioioo,
                n_oiooo + n_iiiii + n_iioii + n_iiioi + n_iiiio + n_iiooi + n_iioio + n_iiioo + n_iiooo + n_oiiii + n_oioii + n_oiioi + n_oiiio + n_oiooi + n_oioio + n_oiioo ),
               sum(contingency))