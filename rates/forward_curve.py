from datetime import datetime as dt
import warnings
warnings.filterwarnings("ignore")

from data.loader import tsd_loader as dl
from data.term_data import TermStructureData as tsd
from data.registry import DEFAULT_TERM_STRUCTURE_LOADER

class ForwardCurve:
    def __init__(self, 
                loader: dl = DEFAULT_TERM_STRUCTURE_LOADER,
                sdate: dt = None, 
                edate: dt = None
    ):
        
        self.sdate = sdate
        self.edate = edate
        self.loader = loader
        self.curve = None
    
    def __repr__(self):
        pass
        
    def compute(self) -> tsd:
        if not self.curve:
            d = self.loader.load(self.sdate, self.edate)
            self.curve = self.loader.build_curve(d)
        return self.curve