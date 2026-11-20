"""WY-first acceptance with measured density diagnostics as a secondary key."""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class WYSelection:
    density_scales: tuple
    tolerance: float = 1e-8

    def __post_init__(self):
        object.__setattr__(self,'density_scales',tuple(self.density_scales))
        if (len(self.density_scales)!=2 or any(not isfinite(s) or s<=0 for s in self.density_scales)
                or not isfinite(self.tolerance) or self.tolerance<0):
            raise ValueError('Require two positive density scales and a finite WY tolerance')

    @property
    def configuration(self):
        return {'kind':'wy_first','density_scales':list(self.density_scales),'tolerance':self.tolerance}

    def state(self,evaluation):
        samples=list(evaluation.samples.values())
        if not samples:raise ValueError('WY selection requires measured density diagnostics')
        try:
            values=[.5*(s['density_l1']/self.density_scales[0]
                        +s['density_error']**2/self.density_scales[1]) for s in samples]
        except (TypeError,KeyError) as exc:
            raise ValueError('WY selection requires measured L1/L2 diagnostics') from exc
        if not all(isfinite(v) for v in values):
            raise ValueError('Density diagnostics must be finite')
        return {'secondary_loss':sum(values)/len(values)}

    def better(self,evaluation,incumbent):
        if not evaluation.valid or evaluation.reward is None or not isfinite(evaluation.reward):
            return False
        secondary=self.state(evaluation)['secondary_loss']
        delta=evaluation.reward-incumbent['reward']
        return (delta>self.tolerance or
                (delta>=0 and secondary<incumbent.get('secondary_loss',1.)-1e-12))
