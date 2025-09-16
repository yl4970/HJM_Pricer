from dataclasses import dataclass
from typing import Sequence, NamedTuple, Literal, Optional

class CashFlow(NamedTuple):
    pay_date: float       # year fraction from t0, or a datetime you convert later
    fixing_time: float
    accrual: float
    start: float
    end: float

PayoffType = Literal["cap", "floor"]
ModelHint  = Literal["auto", "lognormal", "normal"]

@dataclass(frozen=True, slots=True)
class CapFloor:
    strike: float
    notional: float
    schedule: Sequence[CashFlow]        # prebuilt cashflows on a floating index
    payoff_type: PayoffType = "cap"
    index_name: Optional[str] = None    # e.g., "SOFR-3M", "EURIBOR-6M"
    pay_conv: str = "ACT/360"
    model_hint: ModelHint = "auto"
    
    def __post_init__(self):
        if self.notional <= 0:
            raise ValueError("Notional must be non-negative.")
        if self.strike < 0:
            raise ValueError("Strike must be positive.")
        if not self.schedule:
            raise ValueError("Schedule cannot be empty.")
        if any(cf.accrual <= 0 for cf in self.schedule):
            raise ValueError("all accruals must be > 0")
    
    @property
    def sign(self) -> int:
        return 1 if self.payoff_type == "cap" else -1
    
    @staticmethod
    def choose_model(F: float, K: float, sigma: float, T: float, hint: str) -> str:
        if hint != "auto":
            return hint
        if F <= 0.0:
            return "normal"
        z = abs(F - K) / (sigma * T**0.5) if sigma > 0 and T > 0 else float('inf')
        if F < 0.01 and z < 1.0:
            return "normal"
        return "lognormal"