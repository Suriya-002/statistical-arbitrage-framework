import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class SectorExposure:
    sector: str
    long_exposure: float
    short_exposure: float
    net_exposure: float
    gross_exposure: float


class SectorConstraintManager:
    """Enforce sector-level exposure limits on the pairs portfolio."""

    def __init__(self, max_sector_net=0.15, max_sector_gross=0.30, sector_map=None):
        self.max_sector_net = max_sector_net
        self.max_sector_gross = max_sector_gross
        self.sector_map = sector_map or {}

    def compute_sector_exposures(self, active_pairs, position_sizes):
        sector_long, sector_short = {}, {}
        for pair in active_pairs:
            pk = f"{pair['ticker_y']}/{pair['ticker_x']}"
            size = position_sizes.get(pk, 0)
            if size == 0: continue
            ys = self.sector_map.get(pair['ticker_y'], 'Unknown')
            xs = self.sector_map.get(pair['ticker_x'], 'Unknown')
            sector_long[ys] = sector_long.get(ys, 0) + abs(size)
            sector_short[xs] = sector_short.get(xs, 0) + abs(size)

        exposures = {}
        for s in set(list(sector_long) + list(sector_short)):
            l, sh = sector_long.get(s, 0), sector_short.get(s, 0)
            exposures[s] = SectorExposure(s, l, sh, l-sh, l+sh)
        return exposures

    def check_constraints(self, exposures):
        return {s: abs(e.net_exposure) <= self.max_sector_net and
                   e.gross_exposure <= self.max_sector_gross
                for s, e in exposures.items()}

    def scale_to_constraints(self, active_pairs, position_sizes):
        exposures = self.compute_sector_exposures(active_pairs, position_sizes)
        violations = self.check_constraints(exposures)
        scaled = dict(position_sizes)
        for sector, ok in violations.items():
            if ok: continue
            exp = exposures[sector]
            if exp.gross_exposure > self.max_sector_gross:
                scale = self.max_sector_gross / exp.gross_exposure
            else:
                scale = self.max_sector_net / abs(exp.net_exposure)
            for p in active_pairs:
                pk = f"{p['ticker_y']}/{p['ticker_x']}"
                if self.sector_map.get(p['ticker_y']) == sector or \
                   self.sector_map.get(p['ticker_x']) == sector:
                    scaled[pk] = scaled.get(pk, 0) * scale
        return scaled
