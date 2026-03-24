"""
api/main.py

FastAPI backend for the Quant Research Terminal.
Returns structured scan payloads with diagnostics so the frontend can
surface trust and data-quality metadata alongside pricing output.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from datetime import datetime, timezone
import uuid
from src.core.scanner_engine import scan_for_valuation_gaps

app = FastAPI(title="Quant Research Terminal API", version="2.0")

class ScanRequest(BaseModel):
    ticker: str
    r: float = 0.05
    q: float = 0.0
    model: str = 'heston'
    n_sims: int = 10000
    jump_intensity: float = 0.1
    jump_mean: float = -0.05
    jump_std: float = 0.03
    heston_V0: float = 0.04
    heston_kappa: float = 2.0
    heston_theta: float = 0.04
    heston_xi: float = 0.3
    heston_rho: float = -0.7
    max_spread_pct: float = 0.25
    options_data: list = []
    spot: float = 100.0
    historical_vol: float = 0.2

@app.get("/")
async def root():
    return {"status": "online", "application": "Quant Research Terminal Engine V2"}

@app.post("/scan")
async def run_options_scan(req: ScanRequest):
    """
    1. Receives structured options data & spot parameters from the frontend.
    2. Runs the pricing engine with structured diagnostics.
    3. Returns metadata the frontend can expose for auditability.
    """
    try:
        options_df = pd.DataFrame(req.options_data)
        if options_df.empty:
            raise HTTPException(status_code=400, detail=f"No active options data provided in payload for {req.ticker}")
            
        S0 = req.spot
        fallback_vol = req.historical_vol
            
        # Run Pricing Engine
        result_df, diagnostics = scan_for_valuation_gaps(
            options_df, S0, req.r, req.q,
            model=req.model,
            n_sims=req.n_sims,
            jump_intensity=req.jump_intensity,
            jump_mean=req.jump_mean,
            jump_std=req.jump_std,
            heston_V0=req.heston_V0,
            heston_kappa=req.heston_kappa,
            heston_theta=req.heston_theta,
            heston_xi=req.heston_xi,
            heston_rho=req.heston_rho,
            sigma_fallback=fallback_vol,
            max_spread_pct=req.max_spread_pct,
            return_diagnostics=True,
        )
        diagnostics['engine'] = 'async_api'
        
        return {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": f"scan_{uuid.uuid4().hex[:12]}",
            "ticker": req.ticker,
            "spot": S0,
            "historical_vol": fallback_vol,
            "results_count": len(result_df),
            "diagnostics": diagnostics,
            "data": result_df.to_dict(orient="records")
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal scanning error: {str(e)}")
