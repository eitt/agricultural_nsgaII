from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler


@dataclass
class ForecastResult:
    product: str
    model: str
    mae: float
    rmse: float
    smape: float
    mase: float
    arima_order: tuple[int, int, int] | None = None


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape = float(100 * np.mean(np.divide(np.abs(y_true-y_pred), denom, out=np.zeros_like(denom), where=denom>1e-12)))
    scale = float(np.mean(np.abs(np.diff(y_train))))
    mase = mae / scale if scale > 1e-12 else np.nan
    return dict(mae=mae, rmse=rmse, smape=smape, mase=mase)


def select_arima(log_train: np.ndarray, pmax: int = 1, qmax: int = 1) -> tuple[tuple[int,int,int], object]:
    best_aic = np.inf
    best_order = None
    best_fit = None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for d in (0, 1):
            for p in range(pmax + 1):
                for q in range(qmax + 1):
                    if p == 0 and d == 0 and q == 0:
                        continue
                    try:
                        fit = ARIMA(log_train, order=(p,d,q), trend='t' if d==1 else 'ct').fit()
                        if np.isfinite(fit.aic) and fit.aic < best_aic:
                            best_aic = fit.aic
                            best_order = (p,d,q)
                            best_fit = fit
                    except Exception:
                        continue
    if best_fit is None:
        best_order = (1,1,0)
        best_fit = ARIMA(log_train, order=best_order).fit()
    return best_order, best_fit


def _lag_matrix(y: np.ndarray, lags=(1,2,4,8), period: int = 52):
    y=np.asarray(y,float)
    m=max(lags)
    X=[]; target=[]
    for t in range(m,len(y)):
        row=[y[t-l] for l in lags]
        row += [np.sin(2*np.pi*t/period), np.cos(2*np.pi*t/period), t/period]
        X.append(row); target.append(y[t])
    return np.asarray(X,float), np.asarray(target,float)


def _fit_gp_recursive(train: np.ndarray, horizon: int, lags=(1,2,4,8), period: int=52, alpha: float=1e-6):
    X,y=_lag_matrix(train,lags=lags,period=period)
    xs=StandardScaler().fit(X)
    ys=StandardScaler().fit(y.reshape(-1,1))
    Xz=xs.transform(X); yz=ys.transform(y.reshape(-1,1)).ravel()
    kernel = ConstantKernel(1.0, (1e-2,1e2)) * Matern(length_scale=np.ones(Xz.shape[1]), length_scale_bounds=(1e-2,1e2), nu=1.5) + WhiteKernel(0.05,(1e-5,1.0))
    gp=GaussianProcessRegressor(kernel=kernel, normalize_y=False, alpha=alpha, optimizer=None, random_state=2026)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gp.fit(Xz,yz)
    hist=list(map(float,train))
    means=[]; stds=[]
    for h in range(horizon):
        t=len(hist)
        row=[hist[t-l] for l in lags] + [np.sin(2*np.pi*t/period), np.cos(2*np.pi*t/period), t/period]
        mz,sz=gp.predict(xs.transform([row]),return_std=True)
        m=float(ys.inverse_transform(np.array(mz).reshape(-1,1))[0,0])
        # local conversion for std from standardized target
        s=float(sz[0]*ys.scale_[0])
        hist.append(m); means.append(m); stds.append(s)
    return np.asarray(means), np.asarray(stds), gp


def forecast_one(series: np.ndarray, product: str, test_size: int=52) -> tuple[list[ForecastResult], dict[str,np.ndarray]]:
    y=np.asarray(series,float)
    if np.any(y<=0):
        raise ValueError(f'{product}: prices must be positive')
    train=y[:-test_size]; test=y[-test_size:]
    log_train=np.log(train)
    order,fit=select_arima(log_train)
    arima_log=np.asarray(fit.forecast(test_size),float)
    arima=np.exp(arima_log)

    # Direct GP on log prices with autoregressive lag features.
    gp_log,gp_std,_=_fit_gp_recursive(log_train,test_size)
    gp=np.exp(gp_log)

    # Hybrid: ARIMA mean plus a GP model for serial structure remaining in residuals.
    resid=np.asarray(fit.resid,float)
    # discard diffuse/start-up residuals and keep enough observations for lags
    resid = resid[max(8, len(resid)//20):]
    if len(resid) >= 30:
        gp_resid,gp_resid_std,_=_fit_gp_recursive(resid,test_size,lags=(1,2,4),period=52)
    else:
        gp_resid=np.zeros(test_size); gp_resid_std=np.zeros(test_size)
    hybrid_log=arima_log + gp_resid
    hybrid=np.exp(hybrid_log)

    preds={'ARIMA':arima,'GP':gp,'ARIMA_GP':hybrid,'actual':test,'gp_std_log':gp_std,'hybrid_resid_std_log':gp_resid_std}
    rows=[]
    for model,pred in [('ARIMA',arima),('GP',gp),('ARIMA_GP',hybrid)]:
        met=_metrics(test,pred,train)
        rows.append(ForecastResult(product,model,**met,arima_order=order if model in ('ARIMA','ARIMA_GP') else None))
    return rows,preds


def backtest_dataframe(prices: pd.DataFrame, test_size: int=52) -> tuple[pd.DataFrame, dict[str,dict[str,np.ndarray]]]:
    rows=[]; pred={}
    for c in prices.columns:
        r,p=forecast_one(prices[c].to_numpy(float),c,test_size=test_size)
        rows.extend([x.__dict__ for x in r]); pred[c]=p
    return pd.DataFrame(rows),pred

def forecast_full_series(series: np.ndarray, horizon: int, model: str, arima_order: tuple[int,int,int] | None=None):
    y=np.asarray(series,float)
    logy=np.log(y)
    if model=='GP':
        pred_log,std,_=_fit_gp_recursive(logy,horizon)
        return np.exp(pred_log), std
    if arima_order is None:
        arima_order,fit=select_arima(logy)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            d=arima_order[1]
            fit=ARIMA(logy, order=arima_order, trend='t' if d==1 else 'ct').fit()
    ar=np.asarray(fit.forecast(horizon),float)
    if model=='ARIMA':
        return np.exp(ar), np.zeros(horizon)
    if model=='ARIMA_GP':
        resid=np.asarray(fit.resid,float)
        resid=resid[max(8,len(resid)//20):]
        corr,std,_=_fit_gp_recursive(resid,horizon,lags=(1,2,4),period=52)
        return np.exp(ar+corr), std
    raise ValueError(model)


def selected_forecasts(prices: pd.DataFrame, metrics: pd.DataFrame, horizon: int):
    best=metrics.loc[metrics.groupby('product')['smape'].idxmin()].set_index('product')
    out={}; meta=[]
    for c in prices.columns:
        row=best.loc[c]; model=str(row['model'])
        order=row.get('arima_order',None)
        if isinstance(order,str):
            import ast; order=ast.literal_eval(order)
        if model=='GP': order=None
        pred,std=forecast_full_series(prices[c].to_numpy(float),horizon,model,order)
        out[c]=pred
        meta.append({'product':c,'selected_model':model,'cv_smape':float(row['smape']),'cv_rmse':float(row['rmse'])})
    return pd.DataFrame(out),pd.DataFrame(meta)
