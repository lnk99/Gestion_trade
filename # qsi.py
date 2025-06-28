# qsi.py - Analyse technique unifiée pour les actions avec MACD et RSI
# Ce script télécharge les données boursières, calcule les indicateurs techniques et affiche
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ta
import time
import csv
from matplotlib import dates as mdates

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calcule le MACD et sa ligne de signal"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def get_trading_signal(prices, volumes, variation_seuil=-20, volume_seuil=100000):
    """Détermine les signaux de trading avec validation des données"""
    # Correction : assure que prices et volumes sont bien des Series 1D
    if isinstance(prices, pd.DataFrame):
        prices = prices.squeeze()
    if isinstance(volumes, pd.DataFrame):
        volumes = volumes.squeeze()
    
    if len(prices) < 50:
        return "Données insuffisantes", None, None, None, None, None, None
    
    # Calcul des indicateurs
    macd, signal_line = calculate_macd(prices)
    rsi = ta.momentum.RSIIndicator(close=prices, window=17).rsi()
    
    # Validation des derniers points
    if len(macd) < 2 or len(rsi) < 1:
        return "Données récentes manquantes", None, None, None, None, None, None
    
    last_close = prices.iloc[-1]
    ema20 = prices.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = prices.ewm(span=50, adjust=False).mean().iloc[-1]
    
    # Calcul de la variation sur 30 jours
    if len(prices) >= 30:
        variation_30j = (last_close - prices.iloc[-30]) / prices.iloc[-30] * 100
    else:
        variation_30j = None
    
    # Calcul du volume moyen sur 20 jours
    if len(volumes) >= 20:
        volume_mean = volumes.rolling(window=20).mean().iloc[-1]
    else:
        volume_mean = None
    
    # Filtrage basé sur la variation et le volume
    if variation_30j is not None and volume_mean is not None:
        if variation_30j > variation_seuil and volume_mean > volume_seuil:
            # Détection des signaux
            if (macd.iloc[-2] < signal_line.iloc[-2] and  rsi.iloc[-1] < 70 and last_close > (ema50) and 
                macd.iloc[-1] > signal_line.iloc[-1] ):
                signal = "ACHAT"
            elif (macd.iloc[-2] > signal_line.iloc[-2] and rsi.iloc[-1] > 30 and last_close <(ema50) and 
                 macd.iloc[-1] < signal_line.iloc[-1] ):
                signal = "VENTE"
            else:
                signal = "NEUTRE"
        else:
            signal = "NEUTRE"
    else:
        signal = "NEUTRE"
    
    return (signal, last_close, last_close > ema20, rsi.iloc[-1], 
            last_close > ema50, variation_30j, volume_mean)

def download_stock_data(symbols, period):
    """Télécharge les données avec gestion des erreurs"""
    valid_data = {}
    
    for symbol in symbols:
        try:
            data = yf.download(symbol, period=period, interval="1d", progress=False, timeout=10)
            
            if data.empty:
                print(f"⚠️ Aucune donnée pour {symbol}")
                continue
                
            if 'Close' not in data.columns or 'Volume' not in data.columns:
                print(f"⚠️ Colonne 'Close' ou 'Volume' manquante pour {symbol}")
                continue
                
            valid_data[symbol] = {
                'Close': data['Close'],
                'Volume': data['Volume']
            }
            
        except Exception as e:
            print(f"🚨 Erreur sur {symbol}: {str(e)}")
    
    return valid_data

def backtest_signals(prices, volumes, montant=50):
    """
    Effectue un backtest sur la série de prix.
    Un 'trade' correspond ici à un cycle complet ACHAT puis VENTE (entrée puis sortie).
    Le gain est calculé pour chaque cycle achat-vente.
    """
    # Correction : refuse les scalaires et séries trop courtes
    if not isinstance(prices, (pd.Series, pd.DataFrame)) or len(prices) < 2:
        return {
            "trades": 0,
            "gagnants": 0,
            "taux_reussite": 0,
            "gain_total": 0.0
        }
    if isinstance(prices, pd.DataFrame):
        prices = prices.squeeze()
    if isinstance(volumes, pd.DataFrame):
        volumes = volumes.squeeze()
    positions = []
    for i in range(1, len(prices)):
        signal, *_ = get_trading_signal(prices[:i], volumes[:i])
        if signal == "ACHAT":
            #print(f"i={i} | signal={signal} | prix={prices.iloc[i]}")
            positions.append({"entry": prices.iloc[i], "entry_idx": i, "type": "buy"})
        elif signal == "VENTE" and positions and "exit" not in positions[-1]:
            #print(f"i={i} | signal={signal} | prix={prices.iloc[i]}")
            positions[-1]["exit"] = prices.iloc[i]
            positions[-1]["exit_idx"] = i

    nb_trades = 0
    nb_gagnants = 0
    gain_total = 0.0

    for pos in positions:
        if "exit" in pos:
            nb_trades += 1
            entry = pos["entry"]
            exit = pos["exit"]
            rendement = (exit - entry) / entry
            gain = montant * rendement
            gain_total += gain
            if gain > 0:
                nb_gagnants += 1

    return {
        "trades": nb_trades,
        "gagnants": nb_gagnants,
        "taux_reussite": (nb_gagnants / nb_trades * 100) if nb_trades else 0,
        "gain_total": gain_total
    }

def plot_unified_chart(symbol, prices, volumes, ax):
    """Trace un graphique unifié avec prix, MACD et RSI intégré"""
    # Correction : assure que prices est une série 1D
    if isinstance(prices, pd.DataFrame):
        prices = prices.squeeze()
    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices, name=symbol)
    
    ema20 = prices.ewm(span=20, adjust=False).mean()
    ema50 = prices.ewm(span=50, adjust=False).mean()
    sma50 = prices.rolling(window=50).mean() if len(prices) >= 50 else pd.Series()
    macd, signal_line = calculate_macd(prices)
    
    try:
        rsi = ta.momentum.RSIIndicator(close=prices, window=14).rsi()
    except Exception as e:
        print(f"⚠️ Erreur RSI pour {symbol}: {str(e)}")
        rsi = pd.Series(np.zeros(len(prices), index=prices.index))
    
    color = 'tab:blue'
    ax.plot(prices.index, prices, label='Prix', color=color, linewidth=1.8)
    ax.plot(ema20.index, ema20, label='EMA20', linestyle='--', color='orange', linewidth=1.4)
    ax.plot(ema50.index, ema50, label='EMA50', linestyle='-.', color='purple', linewidth=1.4)
    
    if not sma50.empty:
        ax.plot(sma50.index, sma50, label='SMA50', linestyle=':', color='green', linewidth=1.4)
    
    ax.set_ylabel('Prix', color=color, fontsize=10)
    ax.tick_params(axis='y', labelcolor=color)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    ax2 = ax.twinx()
    
    color = 'tab:purple'
    ax2.plot(macd.index, macd, label='MACD', color=color, linewidth=1.2)
    ax2.plot(signal_line.index, signal_line, label='Signal', color='tab:orange', linewidth=1.2)
    ax2.fill_between(
        macd.index, 0, macd - signal_line, 
        where=(macd - signal_line) >= 0, 
        facecolor='green', alpha=0.3, interpolate=True
    )
    ax2.fill_between(
        macd.index, 0, macd - signal_line, 
        where=(macd - signal_line) < 0, 
        facecolor='red', alpha=0.3, interpolate=True
    )
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax2.set_ylabel('MACD', color=color, fontsize=10)
    ax2.tick_params(axis='y', labelcolor=color)
    
    for i in range(1, len(prices)):
        start = prices.index[i-1]
        end = prices.index[i]
        rsi_val = rsi.iloc[i-1]
        
        if rsi_val > 70:
            color = 'lightcoral'
        elif rsi_val < 30:
            color = 'lightgreen'
        else:
            color = 'lightgray'
            
        ax.axvspan(start, end, facecolor=color, alpha=0.1, zorder=-1)
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)
    
    signal, last_price, trend, last_rsi, above_ema50, variation_30j, volume_mean = get_trading_signal(prices, volumes)
    
    if len(prices) > 1:
        progression = ((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]) * 100
    else:
        progression = 0.0

    if last_price is not None:
        trend_symbol = "Haussière" if trend else "Baissière"
        rsi_status = "SURACH" if last_rsi > 70 else "SURVENTE" if last_rsi < 30 else "NEUTRE"
        signal_color = 'green' if signal == "ACHAT" else 'red' if signal == "VENTE" else 'black'

        # Correction robustesse affichage
        try:
            if isinstance(variation_30j, pd.Series):
                var30 = float(variation_30j.iloc[0])
            else:
                var30 = float(variation_30j) if variation_30j is not None else float('nan')
        except Exception:
            var30 = float('nan')
        try:
            if isinstance(volume_mean, pd.Series):
                volmean = float(volume_mean.iloc[0])
            else:
                volmean = float(volume_mean) if volume_mean is not None else float('nan')
        except Exception:
            volmean = float('nan')

        title = (
            f"{symbol} | Prix: {last_price:.2f} | Signal: {signal} | "
            f"Tendance: {trend_symbol} | RSI: {last_rsi:.1f} ({rsi_status}) | "
            f"Prog: {progression:+.2f}% | Var 30j: {var30:.2f}% | Vol: {volmean:.0f}"
        )
        ax.set_title(title, fontsize=12, fontweight='bold', color=signal_color)
    
    return ax2

# ======================================================================
# CONFIGURATION PRINCIPALE
# ======================================================================
symbols = ["HSBA.L", "ALDX", "INGA.AS", "MP", "OKTA"]
period = "12mo"

print("⏳ Téléchargement des données...")
data = download_stock_data(symbols, period)

if not data:
    print("❌ Aucune donnée valide disponible. Vérifiez les symboles ou la connexion internet.")
    exit()

num_plots = len(data)
fig, axes = plt.subplots(num_plots, 1, figsize=(14, 5 * num_plots), sharex=False)

if num_plots == 1:
    axes = [axes]
elif num_plots == 0:
    print("❌ Aucun symbole valide à afficher")
    exit()

achat_signals = []
vente_signals = []
all_signals = []

for i, (symbol, stock_data) in enumerate(data.items()):
    prices = stock_data['Close']
    volumes = stock_data['Volume']
    print(f"📊 Traitement de {symbol}...")
    
    ax2 = plot_unified_chart(symbol, prices, volumes, axes[i])
    
    signal, last_price, above_ema20, rsi, above_ema50, variation_30j, volume_mean = get_trading_signal(prices, volumes)
    
    if signal == "ACHAT":
        achat_signals.append({
            'Symbole': symbol,
            'Prix': last_price,
            'RSI': rsi,
            'EMA20+': above_ema20,
            'EMA50+': above_ema50,
            'Variation_30j': variation_30j,
            'Volume_moyen': volume_mean
        })
        print(f"✅ {symbol} - Signal: {signal} | Prix: {last_price:.2f} | RSI: {rsi:.1f} | EMA20+: {above_ema20} | EMA50+: {above_ema50}")
    elif signal == "VENTE":
        vente_signals.append({
            'Symbole': symbol,
            'Prix': last_price,
            'RSI': rsi,
            'EMA20+': above_ema20,
            'EMA50+': above_ema50,
            'Variation_30j': variation_30j,
            'Volume_moyen': volume_mean
        })
        print(f"⚠️ {symbol} - Signal: {signal} | Prix: {last_price:.2f} | RSI: {rsi:.1f} | EMA20+: {above_ema20} | EMA50+: {above_ema50}")

plt.tight_layout()
plt.subplots_adjust(top=0.95, hspace=0.4)
plt.show()

# Exportation des signaux d'achat dans un fichier CSV
if achat_signals:
    with open('signaux_achat.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Symbole', 'Prix', 'RSI', 'EMA20+', 'EMA50+', 'Variation_30j', 'Volume_moyen'])
        for signal in achat_signals:
            writer.writerow([
                signal['Symbole'], 
                round(signal['Prix'], 2), 
                round(signal['RSI'], 1), 
                signal['EMA20+'], 
                signal['EMA50+'], 
                round(signal['Variation_30j'], 2), 
                round(signal['Volume_moyen'], 0)
            ])
    print("📝 Signaux d'achat exportés dans 'signaux_achat.csv'")

# ======================================================================
# SIGNEAUX POUR ACTIONS POPULAIRES
# ======================================================================
popular_symbols = list(dict.fromkeys([
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK-B", "JPM", "V", "BABA", "DIS", "NFLX", "PYPL", "INTC", "AMD", "CSCO", "WMT", "VZ", "KO", "SGMT",
    "PEP", "MRK", "T", "NKE", "XOM", "CVX", "ABT", "CRM", "IBM", "ORCL", "MCD", "TMDX", "BA", "CAT", "GS", "RTX", "MMM", "HON", "LMT", "SBUX", "ADBE", "SMSN.L",
    "EEM", "INTU", "NOW", "ZM", "SHOP", "SNAP", "PFE", "TGT", "CVS", "WFC", "RHM.DE", "SAP.DE", "BAS.DE", "ALV.DE", "BMW.DE", "VOW3.DE", "SMTC", "ZS", "ZTS",
    "DTE.DE", "DBK.DE", "LHA.DE", "FME.DE", "BAYN.DE", "LIN.DE", "ENR.DE", "VNA.DE", "1COV.DE", "FRE.DE", "HEN3.DE", "HEI.DE", "RWE.DE", "VOW.DE", "GLW", "TMO",
    "DHR", "ABB", "BAX", "MDT", "GE", "NOC", "GD", "HII", "TXT", "LHX", "TDY", "CARR", "OTIS", "JCI", "INOD", "BIDU", "JD", "PDD", "TCEHY", "NTES", "BILI",
    "XPEV", "LI", "NIO", "BYDDF", "GME", "AMC", "BB", "NOK", "RBLX", "PLTR", "FSLY", "CRWD", "OKTA", "Z", "DOCU", "PINS", "SPOT", "LYFT", "UBER", "SNOW", "TTWO",
    "VRSN", "WDAY", "2318.HK", "2382.HK", "2388.HK", "2628.HK", "3328.HK", "3988.HK", "9988.HK", "2319.HK", "0700.HK", "3690.HK", "ADSK", "02020.HK",
    "9618.HK", "1810.HK", "1211.HK", "1299.HK", "2313.HK", "2386.HK", "2623.HK", "2385.HK", "0005.HK", "0011.HK", "0027.HK", "0038.HK", "0066.HK", "0083.HK",
    "0101.HK", "0117.HK", "0120.HK", "LSEG.L", "VOD.L", "BP.L", "HSBA.L", "GSK.L", "ULVR.L", "AZN.L", "RIO.L", "BATS.L", "ADYEN.AS", "TM", "MU", "GILT",
    "ASML.AS", "PHIA.AS", "INGA.AS", "MC.PA", "OR.PA", "AIR.PA", "BNP.PA", "SAN.PA", "ENGI.PA", "CAP.PA", "LVMH.PA", "WELL", "O", "VICI", "ETOR", "ABR",
    "PLD", "PSA", "AMT", "CCI", "DLR", "EXR", "EQR", "ESS", "AVB", "MAA", "UDR", "SBRA", "UNH", "HD", "MA", "PG", "LLY", "COST", "AVGO", "ABBV", "QCOM",
    "DDOG", "CRL", "EXAS", "ILMN", "INCY", "MELI", "MRNA", "NTLA", "REGN", "ROKU", "QSI", "SYM", "IONQ", "QBTS", "RGTI", "SMCI", "TSM", "ALDX", "CSX", "LRCX",
    "BIIB", "CDNS", "CTSH", "EA", "FTNT", "GILD", "IDXX", "MP", "MTCH", "MRVL", "PAYX", "PTON", "AAL", "UAL", "DAL", "LUV", "JBLU", "ALK", "FLEX", "CACI",
    "CRIS", "CYTK", "EXEL", "FATE", "INSM", "KPTI", "NBIX", "NTRA", "PGEN", "RGEN", "SAGE", "SNY", "TGTX", "VYGR", "ARCT", "AXSM", "BMRN", "KTOS"
]))

print("\n🔍 Analyse des signaux pour actions populaires...")
signals = []

CHUNK_SIZE = 16
for i in range(0, len(popular_symbols), CHUNK_SIZE):
    chunk = popular_symbols[i:i+CHUNK_SIZE]
    print(f"\n🔎 Lot {i//CHUNK_SIZE + 1}: {', '.join(chunk)}")
    
    try:
        chunk_data = download_stock_data(chunk, period)
        
        for symbol, stock_data in chunk_data.items():
            prices = stock_data['Close']
            volumes = stock_data['Volume']
            if len(prices) < 50:
                continue

            (signal, last_price, trend, last_rsi, 
             above_ema50, variation_30j, volume_mean) = get_trading_signal(prices, volumes)

            if signal != "NEUTRE":
                try:
                    info = yf.Ticker(symbol).info
                    domaine = info.get("sector", "Inconnu")
                except Exception:
                    domaine = "Inconnu"
                signals.append({
                    'Symbole': symbol,
                    'Signal': signal,
                    'Prix': last_price,
                    'Tendance': "Hausse" if trend else "Baisse",
                    'RSI': last_rsi,
                    'Domaine': domaine,
                    'EMA50+': above_ema50,
                    'Variation_30j': variation_30j,
                    'Volume_moyen': volume_mean
                })
    
    except Exception as e:
        print(f"⚠️ Erreur: {str(e)}")
    
    time.sleep(1)

if signals:
    print("\n" + "=" * 105)
    print("RÉSULTATS DES SIGNEAUX")
    print("=" * 105)
    print(f"{'Symbole':<8} {'Signal':<8} {'Prix':<10} {'Tendance':<10} {'RSI':<6} {'Domaine':<24} Analyse")
    print("-" * 105)

    signaux_tries = {"ACHAT": {"Hausse": [], "Baisse": []}, "VENTE": {"Hausse": [], "Baisse": []}}
    for s in signals:
        if s['Signal'] in signaux_tries and s['Tendance'] in signaux_tries[s['Signal']]:
            signaux_tries[s['Signal']][s['Tendance']].append(s)

    for signal_type in ["ACHAT", "VENTE"]:
        for tendance in ["Hausse", "Baisse"]:
            if signaux_tries[signal_type][tendance]:
                signaux_tries[signal_type][tendance].sort(key=lambda x: x['Prix'])
                print(f"\n--- Signal {signal_type} | Tendance {tendance} ---")
                for s in signaux_tries[signal_type][tendance]:
                    analysis = ""
                    if s['Signal'] == "ACHAT":
                        analysis += "RSI bas" if s['RSI'] < 40 else ""
                        analysis += " + Tendance haussière" if s['Tendance'] == "Hausse" else ""
                    else:
                        analysis += "RSI élevé" if s['RSI'] > 60 else ""
                        analysis += " + Tendance baissière" if s['Tendance'] == "Baisse" else ""
                    print(f"{s['Symbole']:<8} {s['Signal']:<8} {s['Prix']:<10.2f} {s['Tendance']:<10} {s['RSI']:<6.1f} {s['Domaine']:<24} {analysis}")

    print("=" * 105)
else:
    print("\nℹ️ Aucun signal fort détecté parmi les actions populaires")

if signals:
    print("\n📈 Résultats du backtest sur les signaux détectés :")
    total_trades = 0
    total_gagnants = 0
    total_gain = 0.0
    number = 0
    cout_par_trade = 1.0

    backtest_results = []

    for s in signals:
        try:
            stock_data = download_stock_data([s['Symbole']], period)[s['Symbole']]
            prices = stock_data['Close']
            if not isinstance(prices, (pd.Series, pd.DataFrame)) or len(prices) < 2:
                print(f"{s['Symbole']:<8} : Données insuffisantes pour le backtest")
                continue
            resultats = backtest_signals(prices, volumes, montant=50)
            backtest_results.append({
                "Symbole": s['Symbole'],
                "trades": resultats['trades'],
                "gagnants": resultats['gagnants'],
                "taux_reussite": resultats['taux_reussite'],
                "gain_total": resultats['gain_total']
            })
            total_trades += resultats['trades']
            total_gagnants += resultats['gagnants']
            total_gain += resultats['gain_total']
            number += 1
        except Exception as e:
            print(f"{s['Symbole']:<8} : Erreur backtest ({e})")

    backtest_results.sort(key=lambda x: x['taux_reussite'], reverse=True)

    for res in backtest_results:
        print(
            f"{res['Symbole']:<8} | Trades: {res['trades']:<2} | "
            f"Gagnants: {res['gagnants']:<2} | "
            f"Taux réussite: {res['taux_reussite']:.0f}% | "
            f"Gain total: {res['gain_total']:.2f} $"
        )

    cout_total_trades = total_trades * cout_par_trade
    total_investi_reel = len(backtest_results) * 50
    gain_total_reel = total_gain - cout_total_trades

    if total_trades > 0:
        taux_global = total_gagnants / total_trades * 100
        print("\n" + "="*105)
        print(f"🌍 Résultat global :")
        print(f"  - Taux de réussite = {taux_global:.1f}%")
        print(f"  - Nombre de trades = {total_trades}")
        print(f"  - Total investi réel = {total_investi_reel:.2f} $ (50 $ par action analysée)")
        print(f"  - Coût total des trades = {cout_total_trades:.2f} $ (à {cout_par_trade:.2f} $ par trade)")
        print(f"  - Gain total brut = {total_gain:.2f} $")
        print(f"  - Gain total net (après frais) = {gain_total_reel:.2f} $")
        print("="*105)
    else:
        print("\nAucun trade détecté pour le calcul global.")   

    filtres = [res for res in backtest_results if res['taux_reussite'] >= 60 and res['gain_total'] > 0]
    nb_actions_filtrees = len(filtres)
    total_trades_filtre = sum(res['trades'] for res in filtres)
    total_gagnants_filtre = sum(res['gagnants'] for res in filtres)
    total_gain_filtre = sum(res['gain_total'] for res in filtres)
    cout_total_trades_filtre = total_trades_filtre * cout_par_trade
    total_investi_filtre = nb_actions_filtrees * 50
    gain_total_reel_filtre = total_gain_filtre - cout_total_trades_filtre

    print("\n" + "="*105)
    print("🔎 Évaluation si investissement SEULEMENT sur les actions à taux de réussite >= 60% ET gain total positif :")
    print(f"  - Nombre d'actions sélectionnées = {nb_actions_filtrees}")
    print(f"  - Nombre de trades = {total_trades_filtre}")
    print(f"  - Taux de réussite global = {(total_gagnants_filtre / total_trades_filtre * 100) if total_trades_filtre else 0:.1f}%")
    print(f"  - Total investi réel = {total_investi_filtre:.2f} $ (50 $ par action sélectionnée)")
    print(f"  - Coût total des trades = {cout_total_trades_filtre:.2f} $ (à {cout_par_trade:.2f} $ par trade)")
    print(f"  - Gain total brut = {total_gain_filtre:.2f} $")
    print(f"  - Gain total net (après frais) = {gain_total_reel_filtre:.2f} $")
    print("="*105)

    fiables_ou_non_eval = set()
    for res in backtest_results:
        if res['taux_reussite'] >= 60 or res['trades'] == 0:
            fiables_ou_non_eval.add(res['Symbole'])

    print("\n" + "=" * 105)
    print("SIGNES UNIQUEMENT POUR ACTIONS FIABLES (>=60% taux de réussite) OU NON ÉVALUÉES")
    print("=" * 105)
    print(f"{'Symbole':<8} {'Signal':<8} {'Prix':<10} {'Tendance':<10} {'RSI':<6} {'Domaine':<24} Analyse")
    print("-" * 105)

    for signal_type in ["ACHAT", "VENTE"]:
        for tendance in ["Hausse", "Baisse"]:
            filtered = [
                s for s in signaux_tries[signal_type][tendance]
                if s['Symbole'] in fiables_ou_non_eval
            ]
            if filtered:
                print(f"\n--- Signal {signal_type} | Tendance {tendance} ---")
                for s in filtered:
                    analysis = ""
                    if s['Signal'] == "ACHAT":
                        analysis += "RSI bas" if s['RSI'] < 40 else ""
                        analysis += " + Tendance haussière" if s['Tendance'] == "Hausse" else ""
                    else:
                        analysis += "RSI élevé" if s['RSI'] > 60 else ""
                        analysis += " + Tendance baissière" if s['Tendance'] == "Baisse" else ""
                    print(f"{s['Symbole']:<8} {s['Signal']:<8} {s['Prix']:<10.2f} {s['Tendance']:<10} {s['RSI']:<6.1f} {s['Domaine']:<24} {analysis}")

    print("=" * 105)
