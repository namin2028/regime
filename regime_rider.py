"""
REGIME RIDER — a terminal market-timing game
============================================
The market secretly switches between three regimes (BULL, CHOP, CRISIS)
driven by a hidden Markov chain — the same idea this repo's models try to
detect. Each week you read the tape: a price sparkline, realized volatility,
drawdown, and a (noisy, occasionally lying) news headline. Then you choose:

    [L]ong   — ride the market
    [C]ash   — sit in T-bills (small steady yield)
    [S]hort  — bet on a crash (pay borrow costs while you wait)

After a year of trading, the hidden regimes are revealed and you're graded
on whether you beat buy & hold — and whether you actually read the regimes
or just got lucky.

Usage:
  python regime_rider.py                # play, random market
  python regime_rider.py --seed 7       # replayable market
  python regime_rider.py --weeks 26     # shorter game
  python regime_rider.py --auto         # watch a trend-following bot play

Pure standard library. No dependencies, no investment advice, no refunds.
"""

import argparse
import math
import random
import statistics
import sys

# ─────────────────────────────────────────────
# ANSI helpers
# ─────────────────────────────────────────────
RESET, BOLD, DIM = '\033[0m', '\033[1m', '\033[2m'
GREEN, YELLOW, RED, CYAN, GRAY = '\033[32m', '\033[33m', '\033[31m', '\033[36m', '\033[90m'
SPARK = '▁▂▃▄▅▆▇█'


def paint(text, color, enabled=True):
    return f'{color}{text}{RESET}' if enabled else text


# ─────────────────────────────────────────────
# Hidden market
# ─────────────────────────────────────────────
REGIMES = ['BULL', 'CHOP', 'CRISIS']
REGIME_PARAMS = {                     # daily mean, daily stdev
    'BULL':   (0.00075, 0.007),
    'CHOP':   (0.00000, 0.013),
    'CRISIS': (-0.0030, 0.028),
}
TRANSITIONS = {                       # sticky chains so regimes persist
    'BULL':   [('BULL', 0.970), ('CHOP', 0.025), ('CRISIS', 0.005)],
    'CHOP':   [('BULL', 0.030), ('CHOP', 0.945), ('CRISIS', 0.025)],
    'CRISIS': [('BULL', 0.010), ('CHOP', 0.060), ('CRISIS', 0.930)],
}
REGIME_COLOR = {'BULL': GREEN, 'CHOP': YELLOW, 'CRISIS': RED}

HEADLINES = {
    'BULL': [
        'Earnings season smashes expectations across the board',
        'Fed chair smiles faintly; traders interpret as dovish',
        'Retail investors pile in: "stocks only go up," says man with 3x leverage',
        'IPO triples on first day of trading',
        'Unemployment hits multi-decade low',
    ],
    'CHOP': [
        'Markets mixed as traders await literally any signal',
        'Analysts divided: "could go either way," reports analyst',
        'Volume thins as Wall Street decamps to the Hamptons',
        'CPI comes in exactly as expected; everyone upset anyway',
        'Tech rallies, energy slumps, indices finish flat',
    ],
    'CRISIS': [
        'Margin calls cascade as funds rush for the exits',
        'Bank CEO insists balance sheet is "rock solid" (uh oh)',
        'VIX spikes; options desk orders pizza for the third night running',
        'Sovereign default rumors rattle global markets',
        'Circuit breakers triggered minutes after the open',
    ],
}
HEADLINE_NOISE = 0.25   # chance the headline comes from the wrong regime


class Market:
    """Hidden-regime geometric random walk, one bar per trading day."""

    def __init__(self, rng):
        self.rng = rng
        self.regime = rng.choices(REGIMES, weights=[0.6, 0.3, 0.1])[0]
        self.price = 100.0
        self.closes = [self.price]
        self.regime_history = []

    def _step_regime(self):
        states, probs = zip(*TRANSITIONS[self.regime])
        self.regime = self.rng.choices(states, weights=probs)[0]

    def step_week(self):
        """Advances 5 trading days; returns the week's log return."""
        start = self.price
        week_regimes = []
        for _ in range(5):
            self._step_regime()
            week_regimes.append(self.regime)
            mu, sigma = REGIME_PARAMS[self.regime]
            self.price *= math.exp(self.rng.gauss(mu, sigma))
            self.closes.append(self.price)
        # Record the regime that drove most of this week's bars
        self.regime_history.append(max(set(week_regimes), key=week_regimes.count))
        return math.log(self.price / start)

    def headline(self):
        regime = self.regime
        if self.rng.random() < HEADLINE_NOISE:
            regime = self.rng.choice([r for r in REGIMES if r != regime])
        return self.rng.choice(HEADLINES[regime])

    # --- tape-reading stats the player sees -------------------------------
    def sparkline(self, n=60, width=48):
        closes = self.closes[-n:]
        if len(closes) > width:                      # downsample to fit
            step = len(closes) / width
            closes = [closes[int(i * step)] for i in range(width)]
        lo, hi = min(closes), max(closes)
        span = (hi - lo) or 1e-9
        return ''.join(SPARK[int((c - lo) / span * (len(SPARK) - 1))] for c in closes)

    def realized_vol(self, days=21):
        closes = self.closes[-(days + 1):]
        if len(closes) < 3:
            return 0.0
        rets = [math.log(b / a) for a, b in zip(closes, closes[1:])]
        return statistics.stdev(rets) * math.sqrt(252)

    def drawdown(self, days=252):
        closes = self.closes[-days:]
        return self.price / max(closes) - 1.0

    def week_return(self):
        if len(self.closes) < 6:
            return 0.0
        return self.closes[-1] / self.closes[-6] - 1.0


# ─────────────────────────────────────────────
# Player & scoring
# ─────────────────────────────────────────────
RF_WEEKLY = 0.04 / 52          # T-bill yield while in cash
BORROW_WEEKLY = 0.01 / 52      # cost of carrying a short

ACTIONS = {'l': 'LONG', 'c': 'CASH', 's': 'SHORT'}
BEST_ACTION = {'BULL': 'LONG', 'CHOP': 'CASH', 'CRISIS': 'SHORT'}


def position_return(position, week_log_ret):
    market = math.exp(week_log_ret) - 1.0
    if position == 'LONG':
        return market
    if position == 'CASH':
        return RF_WEEKLY
    return -market - BORROW_WEEKLY     # SHORT


def sharpe(weekly_returns):
    if len(weekly_returns) < 2:
        return 0.0
    excess = [r - RF_WEEKLY for r in weekly_returns]
    sd = statistics.stdev(excess)
    return (statistics.mean(excess) / sd) * math.sqrt(52) if sd > 1e-12 else 0.0


def grade(player_total, bnh_total, read_accuracy):
    edge = player_total - bnh_total
    score = edge * 100 + (read_accuracy - 1 / 3) * 40
    for cutoff, letter, blurb in [
        (18, 'S', 'Are you... from the future?'),
        (10, 'A', 'The desk is naming a strategy after you.'),
        (4,  'B', 'Solid tape-reading. Quant team is nervous.'),
        (0,  'C', 'You matched the index. So does a sleeping dog.'),
        (-8, 'D', 'Have you considered index funds?'),
    ]:
        if score >= cutoff:
            return letter, blurb
    return 'F', 'The margin desk would like a word.'


# ─────────────────────────────────────────────
# Bots (for --auto and the rival)
# ─────────────────────────────────────────────
def trend_bot(market):
    """Simple trend/vol filter: long uptrends, short high-vol downtrends."""
    closes = market.closes
    if len(closes) < 25:
        return 'LONG'
    ma_fast = sum(closes[-10:]) / 10
    ma_slow = sum(closes[-25:]) / 25
    if market.realized_vol() > 0.30 and ma_fast < ma_slow:
        return 'SHORT'
    return 'LONG' if ma_fast >= ma_slow else 'CASH'


def rival_bot(market):
    """The in-game rival: a vol-targeting bot — long calm markets, cash otherwise."""
    return 'LONG' if market.realized_vol() < 0.22 else 'CASH'


# ─────────────────────────────────────────────
# Game loop
# ─────────────────────────────────────────────
def vol_gauge(vol, color_on):
    filled = min(10, int(vol / 0.05))
    bar = '█' * filled + '░' * (10 - filled)
    color = GREEN if vol < 0.15 else YELLOW if vol < 0.30 else RED
    return paint(bar, color, color_on) + f' {vol:.0%}'


def read_action(last_action, color_on):
    prompt = (f'  {paint("[L]", GREEN, color_on)}ong  '
              f'{paint("[C]", YELLOW, color_on)}ash  '
              f'{paint("[S]", RED, color_on)}hort  '
              f'[Enter]={last_action.lower()}  [Q]uit > ')
    try:
        raw = input(prompt).strip().lower()
    except EOFError:
        return last_action, True       # stdin exhausted: coast on last action
    if raw == 'q':
        return last_action, 'QUIT'
    return ACTIONS.get(raw[:1], last_action) if raw else last_action, False


def play(weeks, seed, auto, color_on):
    rng = random.Random(seed)
    market = Market(rng)
    for _ in range(12):                # warm-up so week 1 has a chart
        market.step_week()
    market.regime_history.clear()

    equity, bnh, rival = 1.0, 1.0, 1.0
    player_weekly, bnh_weekly = [], []
    positions, action = [], 'LONG'
    quit_early = False

    title = paint('REGIME RIDER', BOLD + CYAN, color_on)
    print(f'\n{title} — beat the market for {weeks} weeks. '
          f'Somewhere in the noise, regimes are shifting.\n')

    for week in range(1, weeks + 1):
        wk_ret = market.week_return()
        ret_color = GREEN if wk_ret >= 0 else RED
        print(f'{paint(f"Week {week:>2}/{weeks}", BOLD, color_on)}'
              f'   price {market.price:8.2f}'
              f'   last wk {paint(f"{wk_ret:+7.2%}", ret_color, color_on)}'
              f'   dd {market.drawdown():+.1%}')
        print(f'  {paint(market.sparkline(), CYAN, color_on)}')
        print(f'  vol {vol_gauge(market.realized_vol(), color_on)}'
              f'    {paint("NEWS:", DIM, color_on)} '
              f'{paint(market.headline(), DIM, color_on)}')
        print(f'  you {equity:6.3f}   buy&hold {bnh:6.3f}   rival-bot {rival:6.3f}')

        if auto:
            action = trend_bot(market)
            print(f'  bot plays: {action}')
        else:
            action, stop = read_action(action, color_on)
            if stop == 'QUIT':
                quit_early = True
                print('\nYou walk off the trading floor mid-year. Bold.')
                break

        # Rival decides BEFORE the week unfolds — no look-ahead, same as you.
        rival_action = rival_bot(market)
        week_log = market.step_week()
        positions.append(action)
        p_ret = position_return(action, week_log)
        b_ret = math.exp(week_log) - 1.0
        equity *= 1 + p_ret
        bnh *= 1 + b_ret
        rival *= 1 + position_return(rival_action, week_log)
        player_weekly.append(p_ret)
        bnh_weekly.append(b_ret)
        print()

    show_results(market, positions, equity, bnh, rival,
                 player_weekly, bnh_weekly, quit_early, color_on)


def show_results(market, positions, equity, bnh, rival,
                 player_weekly, bnh_weekly, quit_early, color_on):
    if not positions:
        return
    regimes = market.regime_history[:len(positions)]

    print(paint('\n════════ THE BIG REVEAL ════════', BOLD, color_on))
    regime_glyph = {'BULL': 'B', 'CHOP': '~', 'CRISIS': 'X'}
    print('Hidden regimes:  ' + ''.join(
        paint('█' if color_on else regime_glyph[r], REGIME_COLOR[r], color_on)
        for r in regimes))
    print('Your positions:  ' + ''.join(
        paint(p[0], GREEN if p == 'LONG' else YELLOW if p == 'CASH' else RED,
              color_on) for p in positions))
    legend = ('█ green=BULL yellow=CHOP red=CRISIS' if color_on
              else 'B=BULL ~=CHOP X=CRISIS')
    print(paint(f'                 ({legend} · L=long C=cash S=short)', DIM, color_on))

    hits = sum(1 for p, r in zip(positions, regimes) if p == BEST_ACTION[r])
    accuracy = hits / len(positions)
    crisis_weeks = [i for i, r in enumerate(regimes) if r == 'CRISIS']
    dodged = sum(1 for i in crisis_weeks if positions[i] != 'LONG')

    print(f'\nRegime read accuracy : {accuracy:.0%} '
          f'({hits}/{len(positions)} weeks in the ideal position)')
    if crisis_weeks:
        print(f'Crises dodged        : {dodged}/{len(crisis_weeks)} '
              f'crisis weeks spent out of the market')
    print(f'\n{"":20}{"You":>10}{"Buy&Hold":>12}{"Rival bot":>12}')
    print(f'{"Total return":20}{equity - 1:>10.1%}{bnh - 1:>12.1%}{rival - 1:>12.1%}')
    print(f'{"Sharpe ratio":20}{sharpe(player_weekly):>10.2f}'
          f'{sharpe(bnh_weekly):>12.2f}{"—":>12}')

    letter, blurb = grade(equity - 1, bnh - 1, accuracy)
    if quit_early:
        blurb += ' (Quitting early was noted in your performance review.)'
    color = GREEN if letter in 'SAB' else YELLOW if letter == 'C' else RED
    print(f'\nFinal grade: {paint(letter, BOLD + color, color_on)} — {blurb}\n')


def main():
    parser = argparse.ArgumentParser(description='REGIME RIDER — a market-timing game.')
    parser.add_argument('--weeks', type=int, default=52, help='game length (default 52)')
    parser.add_argument('--seed', type=int, default=None, help='replayable market seed')
    parser.add_argument('--auto', action='store_true', help='let a trend-following bot play')
    parser.add_argument('--no-color', action='store_true', help='disable ANSI colors')
    args = parser.parse_args()

    color_on = not args.no_color
    try:
        play(max(4, args.weeks), args.seed, args.auto, color_on)
    except KeyboardInterrupt:
        print('\n\nPosition liquidated. See you tomorrow at the open.')


if __name__ == '__main__':
    main()
