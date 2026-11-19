// gta6-trophy-rarity-forecast-01 entry: "dilution-power" forecaster (Node 22, standard library only).
//
// The rate a trophy shows on Exophase is earners / players-tracked, and in a launch week both
// move: tracked explodes as new players link their accounts (638 tracked on launch night, 3,868
// at the first refresh a few hours later, ~13,400 by that afternoon), while earners grow only as
// players actually reach the trophy. So a rate can FALL fast even though nobody un-earns anything:
// dilution dominates. The house persistence baseline ignores this and scores ~1.51 on the practice
// (its narrow band is blown open by the first refresh); the house curve baseline fits a straight
// line to logit(rate) against ln(1 + hours) over the last 24 h, which never saturates and uses no
// cross-trophy information.
//
// This program models the two moving parts separately, from the case's own history:
//
//   1. Refresh states. The house reads the page every 10-30 minutes but Exophase refreshes its
//      counts on its own schedule, so long runs of reads are the same page. Consecutive reads with
//      identical (players_tracked, rates) collapse to one state; a state is one observation.
//   2. Tracked growth N(t). log N is fit against ln(1 + hours since release), linear and (when it
//      clearly fits better) concave-quadratic, over the states; the fit is read at the target. With
//      fewer than two distinct states the launch prior applies: log N grows with slope 1.0 in
//      ln(1 + hours), the slope of the documented first refresh (638 -> 3,868 players in ~5 h).
//   3. Earners E_j(t). Per trophy, log E_j is fit against log N (a power law: earners grow as a
//      power of the tracked base), anchored at the last observed state: E_j(target) =
//      E_j(last) * (N_hat / N_last)^beta_j. Slopes are shrunk toward their tier median (bronze /
//      silver / gold / platinum) with a pseudo-count of 3, and clamped to [0, 1.2] (earners never
//      shrink). With no movement in the history, tier priors apply (bronze 0.08, silver/gold 0.04,
//      platinum 0.0): most trophies' earners barely move in the first hours, a few early ones do.
//   4. Median rate r_hat = 100 * E_hat / N_hat. A trophy at 100% stays there (every tracked player
//      has it, through a 6x dilution): it is forecast at 99.99 with a hairline spread.
//   5. Spread. Per-trophy logit sigma combines the earners fit's residual, the tracked fit's
//      prediction-interval width at the target, and the pooled refresh-to-refresh logit jump size
//      (1.0 logit units when no refresh has been seen yet: a refresh the history has not seen is
//      exactly what the practice punishes). Sigma is clamped to [0.35, 2.5]; quantiles are
//      logit(r_hat) + z_tau * sigma at the standard normal levels, rounded to 4 decimals.
//
// Everything is a pure function of the case file: sorted inputs, no clock, no randomness, no
// network, so two runs write the same bytes. One case file in, one forecast file out.
//
// Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input
// and /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];

/** Prior earners-vs-tracked power-law slope per tier, used when a trophy's history shows no movement. */
const BETA_PRIOR: Record<string, number> = { bronze: 0.08, silver: 0.04, gold: 0.04, platinum: 0.0 };
/** Prior slope of log tracked against ln(1 + hours since release), from the documented first refresh. */
const N_SLOPE_PRIOR = 1.0;
/** Pseudo-count for shrinking a trophy's fitted slope toward its tier median. */
const SHRINK_K = 3;

const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
/** logit of a percent, with the scorer's own clamp. */
const logit = (pct: number): number => {
  const p = clamp(pct, 0.01, 99.99) / 100;
  return Math.log(p / (1 - p));
};
const invLogitPct = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
const median = (xs: number[]): number => {
  if (xs.length === 0) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const mean = (xs: number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length;
const std = (xs: number[]): number => {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) * (b - m), 0) / (xs.length - 1));
};

/** Least squares y = a + b*x; returns [a, b, residualStd, sxx, xMean, n]. */
function linFit(xs: number[], ys: number[]): [number, number, number, number, number, number] {
  const n = xs.length;
  const xm = mean(xs);
  const ym = mean(ys);
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - xm) * (xs[i] - xm);
    sxy += (xs[i] - xm) * (ys[i] - ym);
  }
  const b = sxx > 1e-12 ? sxy / sxx : 0;
  const a = ym - b * xm;
  const resid = ys.map((y, i) => y - (a + b * xs[i]));
  return [a, b, std(resid), sxx, xm, n];
}

/** Least squares y = a + b*x + c*x^2 by Gaussian elimination on the 3x3 normal equations. */
function quadFit(xs: number[], ys: number[]): [number, number, number, number] | null {
  const n = xs.length;
  if (n < 4) return null;
  const s = (k: number): number => xs.reduce((acc, x) => acc + Math.pow(x, k), 0);
  const t = (k: number): number => xs.reduce((acc, x, i) => acc + Math.pow(x, k) * ys[i], 0);
  // Matrix [[n, s1, s2],[s1, s2, s3],[s2, s3, s4]] * [a,b,c] = [t0, t1, t2]
  const m: number[][] = [
    [n, s(1), s(2), t(0)],
    [s(1), s(2), s(3), t(1)],
    [s(2), s(3), s(4), t(2)],
  ];
  for (let col = 0; col < 3; col++) {
    let piv = col;
    for (let r = col + 1; r < 3; r++) if (Math.abs(m[r][col]) > Math.abs(m[piv][col])) piv = r;
    if (Math.abs(m[piv][col]) < 1e-12) return null;
    [m[col], m[piv]] = [m[piv], m[col]];
    for (let r = 0; r < 3; r++) {
      if (r === col) continue;
      const f = m[r][col] / m[col][col];
      for (let k = col; k < 4; k++) m[r][k] -= f * m[col][k];
    }
  }
  const a = m[0][3] / m[0][0];
  const b = m[1][3] / m[1][1];
  const c = m[2][3] / m[2][2];
  const resid = ys.map((y, i) => y - (a + b * xs[i] + c * xs[i] * xs[i]));
  return [a, b, c, std(resid)];
}

interface Trophy {
  id: number;
  tier?: string;
}
interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}
interface Case {
  id: string;
  at: string;
  target: string;
  horizon_minutes: number;
  levels: number[];
  game?: { release_date?: string };
  trophies: Trophy[];
  history: Read[];
}

interface State {
  atMs: number;
  n: number;
  rates: Array<number | null>;
}

function forecast(c: Case): Array<{ id: number; q: number[] }> {
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${c.id} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (!c.history || c.history.length === 0) throw new Error(`${c.id} has no history`);
  const targetMs = Date.parse(c.target);
  const atMs = Date.parse(c.at);
  if (Number.isNaN(targetMs) || Number.isNaN(atMs)) throw new Error(`${c.id} has an unparsable at/target`);

  // Release midnight UTC, from the case's game data; fall back to the first read's day.
  let t0Ms = Date.parse(`${c.game?.release_date ?? ''}T00:00:00Z`);
  if (Number.isNaN(t0Ms)) t0Ms = Date.parse(c.history[0].at);
  const hoursSinceRelease = (ms: number): number => (ms - t0Ms) / 3_600_000;
  const xOf = (ms: number): number => Math.log(Math.max(0.05, 1 + hoursSinceRelease(ms)));

  // 1. Collapse consecutive identical reads into refresh states (one observation each).
  const states: State[] = [];
  for (const r of c.history) {
    const atMsR = Date.parse(r.at);
    if (Number.isNaN(atMsR) || !(r.players_tracked > 0)) continue;
    const key = `${r.players_tracked}|${JSON.stringify(r.rates)}`;
    if (states.length > 0 && (states[states.length - 1] as unknown as { key: string }).key === key) continue;
    const s = { atMs: atMsR, n: r.players_tracked, rates: r.rates } as State & { key: string };
    s.key = key;
    states.push(s);
  }
  if (states.length === 0) throw new Error(`${c.id} has no usable history`);
  const last = states[states.length - 1];
  const xTarget = xOf(targetMs);
  const xLast = xOf(last.atMs);

  // 2. Tracked growth: log N against ln(1 + hours since release).
  const xs = states.map((s) => xOf(s.atMs));
  const logN = states.map((s) => Math.log(s.n));
  let logNHat: number;
  let sigmaNPred: number;
  const distinctN = new Set(states.map((s) => s.n)).size;
  if (states.length >= 2 && distinctN >= 2 && Math.max(...xs) - Math.min(...xs) > 1e-9) {
    const [a, b, residStd, sxx, xm] = linFit(xs, logN);
    const q = quadFit(xs, logN);
    let useQuad = false;
    let aQ = a;
    let bQ = b;
    let cQ = 0;
    let residQ = residStd;
    if (q !== null && q[2] < 0 && q[3] < residStd * 0.9) {
      useQuad = true;
      [aQ, bQ, cQ, residQ] = q;
    }
    const bClamped = clamp(useQuad ? bQ : b, 0, 2.5);
    if (useQuad) {
      logNHat = aQ + bClamped * xTarget + cQ * xTarget * xTarget;
      // The concave fit's own curvature already brakes the growth; still cap the multiple.
      logNHat = Math.min(logNHat, Math.log(last.n) + Math.log(15));
      sigmaNPred = Math.max(residQ, 0.1) * Math.sqrt(1 + Math.pow(xTarget - xm, 2) / Math.max(sxx, 1e-9));
    } else {
      logNHat = a + bClamped * xTarget;
      logNHat = Math.min(logNHat, Math.log(last.n) + Math.log(15));
      sigmaNPred = Math.max(residStd, 0.1) * Math.sqrt(1 + Math.pow(xTarget - xm, 2) / Math.max(sxx, 1e-9));
    }
    logNHat = Math.max(logNHat, Math.log(last.n)); // tracked never shrinks
  } else {
    // Regime A: the page has not moved yet. Prior slope from the documented first refresh.
    logNHat = Math.log(states[0].n) + N_SLOPE_PRIOR * (xTarget - xOf(states[0].atMs));
    logNHat = Math.min(logNHat, Math.log(last.n) + Math.log(15));
    logNHat = Math.max(logNHat, Math.log(last.n));
    sigmaNPred = 0.6; // the prior slope is a guess; the spread carries it
  }
  const nHat = Math.exp(logNHat);

  // Refresh-to-refresh logit jump size, pooled across trophies (the spread's honest core).
  const jumps: number[] = [];
  for (let s = 1; s < states.length; s++) {
    for (let j = 0; j < c.trophies.length; j++) {
      const r0 = states[s - 1].rates[j];
      const r1 = states[s].rates[j];
      if (r0 === null || r1 === null) continue;
      jumps.push(Math.abs(logit(r1) - logit(r0)));
    }
  }
  const sigmaJump = states.length >= 2 ? clamp(median(jumps) * 1.4826, 0.3, 2.0) : 1.0;

  // 3. Per-trophy earners power law: log E against log N, anchored at the last state.
  // First pass: raw fitted slopes per trophy, to take tier medians from.
  interface TrophyFit {
    betaRaw: number | null;
    nPts: number;
    residStd: number | null;
    eLast: number;
    rateLast: number;
  }
  const fits: TrophyFit[] = c.trophies.map((t, j) => {
    // Last known rate for this trophy (carry forward over nulls).
    let rateLast: number | null = null;
    for (let s = states.length - 1; s >= 0; s--) {
      if (states[s].rates[j] !== null) {
        rateLast = states[s].rates[j] as number;
        break;
      }
    }
    if (rateLast === null) rateLast = 0.01;
    const eLast = Math.max(rateLast, 0.01) * last.n / 100;
    const pts: Array<[number, number]> = [];
    for (const s of states) {
      const r = s.rates[j];
      if (r === null) continue;
      const e = Math.max(r, 0.01) * s.n / 100;
      if (e >= 0.5) pts.push([Math.log(s.n), Math.log(e)]);
    }
    let betaRaw: number | null = null;
    let residStd: number | null = null;
    let nPts = 0;
    if (pts.length >= 3) {
      const logNs = pts.map((p) => p[0]);
      const logEs = pts.map((p) => p[1]);
      if (Math.max(...logNs) - Math.min(...logNs) > 0.2) {
        const [, b, rs] = linFit(logNs, logEs);
        betaRaw = clamp(b, -0.5, 1.5);
        residStd = rs;
        nPts = pts.length;
      }
    }
    return { betaRaw, nPts, residStd, eLast, rateLast };
  });
  const tierBeta: Record<string, number> = {};
  for (const tier of ['bronze', 'silver', 'gold', 'platinum']) {
    const bs = c.trophies
      .map((t, j) => (t.tier === tier && fits[j].betaRaw !== null ? (fits[j].betaRaw as number) : null))
      .filter((b): b is number => b !== null);
    tierBeta[tier] = bs.length > 0 ? median(bs) : (BETA_PRIOR[tier] ?? 0.05);
  }
  const allBetas = fits.map((f) => f.betaRaw).filter((b): b is number => b !== null);
  const globalBeta = allBetas.length > 0 ? median(allBetas) : 0.05;
  const tierResid: Record<string, number> = {};
  for (const tier of ['bronze', 'silver', 'gold', 'platinum']) {
    const rs = c.trophies
      .map((t, j) => (t.tier === tier && fits[j].residStd !== null ? (fits[j].residStd as number) : null))
      .filter((r): r is number => r !== null);
    tierResid[tier] = rs.length > 0 ? median(rs) : 0.35;
  }

  // 4 + 5. Median and spread per trophy.
  const logRatio = Math.log(Math.max(nHat, last.n) / last.n);
  return c.trophies.map((t, j) => {
    const f = fits[j];
    if (f.rateLast >= 99.99) {
      // Everyone has it and it survived a 6x dilution at 100%: it stays 100%.
      return { id: t.id, q: [99.99, 99.99, 99.99, 99.99, 99.99] };
    }
    const prior = BETA_PRIOR[t.tier ?? ''] ?? tierBeta[t.tier ?? ''] ?? globalBeta;
    const tierMed = tierBeta[t.tier ?? ''] ?? globalBeta;
    const beta =
      f.betaRaw !== null
        ? clamp((f.nPts * f.betaRaw + SHRINK_K * tierMed) / (f.nPts + SHRINK_K), 0, 1.2)
        : clamp(prior, 0, 1.2);
    const eHat = Math.max(f.eLast * Math.exp(beta * logRatio), 1);
    const rHat = clamp((100 * eHat) / nHat, 0.01, 99.99);
    const sigmaFit = f.residStd !== null ? clamp(f.residStd, 0.15, 1.0) : clamp(tierResid[t.tier ?? ''] ?? 0.35, 0.15, 1.0);
    const sigma = clamp(Math.sqrt(sigmaFit * sigmaFit + sigmaNPred * sigmaNPred + sigmaJump * sigmaJump), 0.35, 2.5);
    const l0 = logit(rHat);
    const q = Z.map((z) => clamp(round4(invLogitPct(l0 + z * sigma)), 0.01, 99.99));
    // Enforce non-decreasing after rounding.
    for (let k = 1; k < q.length; k++) q[k] = Math.max(q[k], q[k - 1]);
    return { id: t.id, q };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const trophies = forecast(c);
  const out = `${JSON.stringify({ trophies })}\n`;
  if (Buffer.byteLength(out) > 65_536) throw new Error(`${id}.out would exceed 65,536 bytes`);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), out);
}
