// pumpfun-top5-forecast-01: Calibrated probabilistic forecaster v2
// Multi-horizon Monte Carlo simulation with reserve-weighted hazard dynamics, momentum, and newcomer fields.
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

interface Row {
  rank: number;
  mint: string;
  market_cap_lamports: string | number;
  venue: string;
  slot: number;
  name: string;
  symbol: string;
  created_at: string;
  complete: boolean;
  mayhem: boolean;
  quote_reserves_lamports: string | number;
}

interface Mark {
  schema: string;
  at: string;
  ok: boolean;
  counts: any;
  rows: Row[] | null;
}

interface Task {
  places: number;
  output: string;
  history_until: string;
  reads: string[];
  size: { edges: number[] };
}

// 1. Read task and history
const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as Task;
const lines = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8').split('\n').filter((l) => l.trim() !== '');
const marks = lines.map((l) => JSON.parse(l) as Mark).filter((m) => m.ok && Array.isArray(m.rows) && m.rows.length > 0);

if (marks.length === 0) throw new Error('the history holds no ok mark');
const lastMark = marks[marks.length - 1]!;
const cutTime = new Date(task.history_until).getTime();

// 2. Compute evaluation horizons
const readTimes = task.reads.map((r) => new Date(r).getTime());
const horizonHours = readTimes.map((t) => Math.max(0, (t - cutTime) / 3_600_000));

// 3. Track coin history across marks
const sizeOf = (lamports: string | number | bigint): number => Math.log10(Number(BigInt(lamports))) - 9;

const coinHistories = new Map<string, Array<{ time: number; logCap: number; rank: number }>>();
for (const m of marks) {
  const mTime = new Date(m.at).getTime();
  for (const r of m.rows ?? []) {
    if (!coinHistories.has(r.mint)) coinHistories.set(r.mint, []);
    coinHistories.get(r.mint)!.push({
      time: mTime,
      logCap: sizeOf(r.market_cap_lamports),
      rank: r.rank,
    });
  }
}

interface Candidate {
  mint: string;
  symbol: string;
  venue: string;
  logCap: number;
  reservesSol: number;
  ageHours: number;
  velocity: number;
  lastRank: number;
}

const candidates: Candidate[] = [];
for (const r of lastMark.rows ?? []) {
  const cap = BigInt(r.market_cap_lamports);
  const logCap = sizeOf(cap);
  const reserves = Number(BigInt(r.quote_reserves_lamports ?? 0)) / 1e9;
  const createTime = new Date(r.created_at).getTime();
  const ageHours = Math.max(0.1, (cutTime - createTime) / 3_600_000);

  const history = coinHistories.get(r.mint) ?? [];
  let velocity = 0;
  if (history.length >= 2) {
    const oldest = history[Math.max(0, history.length - 3)]!;
    const newest = history[history.length - 1]!;
    const dt = (newest.time - oldest.time) / 3_600_000;
    if (dt > 0.05) velocity = (newest.logCap - oldest.logCap) / dt;
  }

  // Focus on pool tokens or mature curves
  if (r.venue === 'pool' || (r.venue === 'migrating' && r.complete) || (r.venue === 'curve' && logCap > 2.2)) {
    candidates.push({
      mint: r.mint,
      symbol: r.symbol,
      venue: r.venue,
      logCap,
      reservesSol: reserves,
      ageHours,
      velocity: Math.max(-0.4, Math.min(0.4, velocity)),
      lastRank: r.rank,
    });
  }
}

candidates.sort((a, b) => b.logCap - a.logCap);

// 4. Deterministic Seeded PRNG (xoshiro128**)
let s0 = 2463534242, s1 = 123456789, s2 = 987654321, s3 = 135792468;
function rand(): number {
  const t = s1 << 9;
  let v = s0 * 5;
  v = ((v << 7) | (v >>> 25)) * 9;
  s2 ^= s0;
  s3 ^= s1;
  s1 ^= s2;
  s0 ^= s3;
  s2 ^= t;
  s3 = (s3 << 11) | (s3 >>> 21);
  return (v >>> 0) / 4294967296;
}

function randn(): number {
  let u = 0, v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

// 5. Monte Carlo Simulation Engine
const N_SIMS = 30_000;
const PLACES = task.places;

const placeCounts: Array<Map<string, number>> = Array.from({ length: PLACES }, () => new Map<string, number>());
const newcomerCounts: number[] = new Array(PLACES).fill(0);
const placeCapSamples: number[][] = Array.from({ length: PLACES }, () => []);

for (let rIdx = 0; rIdx < horizonHours.length; rIdx++) {
  const dt = horizonHours[rIdx]!;
  const sqrtDt = Math.sqrt(Math.max(0.1, dt));

  for (let sim = 0; sim < N_SIMS; sim++) {
    const simPool: Array<{ id: string; cap: number; isNew: boolean }> = [];

    // Incumbents simulation with reserve weighting
    for (const c of candidates) {
      const reserveFactor = Math.min(1.0, c.reservesSol / 200);
      const hazard = Math.max(0.06, 0.20 - 0.06 * (c.logCap - 3.0) - 0.04 * reserveFactor - 0.03 * Math.min(1.0, c.ageHours));
      const crashProb = 1 - Math.exp(-hazard * Math.max(0.1, dt));

      if (rand() < crashProb) {
        simPool.push({ id: c.mint, cap: 1.55 + 0.12 * randn(), isNew: false });
      } else {
        const drift = -0.04 * dt + 0.10 * c.velocity * Math.min(1.0, dt);
        const diff = 0.22 * sqrtDt * randn();
        simPool.push({ id: c.mint, cap: c.logCap + drift + diff, isNew: false });
      }
    }

    // Newcomer simulation: rate scales with elapsed time
    const lambdaNew = 0.5 + 0.6 * dt;
    const nNew = Math.floor(lambdaNew) + (rand() < (lambdaNew - Math.floor(lambdaNew)) ? 1 : 0);
    for (let k = 0; k < nNew; k++) {
      const newCap = 3.35 + 0.35 * randn();
      simPool.push({ id: `__new_${k}__`, cap: newCap, isNew: true });
    }

    // Sort descending by simulated cap
    simPool.sort((a, b) => b.cap - a.cap);

    for (let k = 0; k < PLACES; k++) {
      const topCoin = simPool[k];
      if (topCoin) {
        placeCapSamples[k]!.push(topCoin.cap);
        if (topCoin.isNew) {
          newcomerCounts[k] = (newcomerCounts[k] ?? 0) + 1;
        } else {
          const m = placeCounts[k]!;
          m.set(topCoin.id, (m.get(topCoin.id) ?? 0) + 1);
        }
      }
    }
  }
}

const totalScenarios = horizonHours.length * N_SIMS;

// 6. Build probabilities list
const probabilities: Array<{ mint: string; place: number; p: number }> = [];

for (let k = 0; k < PLACES; k++) {
  const place = k + 1;
  const m = placeCounts[k]!;
  const newCount = newcomerCounts[k]!;
  const empiricalNewcomerProb = newCount / totalScenarios;

  const targetUnnamed = Math.min(0.65, Math.max(0.25, empiricalNewcomerProb + 0.05));
  const maxNamedSum = 1.0 - targetUnnamed;

  const sortedCoins = Array.from(m.entries()).sort((a, b) => b[1] - a[1]);
  const incumbentTotal = sortedCoins.reduce((acc, [, cnt]) => acc + cnt, 0);

  let currentNamedSum = 0;
  const placeRows: Array<{ mint: string; place: number; p: number }> = [];

  for (const [mint, count] of sortedCoins) {
    if (placeRows.length >= 15) break;
    const normalizedP = (count / incumbentTotal) * maxNamedSum;
    const p = Math.floor(normalizedP * 100_000) / 100_000;
    if (p >= 0.005) {
      placeRows.push({ mint, place, p });
      currentNamedSum += p;
    }
  }

  probabilities.push(...placeRows);
}

// 7. Top 5 Agreement Maximizer
const candidateMints = candidates.slice(0, 10).map((c) => c.mint);
const coinRankProb = new Map<string, number[]>();
for (const mint of candidateMints) {
  const probs = [0, 0, 0, 0, 0];
  for (let k = 0; k < PLACES; k++) {
    const cnt = placeCounts[k]!.get(mint) ?? 0;
    probs[k] = cnt / totalScenarios;
  }
  coinRankProb.set(mint, probs);
}

let bestAgreement = -1;
let bestTop5: string[] = [];

function searchPerm(current: string[], used: Set<string>) {
  if (current.length === 5) {
    let score = 0;
    for (let slot = 0; slot < 5; slot++) {
      const mint = current[slot]!;
      const probs = coinRankProb.get(mint)!;
      for (let actual = 0; actual < 5; actual++) {
        score += (5 - Math.abs(slot - actual)) * probs[actual]!;
      }
    }
    if (score > bestAgreement) {
      bestAgreement = score;
      bestTop5 = [...current];
    }
    return;
  }

  for (const mint of candidateMints) {
    if (!used.has(mint)) {
      used.add(mint);
      current.push(mint);
      searchPerm(current, used);
      current.pop();
      used.delete(mint);
    }
  }
}

searchPerm([], new Set());

// 8. Size Bin Distribution Optimization
const edges = task.size.edges;
const numBins = edges.length + 1;

function binOf(val: number): number {
  let lo = 0, hi = edges.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (edges[mid]! <= val) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

const sizes: Array<{ place: number; bin: number; p: number }> = [];

for (let k = 0; k < PLACES; k++) {
  const place = k + 1;
  const samples = placeCapSamples[k]!;

  const rawHist = new Array<number>(numBins).fill(0);
  for (const val of samples) {
    rawHist[binOf(val)]! += 1;
  }

  const hist = rawHist.map((cnt) => cnt / samples.length);

  // Kernel smoothing (5-tap Gaussian)
  const smoothed = new Array<number>(numBins).fill(0);
  const kernel = [0.06, 0.24, 0.40, 0.24, 0.06];
  for (let b = 0; b < numBins; b++) {
    let acc = 0;
    for (let offset = -2; offset <= 2; offset++) {
      const idx = b + offset;
      if (idx >= 0 && idx < numBins) {
        acc += hist[idx]! * kernel[offset + 2]!;
      }
    }
    smoothed[b] = acc;
  }

  // Broad regularizing prior around mean
  const meanSample = samples.reduce((a, b) => a + b, 0) / samples.length;
  const PRIOR_SD = 0.45;
  const prior = new Array<number>(numBins).fill(0);
  for (let b = 0; b < numBins; b++) {
    const lo = b === 0 ? 2.0 : edges[b - 1]!;
    const hi = b === edges.length ? 6.0 : edges[b]!;
    const mid = 0.5 * (lo + hi);
    const z = (mid - meanSample) / PRIOR_SD;
    prior[b] = Math.exp(-0.5 * z * z);
  }
  const priorSum = prior.reduce((a, b) => a + b, 0);
  const normalizedPrior = prior.map((v) => v / priorSum);

  const blended = new Array<number>(numBins).fill(0);
  for (let b = 0; b < numBins; b++) {
    blended[b] = 0.92 * smoothed[b]! + 0.08 * normalizedPrior[b]!;
  }

  for (let b = 0; b < numBins; b++) {
    const p = Math.floor(blended[b]! * 1_000_000) / 1_000_000;
    if (p >= 0.0001) {
      sizes.push({ place, bin: b, p });
    }
  }
}

// 9. Write forecast.json
writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities, top5: bestTop5, sizes })}\n`);

