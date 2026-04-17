'use strict';

const KLINE_LIMIT = 300;
const CHART_REFRESH_MS = 60_000;
const STATS_REFRESH_MS = 30_000;
const PRICE_REFRESH_MS = 3_000;

let chart, candleSeries, predSeries;

// ── Chart init ──────────────────────────────────────────────
function initChart() {
  chart = LightweightCharts.createChart(document.getElementById('chart-container'), {
    layout: { background: { color: '#0b0e11' }, textColor: '#6b7784' },
    grid: { vertLines: { color: '#1a1f26' }, horzLines: { color: '#1a1f26' } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: '#252a30' },
    timeScale: { borderColor: '#252a30', timeVisible: true, secondsVisible: false },
    width: document.getElementById('chart-container').clientWidth,
    height: 380,
  });

  candleSeries = chart.addCandlestickSeries({
    upColor: '#0ecb81', downColor: '#f6465d',
    borderUpColor: '#0ecb81', borderDownColor: '#f6465d',
    wickUpColor: '#0ecb81', wickDownColor: '#f6465d',
  });

  predSeries = chart.addLineSeries({
    color: '#00d4aa',
    lineWidth: 2,
    lineStyle: LightweightCharts.LineStyle.Dashed,
    title: 'Prediction',
  });

  window.addEventListener('resize', () => {
    chart.applyOptions({ width: document.getElementById('chart-container').clientWidth });
  });
}

function toChartTime(ms) {
  return Math.floor(ms / 1000);
}

// ── Data loaders ────────────────────────────────────────────
async function loadChart() {
  try {
    const [klRes, predRes] = await Promise.all([
      fetch(`/api/klines?limit=${KLINE_LIMIT}`),
      fetch('/api/latest-prediction'),
    ]);
    const { klines } = await klRes.json();
    const { predictions } = await predRes.json();

    if (klines && klines.length) {
      candleSeries.setData(klines.map(k => ({
        time: toChartTime(k.open_time),
        open: k.open, high: k.high, low: k.low, close: k.close,
      })));
    }

    if (predictions && predictions.length) {
      predSeries.setData(predictions.map(p => ({
        time: toChartTime(p.bar_time),
        value: p.close,
      })));
    } else {
      predSeries.setData([]);
    }
  } catch (e) {
    console.error('Chart load error', e);
  }
}

async function loadStats() {
  try {
    const res = await fetch('/api/stats');
    const { stats } = await res.json();
    renderStats(stats);
  } catch (e) {
    console.error('Stats load error', e);
  }
}

async function loadPrice() {
  try {
    const res = await fetch('/api/price');
    const { price } = await res.json();
    if (price) {
      document.getElementById('price').textContent = price.toLocaleString('en-US', {
        minimumFractionDigits: 2, maximumFractionDigits: 2,
      });
    }
  } catch (_) {}
}

// ── Stats rendering ──────────────────────────────────────────
function fmt(v, decimals = 2) {
  if (v == null) return '—';
  return Number(v).toFixed(decimals);
}

function dirClass(v) {
  if (v == null) return '';
  return v >= 0.55 ? 'good' : v <= 0.45 ? 'bad' : '';
}

function renderStats(stats) {
  const grid = document.getElementById('stats-grid');
  grid.innerHTML = '';
  for (const w of [5, 10, 15, 20]) {
    const s = stats[w];
    const card = document.createElement('div');
    card.className = 'stat-card';
    if (!s) {
      card.innerHTML = `
        <div class="window-label">${w} min</div>
        <div class="no-data">Collecting data…</div>`;
    } else {
      const da = s.direction_accuracy != null ? (s.direction_accuracy * 100).toFixed(1) + '%' : '—';
      const daCls = dirClass(s.direction_accuracy);
      card.innerHTML = `
        <div class="window-label">${w} min</div>
        <div class="stat-row">
          <span class="label">MAE (close)</span>
          <span class="value">${fmt(s.mae, 2)}</span>
        </div>
        <div class="stat-row">
          <span class="label">Max deviation</span>
          <span class="value">${fmt(s.max_deviation, 2)}</span>
        </div>
        <div class="stat-row">
          <span class="label">Min deviation</span>
          <span class="value">${fmt(s.min_deviation, 2)}</span>
        </div>
        <div class="stat-row">
          <span class="label">Direction acc.</span>
          <span class="value ${daCls}">${da}</span>
        </div>
        <div class="stat-row">
          <span class="label">Sample bars</span>
          <span class="value">${s.sample_count ?? '—'}</span>
        </div>`;
    }
    grid.appendChild(card);
  }
}

// ── Predict button ───────────────────────────────────────────
async function onPredict() {
  const btn = document.getElementById('predict-btn');
  const msg = document.getElementById('predict-msg');
  btn.disabled = true;
  msg.className = '';
  msg.textContent = 'Running inference…';
  try {
    const res = await fetch('/api/predict', { method: 'POST' });
    if (res.status === 429) {
      const data = await res.json();
      msg.className = 'err';
      msg.textContent = data.detail;
    } else if (!res.ok) {
      const data = await res.json();
      msg.className = 'err';
      msg.textContent = data.detail || 'Error';
    } else {
      msg.className = 'ok';
      msg.textContent = 'Done! Chart updated.';
      await loadChart();
    }
  } catch (e) {
    msg.className = 'err';
    msg.textContent = String(e);
  } finally {
    setTimeout(() => { btn.disabled = false; msg.textContent = ''; }, 3000);
  }
}

// ── Boot ────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initChart();
  loadChart();
  loadStats();
  loadPrice();

  document.getElementById('predict-btn').addEventListener('click', onPredict);

  setInterval(loadChart, CHART_REFRESH_MS);
  setInterval(loadStats, STATS_REFRESH_MS);
  setInterval(loadPrice, PRICE_REFRESH_MS);
  setInterval(() => {
    document.getElementById('ws-dot').className = 'dot live';
  }, 5000);
});
