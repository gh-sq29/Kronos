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
  return Math.floor(ms / 1000) + 8 * 3600;
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
    const threshold = parseFloat(document.getElementById('volatility-threshold').value) || 0.5;
    const actThreshold = parseFloat(document.getElementById('act-volatility-threshold').value) || 0.1;
    const res = await fetch(`/api/stats?threshold=${threshold}&act_threshold=${actThreshold}`);
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

function dirBar(items) {
  return items.map(([cls, label, v]) =>
    `<span class="${cls}">${label} ${v == null ? '—' : v.toFixed(1) + '%'}</span>`
  ).join('');
}

function renderStats(stats, gridId = 'stats-grid', m = null) {
  const grid = document.getElementById(gridId);
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
      const d = s.direction;
      const daVal = d ? d.dir_acc : (s.direction_accuracy != null ? s.direction_accuracy * 100 : null);
      const da = daVal != null ? daVal.toFixed(1) + '%' : '—';
      const daCls = daVal != null ? (daVal >= 55 ? 'good' : daVal <= 45 ? 'bad' : '') : '';
      const dirSection = d ? `
        <div class="stat-section-label">预测分布</div>
        <div class="dir-row">${dirBar([['dir-long','多',d.pred_long],['dir-flat-long','波多',d.pred_flat_long],['dir-flat-short','波空',d.pred_flat_short],['dir-short','空',d.pred_short]])}</div>
        <div class="stat-section-label">实际分布</div>
        <div class="dir-row">${dirBar([['dir-long','多',d.act_long],['dir-flat-long','波多',d.act_flat_long],['dir-flat-short','波空',d.act_flat_short],['dir-short','空',d.act_short]])}</div>
        <div class="stat-section-label">预测多 → 实际</div>
        <div class="dir-row">${dirBar([['dir-long','多',d.ll],['dir-flat-long','波多',d.lfl],['dir-flat-short','波空',d.lfs],['dir-short','空',d.ls]])}</div>
        <div class="stat-section-label">预测空 → 实际</div>
        <div class="dir-row">${dirBar([['dir-short','空',d.ss],['dir-flat-short','波空',d.sfs],['dir-flat-long','波多',d.sfl],['dir-long','多',d.sl]])}</div>
        ${w === 20 ? `
        <div class="stat-section-label">去重预测多 → 实际（延展跨度）</div>
        <div class="dir-row">${dirBar([['dir-long','多',d.dedup_ll],['dir-flat-long','波多',d.dedup_lfl],['dir-flat-short','波空',d.dedup_lfs],['dir-short','空',d.dedup_ls]])}</div>
        <div class="stat-section-label">去重预测空 → 实际（延展跨度）</div>
        <div class="dir-row">${dirBar([['dir-short','空',d.dedup_ss],['dir-flat-short','波空',d.dedup_sfs],['dir-flat-long','波多',d.dedup_sfl],['dir-long','多',d.dedup_sl]])}</div>` : ''}` : '';
      const dm = s.direction_m;
      const dmVal = dm ? dm.dir_acc : null;
      const dmSection = dm ? `
        <div class="stat-section-label">── 对比 T+${m} ──</div>
        <div class="stat-row">
          <span class="label">方向准确率 (T+${m})</span>
          <span class="value ${dmVal != null ? (dmVal >= 55 ? 'good' : dmVal <= 45 ? 'bad' : '') : ''}">${dmVal != null ? dmVal.toFixed(1) + '%' : '—'}</span>
        </div>
        <div class="stat-section-label">预测多 → 实际(T+${m})</div>
        <div class="dir-row">${dirBar([['dir-long','多',dm.ll],['dir-flat-long','波多',dm.lfl],['dir-flat-short','波空',dm.lfs],['dir-short','空',dm.ls]])}</div>
        <div class="stat-section-label">预测空 → 实际(T+${m})</div>
        <div class="dir-row">${dirBar([['dir-short','空',dm.ss],['dir-flat-short','波空',dm.sfs],['dir-flat-long','波多',dm.sfl],['dir-long','多',dm.sl]])}</div>` : '';
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
          <span class="label">样本数</span>
          <span class="value">${s.sample_count ?? '—'}</span>
        </div>
        ${d ? `
        <div class="stat-row">
          <span class="label">预测多 / 空</span>
          <span class="value"><span class="dir-long">${d.pred_long_count ?? '—'}</span> / <span class="dir-short">${d.pred_short_count ?? '—'}</span></span>
        </div>
        ${w === 20 ? `
        <div class="stat-row">
          <span class="label">去重多 / 空</span>
          <span class="value"><span class="dir-long">${d.dedup_long_count ?? '—'}</span> / <span class="dir-short">${d.dedup_short_count ?? '—'}</span></span>
        </div>
        <div class="stat-row">
          <span class="label">去重 Acc</span>
          <span class="value ${d.dedup_dir_acc != null ? (d.dedup_dir_acc >= 55 ? 'good' : d.dedup_dir_acc <= 45 ? 'bad' : '') : ''}">${d.dedup_dir_acc != null ? d.dedup_dir_acc.toFixed(1) + '%' : '—'}</span>
        </div>
        <div class="stat-row">
          <span class="label">去重 PnL 合计</span>
          <span class="value ${d.dedup_pnl_sum != null ? (d.dedup_pnl_sum > 0 ? 'good' : d.dedup_pnl_sum < 0 ? 'bad' : '') : ''}">${d.dedup_pnl_sum != null ? (d.dedup_pnl_sum > 0 ? '+' : '') + d.dedup_pnl_sum.toFixed(3) + '%' : '—'}</span>
        </div>` : ''}` : ''}
        ${dirSection}
        ${dmSection}`;
    }
    grid.appendChild(card);
  }
}

// ── Historical stats ─────────────────────────────────────────
function fmtLocalDatetime(date) {
  const pad = n => String(n).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth()+1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

async function loadHistoricalStats() {
  const start = document.getElementById('hist-start').value;
  const end   = document.getElementById('hist-end').value;
  const status = document.getElementById('hist-status');
  const btn    = document.getElementById('hist-query-btn');
  if (!start || !end) { status.textContent = '请填写起止时间'; return; }
  if (start >= end)   { status.textContent = '结束时间须晚于开始时间'; return; }
  btn.disabled = true;
  status.textContent = '查询中…';
  const threshold    = parseFloat(document.getElementById('hist-volatility-threshold').value) || 0.5;
  const actThreshold = parseFloat(document.getElementById('hist-act-volatility-threshold').value) || 0.1;
  const m            = parseInt(document.getElementById('hist-m').value) || 0;
  try {
    const res = await fetch(`/api/stats/range?start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}&threshold=${threshold}&act_threshold=${actThreshold}&m=${m}`);
    if (!res.ok) { const d = await res.json(); status.textContent = '查询失败: ' + (d.detail || res.status); return; }
    const { stats, m: mParam } = await res.json();
    renderStats(stats, 'hist-stats-grid', mParam > 0 ? mParam : null);
    status.textContent = `查询成功 · ${start} ~ ${end}`;
  } catch (e) {
    status.textContent = '查询失败: ' + e;
  } finally {
    btn.disabled = false;
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

// ── Point history query ──────────────────────────────────────
function msToLocal(ms) {
  const d = new Date(ms);
  const pad = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function pctCell(pct) {
  if (pct == null) return '<td style="text-align:right;color:var(--muted)">—</td>';
  const cls = pct > 0 ? 'color:var(--red)' : pct < 0 ? 'color:var(--green)' : 'color:var(--muted)';
  const sign = pct > 0 ? '+' : '';
  return `<td style="text-align:right;font-weight:600;${cls}">${sign}${pct.toFixed(3)}%</td>`;
}

async function loadPointHistory() {
  const startVal = document.getElementById('pt-start').value;
  const duration = parseInt(document.getElementById('pt-duration').value) || 20;
  const status   = document.getElementById('pt-status');
  const btn      = document.getElementById('pt-query-btn');

  if (!startVal) { status.textContent = '请填写起始时间'; return; }

  btn.disabled = true;
  status.textContent = '查询中…';
  document.getElementById('pt-result').style.display = 'none';

  try {
    const res = await fetch(`/api/point-history?start=${encodeURIComponent(startVal)}&duration=${duration}`);
    if (!res.ok) {
      const d = await res.json();
      status.textContent = '查询失败: ' + (d.detail || res.status);
      return;
    }
    const { results } = await res.json();
    const tbody = document.getElementById('pt-tbody');
    tbody.innerHTML = '';

    for (const row of results) {
      const baseClose = row.baseline_close != null ? row.baseline_close.toLocaleString() : '—';
      let cells = `
        <td style="padding:5px 10px;white-space:nowrap;">${msToLocal(row.query_ms)}</td>
        <td style="padding:5px 10px;white-space:nowrap;color:var(--muted);">${msToLocal(row.baseline_ms)}</td>
        <td style="padding:5px 10px;text-align:right;">${baseClose}</td>`;

      for (const step of [5, 10, 15, 20]) {
        const s = row.steps[step];
        const borderLeft = 'border-left:1px solid var(--border);';
        if (!s) {
          cells += `<td style="${borderLeft}padding:5px 8px;text-align:right;color:var(--muted)">—</td><td style="text-align:right;color:var(--muted)">—</td>`;
        } else {
          cells += `<td style="${borderLeft}padding:5px 8px;text-align:right;">${s.close.toLocaleString()}</td>${pctCell(s.pct)}`;
        }
      }

      const tr = document.createElement('tr');
      tr.style.borderBottom = '1px solid var(--border)';
      tr.innerHTML = cells;
      tbody.appendChild(tr);
    }

    document.getElementById('pt-result').style.display = 'block';
    status.textContent = `查询完成 · ${results.length} 条记录`;
  } catch (e) {
    status.textContent = '查询失败: ' + e;
  } finally {
    btn.disabled = false;
  }
}

// ── Boot ────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initChart();
  loadChart();
  loadStats();
  loadPrice();

  const now = new Date();
  document.getElementById('hist-end').value   = fmtLocalDatetime(now);
  document.getElementById('hist-start').value = fmtLocalDatetime(new Date(now - 2 * 3600 * 1000));
  document.getElementById('hist-query-btn').addEventListener('click', loadHistoricalStats);
  document.getElementById('pt-query-btn').addEventListener('click', loadPointHistory);
  document.getElementById('pt-start').value = fmtLocalDatetime(new Date(now - 30 * 60 * 1000));

  document.getElementById('predict-btn').addEventListener('click', onPredict);
  document.getElementById('volatility-threshold').addEventListener('change', loadStats);
  document.getElementById('act-volatility-threshold').addEventListener('change', loadStats);
  document.getElementById('refresh-stats-btn').addEventListener('click', loadStats);

  setInterval(loadChart, CHART_REFRESH_MS);
  setInterval(loadStats, STATS_REFRESH_MS);
  setInterval(loadPrice, PRICE_REFRESH_MS);
  setInterval(() => {
    document.getElementById('ws-dot').className = 'dot live';
  }, 5000);
});
