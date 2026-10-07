const $ = (s) => document.querySelector(s);
const state = { settings: {}, status: {}, scan: null, manual: loadManual(), open: new Set() };

// ---------- helpers ----------
const api = async (path, opts = {}) => {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
};
const fmtX = (x) => (x == null ? "—" : x >= 100 ? `${x.toFixed(0)}x` : `${x.toFixed(1)}x`);
const fmtPct = (v, d = 1) => (v == null ? "—" : `${v > 0 ? "+" : ""}${(v * 100).toFixed(d)}%`);
const fmtUsd = (v) => (v == null ? "—" : `$${v.toFixed(2)}`);
const fmtP = (p) => `${(p * 100).toFixed(p < 0.1 ? 1 : 0)}%`;
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const when = (iso) => new Date(iso).toLocaleString([], { weekday: "short", hour: "numeric", minute: "2-digit" });
const age = (ts) => {
  if (!ts) return "";
  const s = Math.max(0, Date.now() / 1000 - ts);
  return s < 90 ? `${Math.round(s)}s ago` : s < 5400 ? `${Math.round(s / 60)}m ago` : `${Math.round(s / 3600)}h ago`;
};

function loadManual() {
  try { return JSON.parse(localStorage.getItem("manualX") || "{}"); } catch { return {}; }
}
function saveManual() {
  try { localStorage.setItem("manualX", JSON.stringify(state.manual)); } catch {}
}

// EV / Kelly for an all-in payout multiple x (what the Kalshi app shows: $1 pays $x).
function evalManual(r, x) {
  const s = state.settings;
  const cost = 1 / x;
  const ev = r.q * x - 1;
  const kelly = Math.max(0, (r.q - cost) / (1 - cost));
  const stake = Math.min(s.kelly_fraction * kelly * s.bankroll, s.max_bet_pct * s.bankroll);
  return { ev, ev_lo: r.q_lo * x - 1, stake: ev > 0 ? Math.round(stake * 100) / 100 : 0, x };
}

// Payout used by the "Odds ≤" filter: Kalshi's (quote or manual entry), else the model's fair payout.
const payoutX = (r) => priced(r)?.x ?? r.fair_x;

// Effective pricing for a row: live quote first, else the user's manual entry.
function priced(r) {
  if (r.ev != null) return { ev: r.ev, ev_lo: r.ev_lo, stake: r.stake, x: r.kalshi_x_after_fees, source: r.quote_source };
  const mx = state.manual[r.result_id];
  if (mx > 1) return { ...evalManual(r, mx), source: "manual" };
  return null;
}

// ---------- status + settings ----------
async function loadStatus() {
  state.status = await api("/api/status");
  const st = state.status;
  const pills = [
    st.auth ? `<span class="pill ok">Kalshi <b>live quotes</b> · ${st.env}</span>`
            : `<span class="pill warn" title="Add KALSHI_KEY_ID + key file to .env">Model-only · no API key</span>`,
    `<span class="pill ${st.history.NFL ? "ok" : "warn"}">NFL history ${st.history.NFL ? "✓" : "missing"}</span>`,
    `<span class="pill ${st.history.MLB ? "ok" : "warn"}">MLB history ${st.history.MLB ? "✓" : "missing"}</span>`,
    `<span class="pill">Combos created this week <b>${st.creations_this_week.toLocaleString()}</b> / ${st.creation_limit.toLocaleString()}</span>`,
  ];
  $("#status").innerHTML = pills.join("");
  return st;
}

async function loadSettings() {
  state.settings = await api("/api/settings");
  document.querySelectorAll("[data-key]").forEach((el) => {
    const v = state.settings[el.dataset.key];
    el.value = el.hasAttribute("data-pct") ? +(v * 100).toFixed(2) : v;
  });
}

let saveTimer;
function onSettingInput(e) {
  const el = e.target;
  if (!el.dataset.key || el.value === "") return;
  let v = parseFloat(el.value);
  if (el.hasAttribute("data-pct")) v /= 100;
  state.settings[el.dataset.key] = v;
  clearTimeout(saveTimer);
  saveTimer = setTimeout(async () => {
    state.settings = await api("/api/settings", { method: "PUT", body: JSON.stringify({ [el.dataset.key]: v }) });
    $("#saved").textContent = "Saved";
    $("#saved").classList.add("show");
    setTimeout(() => $("#saved").classList.remove("show"), 1200);
    if (state.scan) await loadScan(state.scan.id);
  }, 350);
}

// ---------- scans ----------
async function startScan() {
  $("#scanBtn").disabled = true;
  const { scan_id } = await api("/api/scan", { method: "POST" });
  pollScan(scan_id);
}

async function pollScan(id) {
  $("#progress").hidden = false;
  $("#scanBtn").disabled = true;
  const scan = await api(`/api/scan/${id}`);
  const p = scan.progress || { phase: "Starting", done: 0, total: 0 };
  $("#progressText").textContent = p.total ? `${p.phase} ${p.done}/${p.total}` : p.phase;
  $("#progressFill").style.width = p.total ? `${(100 * p.done) / p.total}%` : "15%";
  if (scan.status === "running") return setTimeout(() => pollScan(id), 1000);
  $("#progress").hidden = true;
  $("#scanBtn").disabled = false;
  if (scan.status === "error") alert(`Scan failed: ${scan.error}`);
  state.scan = scan;
  renderOpps();
  loadStatus();
}

async function loadScan(id) {
  state.scan = await api(`/api/scan/${id}`);
  renderOpps();
}

function renderOpps() {
  const scan = state.scan;
  const rows = scan ? scan.results : [];
  $("#oppEmpty").hidden = rows.length > 0;
  if (!scan) return;
  const nPriced = rows.filter((r) => priced(r)).length;
  const nPos = rows.filter((r) => (priced(r)?.ev ?? -1) > 0).length;
  $("#scanMeta").innerHTML =
    `Scan #${scan.id} · ${new Date(scan.started_at).toLocaleString()} · <b>${scan.mode === "live" ? "live Kalshi quotes" : "model-only"}</b> · ` +
    `${scan.n_games} games · ${scan.n_pairs.toLocaleString()} pairs modeled · ${rows.length} candidates · ` +
    `${nPriced} priced · <b>${nPos} +EV</b>` +
    (scan.mode === "model" ? ` · <span class="flag">enter the payout from the Kalshi app in the Kalshi column to price a row</span>` : "");

  fillFilter("#fTemplate", rows.map((r) => [r.template, r.template_name]));
  fillFilter("#fGame", rows.map((r) => [r.game, `${r.sport} ${r.game_label}`]));
  const f = { sport: $("#fSport").value, template: $("#fTemplate").value, game: $("#fGame").value };
  const maxX = parseFloat($("#fMaxX").value);
  const shown = rows.filter((r) => (!f.sport || r.sport === f.sport) && (!f.template || r.template === f.template) &&
    (!f.game || r.game === f.game) && !(maxX > 0 && payoutX(r) > maxX));
  const sorted = [...shown].sort((a, b) => {
    const pa = priced(a), pb = priced(b);
    if (pa && pb) return pb.ev - pa.ev;
    if (pa || pb) return pa ? -1 : 1;
    return b.lift - a.lift;
  });

  const tb = $("#oppRows");
  tb.innerHTML = "";
  for (const r of sorted) {
    const pr = priced(r);
    const tr = document.createElement("tr");
    tr.className = `row${pr ? "" : " model-only"}`;
    const evCls = pr ? (pr.ev > 0 ? "pos" : "neg") : "";
    const leg = (l) =>
      `<div class="leg"><span>${esc(l.label)}</span><span class="p" title="${l.p_source === "ladder" ? "Thin market: priced from the player's liquid lower rung" : "Kalshi bid/ask midpoint"}">${l.p_source === "ladder" ? "≈" : ""}${fmtP(l.p ?? l.mid)}</span><span class="role">${esc(l.role)}</span></div>`;
    const kalshiCell = r.ev != null
      ? `${fmtX(r.kalshi_x)}<small>${r.quote_source} · ${age(r.quoted_at)}</small>`
      : `<input type="number" step="0.1" min="1" placeholder="app x" value="${state.manual[r.result_id] ?? ""}" data-rid="${r.result_id}">` +
        (r.quote_error ? `<small class="flag" title="${esc(r.quote_error)}">quote failed</small>` : "");
    tr.innerHTML = `
      <td class="num"><span class="ev ${evCls}">${pr ? fmtPct(pr.ev) : "—"}</span></td>
      <td class="num"><span class="stake">${pr && pr.stake > 0 ? fmtUsd(pr.stake) : "—"}</span></td>
      <td><div class="legs">${leg(r.leg_a)}${leg(r.leg_b)}</div>
          <div class="tmpl">${esc(r.template_name)}${r.wide_leg ? ' · <span class="flag">wide leg spread</span>' : ""}</div></td>
      <td class="game">${esc(r.game_label)}<small>${r.sport} · ${when(r.start)}</small></td>
      <td class="num x">${kalshiCell}</td>
      <td class="num x">${fmtX(r.fair_x)}<small>${fmtX(1 / r.q_hi)}–${fmtX(1 / r.q_lo)}</small></td>
      <td class="num x">${fmtX(r.indep_x)}<small>lift ${r.lift.toFixed(2)}</small></td>
      <td class="num mono">${r.rho == null ? "subset" : r.rho.toFixed(2)}<br><small class="mono" style="color:var(--faint)">${r.n_hist ? r.n_hist.toLocaleString() : ""}</small></td>
      <td><button class="small" data-bet="${r.result_id}">Log bet</button></td>`;
    tb.appendChild(tr);
    if (state.open.has(r.result_id)) tb.appendChild(detailRow(r));
    tr.addEventListener("click", (e) => {
      if (e.target.closest("input,button")) return;
      state.open.has(r.result_id) ? state.open.delete(r.result_id) : state.open.add(r.result_id);
      renderOpps();
    });
  }
  tb.querySelectorAll("input[data-rid]").forEach((el) =>
    el.addEventListener("change", () => {
      const v = parseFloat(el.value);
      if (v > 1) state.manual[el.dataset.rid] = v; else delete state.manual[el.dataset.rid];
      saveManual();
      renderOpps();
    }));
  tb.querySelectorAll("button[data-bet]").forEach((el) =>
    el.addEventListener("click", () => logBet(rows.find((r) => r.result_id == el.dataset.bet))));
}

function fillFilter(sel, pairs) {
  const el = $(sel);
  const cur = el.value;
  const first = el.options[0].outerHTML;
  const uniq = [...new Map(pairs).entries()].sort((a, b) => a[1].localeCompare(b[1]));
  el.innerHTML = first + uniq.map(([v, label]) => `<option value="${esc(v)}">${esc(label)}</option>`).join("");
  el.value = uniq.some(([v]) => v === cur) ? cur : "";
}

function legDetail(l) {
  const book = `bid ${l.yes_bid.toFixed(2)} / ask ${l.yes_ask.toFixed(2)}`;
  return l.p_source === "ladder" ? `${l.label} · ${book} (thin) → ladder ${fmtP(l.p)}` : `${l.label} · ${book}`;
}

function detailRow(r) {
  const node = $("#detailTpl").content.firstElementChild.cloneNode(true);
  const pr = priced(r);
  const items = [
    ["Leg A", legDetail(r.leg_a)],
    ["Leg B", legDetail(r.leg_b)],
    ["P(A) · P(B)", `${fmtP(r.p_a)} · ${fmtP(r.p_b)}`],
    ["Independent joint", `${(r.indep_prob * 100).toFixed(2)}%`],
    ["Model joint (95%)", `${(r.q * 100).toFixed(2)}% (${(r.q_lo * 100).toFixed(2)}–${(r.q_hi * 100).toFixed(2)}%)`],
    ["ρ (95%)", r.rho == null ? "B implies A (subset)" : `${r.rho.toFixed(2)} (${r.rho_lo.toFixed(2)}–${r.rho_hi.toFixed(2)})`],
    ["History pool", `${r.leg_a.role} × ${r.leg_b.role} · n=${r.n_hist.toLocaleString()} · both hit ${r.n11}`],
    ["Break-even payout", `${fmtX(1 / r.q)} all-in`],
  ];
  if (pr) {
    items.push(["EV (pessimistic ρ)", fmtPct(pr.ev_lo)]);
    items.push(["Payout used", `${fmtX(pr.x)} all-in (${pr.source})`]);
  }
  if (r.kalshi_price != null) items.push(["Quote YES price", `$${r.kalshi_price.toFixed(4)} + fee → $${r.kalshi_cost.toFixed(4)}`]);
  if (r.pred_ev != null) items.push(["Predicted EV before quoting", fmtPct(r.pred_ev)]);
  if (r.combo_ticker) items.push(["Combo ticker", r.combo_ticker]);
  const grid = node.querySelector(".detail-grid");
  grid.innerHTML = items.map(([k, v]) => `<div>${esc(k)}<br><b>${esc(v)}</b></div>`).join("");
  if (r.raw_quotes && r.raw_quotes.length) {
    grid.innerHTML += `<pre>Raw maker quotes (verify YES price semantics vs the app):\n${esc(JSON.stringify(
      r.raw_quotes.map((q) => ({ yes_bid: q.yes_bid_dollars, no_bid: q.no_bid_dollars, contracts: q.contracts_fp })), null, 1))}</pre>`;
  }
  return node;
}

async function logBet(r) {
  const pr = priced(r);
  const x = parseFloat(prompt("Payout multiple you got on Kalshi ($1 pays $x)", pr ? pr.x.toFixed(2) : ""));
  if (!(x > 1)) return;
  const stake = parseFloat(prompt("Stake ($)", pr && pr.stake > 0 ? pr.stake.toFixed(2) : ""));
  if (!(stake > 0)) return;
  await api("/api/bets", { method: "POST", body: JSON.stringify({ result_id: r.result_id, stake, payout_x: x }) });
  loadBets();
}

// ---------- history ----------
async function loadBets() {
  const bets = await api("/api/bets");
  $("#betEmpty").hidden = bets.length > 0;
  const tb = $("#betRows");
  tb.innerHTML = bets.map((b) => {
    const d = b.data;
    const ev = d.q * b.payout_x - 1;
    return `<tr>
      <td>${new Date(b.placed_at).toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })}</td>
      <td><div class="legs"><div class="leg">${esc(d.leg_a.label)}</div><div class="leg">${esc(d.leg_b.label)}</div></div>
          <div class="tmpl">${esc(d.game_label)}</div></td>
      <td class="num mono">${fmtUsd(b.stake)}</td>
      <td class="num mono">${fmtX(b.payout_x)}</td>
      <td class="num mono">${fmtPct(ev)}</td>
      <td class="status-${b.status}">${b.status}</td>
      <td class="num mono">${b.pnl == null ? "—" : (b.pnl >= 0 ? "+" : "") + fmtUsd(b.pnl).replace("$-", "-$")}</td>
      <td><button class="small" data-del="${b.id}">Remove</button></td></tr>`;
  }).join("");
  tb.querySelectorAll("[data-del]").forEach((el) => el.addEventListener("click", async () => {
    if (!confirm("Remove this bet from the log?")) return;
    await api(`/api/bets/${el.dataset.del}`, { method: "DELETE" });
    loadBets();
  }));

  const settled = bets.filter((b) => b.status !== "open");
  const staked = bets.reduce((s, b) => s + b.stake, 0);
  const pnl = settled.reduce((s, b) => s + b.pnl, 0);
  const settledStake = settled.reduce((s, b) => s + b.stake, 0);
  const expected = bets.reduce((s, b) => s + b.stake * (b.data.q * b.payout_x - 1), 0);
  $("#kpis").innerHTML = [
    ["Bets", bets.length], ["Staked", fmtUsd(staked)], ["Model expected P&L", fmtUsd(expected)],
    ["Realized P&L", fmtUsd(pnl)], ["ROI (settled)", settledStake ? fmtPct(pnl / settledStake) : "—"],
  ].map(([k, v]) => `<div class="kpi"><span>${k}</span><b>${v}</b></div>`).join("");

  const stats = await api("/api/quote-stats");
  $("#statsEmpty").hidden = stats.length > 0;
  $("#statRows").innerHTML = stats.map((s) => `<tr>
    <td>${esc(s.template_name)}</td><td>${s.sport}</td><td class="num">${s.quotes}</td>
    <td class="num mono ${s.median_kalshi_over_fair > 1 ? "status-won" : ""}">${s.median_kalshi_over_fair.toFixed(2)}</td>
    <td class="num mono">${(s.pct_pos_ev * 100).toFixed(0)}%</td>
    <td>${s.fitted ? "learned" : `exploring (${s.quotes}/30)`}</td></tr>`).join("");

  const scans = await api("/api/scans");
  $("#scanRows").innerHTML = scans.map((s) => `<tr data-scan="${s.id}" class="row">
    <td>${s.id}</td><td>${new Date(s.started_at).toLocaleString()}</td><td>${s.mode}</td>
    <td class="num">${s.n_games}</td><td class="num">${(s.n_pairs || 0).toLocaleString()}</td>
    <td class="num">${s.n_candidates}</td><td class="num">${s.n_quoted}</td>
    <td>${s.status}${s.error ? ` <span class="flag" title="${esc(s.error)}">error</span>` : ""}</td></tr>`).join("");
  document.querySelectorAll("tr[data-scan]").forEach((el) => el.addEventListener("click", async () => {
    await loadScan(+el.dataset.scan);
    switchTab("opps");
  }));
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".panel").forEach((p) => (p.hidden = p.id !== name));
  if (name === "history") loadBets();
}

// ---------- boot ----------
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => switchTab(t.dataset.tab)));
$("#settings").addEventListener("input", onSettingInput);
$("#scanBtn").addEventListener("click", startScan);
["#fSport", "#fTemplate", "#fGame"].forEach((s) => $(s).addEventListener("change", renderOpps));
try { $("#fMaxX").value = localStorage.getItem("maxX") || ""; } catch {}
$("#fMaxX").addEventListener("input", () => {
  try { localStorage.setItem("maxX", $("#fMaxX").value); } catch {}
  renderOpps();
});
$("#settleBtn").addEventListener("click", async () => {
  const { settled } = await api("/api/bets/settle", { method: "POST" });
  if (settled) loadBets();
});

(async () => {
  await loadSettings();
  const st = await loadStatus();
  if (st.running_scan) pollScan(st.running_scan);
  else if (st.latest_scan) loadScan(st.latest_scan);
})();
