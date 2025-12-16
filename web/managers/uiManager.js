// UIManager: orchestrates DOM, charts, and user interactions for pages
import { DataService } from '../services/dataService.js';
import { I18nService } from '../services/i18nService.js';

let damageChart = null;
let eventsTimeseriesChart = null;
let playersOverTimeChart = null;
let leaderChart = null;
let playerMaxChart = null;
let isUpdatingFromClick = false;

function formatDamage(value) {
  if (value >= 1_000_000_000) return (value / 1_000_000_000).toFixed(2) + 'B';
  if (value >= 1_000_000) return (value / 1_000_000).toFixed(2) + 'M';
  if (value >= 1_000) return (value / 1_000).toFixed(2) + 'K';
  return String(value);
}

function colorPalette(i) {
  const colors = ['#4BC0C0','#FF6384','#36A2EB','#FFCE56','#9966FF','#FF9F40','#66BB6A','#EF5350','#29B6F6','#AB47BC','#D4E157','#FF7043','#EC407A','#26A69A','#8D6E63'];
  return colors[i % colors.length];
}

function updateCurrentEventInfo(event) {
  const t = I18nService.t;
  const nameEl = document.getElementById('last-event-name');
  if (nameEl) nameEl.textContent = event.name;
  const summaryDateEl = document.getElementById('summary-event-date');
  if (summaryDateEl) summaryDateEl.textContent = event.date;
  const rankingDateEl = document.getElementById('ranking-date');
  if (rankingDateEl) rankingDateEl.textContent = event.date;
  const damageChartDateEl = document.getElementById('damage-chart-date');
  if (damageChartDateEl) damageChartDateEl.textContent = event.date;
  const ralliesEl = document.getElementById('last-event-rallies-count');
  if (ralliesEl) {
    const ralliesCount = event.rally_count_total ?? (event.rallies ? event.rallies.length : 0);
    ralliesEl.textContent = ralliesCount || 0;
  }
  const uniquePlayerIds = new Set();
  if (event.rallies) {
    for (const rally of event.rallies) {
      for (const p of rally.participants || []) {
        uniquePlayerIds.add(p.canonical_id || p.id || p.name);
      }
    }
  }
  const playersCountEl = document.getElementById('last-event-players-count');
  if (playersCountEl) {
    const explicitPlayers = Array.isArray(event.players) ? event.players.length : 0;
    const count = explicitPlayers || uniquePlayerIds.size || 0;
    playersCountEl.textContent = count;
  }
  const totalDamageEl = document.getElementById('last-event-total-damage');
  if (totalDamageEl) totalDamageEl.textContent = formatDamage(event.alliance_total_damage || 0);

  // Update section titles with date
  const titleEl = document.getElementById('last-event-label');
  if (titleEl) titleEl.textContent = `${t('summary')} (${event.date})`;
  const rankingTitleEl = document.getElementById('ranking-title');
  if (rankingTitleEl) rankingTitleEl.innerHTML = `${t('ranking')} (<span id="ranking-date">${event.date}</span>)`;
  const damageTitleEl = document.getElementById('damage-chart-title');
  if (damageTitleEl) damageTitleEl.innerHTML = `${t('damage_chart')} (<span id="damage-chart-date">${event.date}</span>)`;
}

function updateRankingTable(event) {
  const tbody = document.querySelector('#ranking-table tbody');
  if (!tbody) return;
  tbody.innerHTML = '';
  const players = event.players || [];
  if (players.length === 0) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = 3;
    td.textContent = I18nService.t('no_data_available');
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }
  const lang = I18nService.getLanguage();
  players.slice().sort((a,b)=>b.total_damage-a.total_damage).forEach((player, index) => {
    const tr = document.createElement('tr');
    const rankTd = document.createElement('td');
    rankTd.textContent = index + 1;
    const nameTd = document.createElement('td');
    const pid = player.id || player.canonical_id || player.name;
    nameTd.textContent = I18nService.playerName(pid, { translit: true });
    const dmgTd = document.createElement('td');
    dmgTd.textContent = formatDamage(player.total_damage);
    tr.appendChild(rankTd); tr.appendChild(nameTd); tr.appendChild(dmgTd);
    tbody.appendChild(tr);
  });
}

function renderDamageChart(event) {
  const t = I18nService.t;
  const el = document.getElementById('damage-chart');
  if (!el) return;
  const players = event.players || [];
  
  // Show "no data available" message if no players
  const parent = el.parentElement;
  const msgEl = parent?.querySelector('.no-data-message');
  if (players.length === 0) {
    if (damageChart) damageChart.destroy();
    if (msgEl) {
      msgEl.style.display = 'block';
      msgEl.textContent = t('no_data_available');
    } else if (parent) {
      const msg = document.createElement('div');
      msg.className = 'no-data-message';
      msg.style.cssText = 'padding: 20px; text-align: center; color: #999; font-size: 16px;';
      msg.textContent = t('no_data_available');
      parent.insertBefore(msg, el);
    }
    el.style.display = 'none';
    return;
  }
  
  // Hide "no data" message and show chart
  if (msgEl) msgEl.style.display = 'none';
  el.style.display = 'block';
  
  const ctx = el.getContext('2d');
  const sortedPlayers = players.slice().sort((a,b)=>b.total_damage-a.total_damage).slice(0,15);
  const labels = sortedPlayers.map(p => {
    const pid = p.id || p.canonical_id || p.name;
    return I18nService.playerName(pid, { translit: true });
  });
  const values = sortedPlayers.map(p => p.total_damage);
  if (damageChart) damageChart.destroy();
  damageChart = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ label: t('label_damage'), data: values, backgroundColor: 'rgba(75, 192, 192, 0.7)' }] },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: t('player') }, ticks: { color: '#f5f5f5' } },
        y: { title: { display: true, text: t('label_damage') }, ticks: { color: '#f5f5f5', callback: v => formatDamage(v) } },
      },
    },
  });
}

function renderLastEventsTable(events) {
  const t = I18nService.t;
  const el = document.getElementById('events-timeseries');
  if (!el) return;
  
  // Find or create container for table
  let container = el.parentElement?.querySelector('.last-events-container');
  if (!container) {
    container = document.createElement('div');
    container.className = 'last-events-container';
    el.parentElement?.insertBefore(container, el);
  }
  
  // Sort and get last 5 events (ascending order - oldest to newest)
  const sorted = events.slice().sort((a, b) => (a.date < b.date ? -1 : 1));
  const lastFive = sorted.slice(-5);
  
  // Build transposed table: dates as columns, metrics as rows
  let html = `<h3>${t('events_list')}</h3>
    <table style="width: 100%; border-collapse: collapse; color: #f5f5f5;">
    <thead style="background-color: #1a1a1a;">
      <tr>
        <th style="padding: 10px; text-align: left; border-bottom: 2px solid #4BC0C0;"></th>`;
  
  // Header row: dates
  for (const ev of lastFive) {
    html += `<th style="padding: 10px; text-align: right; border-bottom: 2px solid #4BC0C0;">${ev.date}</th>`;
  }
  html += `</tr></thead><tbody>`;
  
  // Damage row
  html += `<tr style="border-bottom: 1px solid #333;">
    <td style="padding: 10px; font-weight: bold;">${t('events_damage')}</td>`;
  for (const ev of lastFive) {
    const damage = formatDamage(ev.alliance_total_damage || 0);
    html += `<td style="padding: 10px; text-align: right;">${damage}</td>`;
  }
  html += `</tr>`;
  
  // Players row
  html += `<tr style="border-bottom: 1px solid #333;">
    <td style="padding: 10px; font-weight: bold;">${t('players')}</td>`;
  for (const ev of lastFive) {
    const players = (ev.players ? ev.players.length : 0);
    html += `<td style="padding: 10px; text-align: right;">${players}</td>`;
  }
  html += `</tr>`;
  
  // Rallies row
  html += `<tr style="border-bottom: 1px solid #333;">
    <td style="padding: 10px; font-weight: bold;">${t('events_rallies')}</td>`;
  for (const ev of lastFive) {
    const rallies = ev.rally_count_total ?? (ev.rallies ? ev.rallies.length : 0);
    html += `<td style="padding: 10px; text-align: right;">${rallies}</td>`;
  }
  html += `</tr>`;
  
  html += '</tbody></table>';
  container.innerHTML = html;
}

function renderEventsTimeseries(series, events) {
  const t = I18nService.t;
  const el = document.getElementById('events-timeseries');
  if (!el) return;
  const ctx = el.getContext('2d');
  const damagePoints = series.map(s => ({ x: s.date, y: s.totalDamage }));
  const ralliesPoints = series.map(s => ({ x: s.date, y: s.rallies }));
  const playersPoints = series.map(s => ({ x: s.date, y: s.playersCount }));
  if (eventsTimeseriesChart) eventsTimeseriesChart.destroy();
  eventsTimeseriesChart = new Chart(ctx, {
    data: { datasets: [
      { label: t('label_total_damage'), data: damagePoints, borderColor: '#4BC0C0', backgroundColor: 'rgba(75,192,192,0.2)', yAxisID: 'yDamage', tension: 0.2, type: 'line' },
      { label: t('label_rallies'), data: ralliesPoints, borderColor: '#FF6384', backgroundColor: 'rgba(255,99,132,0.2)', yAxisID: 'yCount', tension: 0.2, type: 'line' },
      { label: t('label_players_present'), data: playersPoints, borderColor: '#36A2EB', backgroundColor: 'rgba(54,162,235,0.2)', yAxisID: 'yCount', tension: 0.2, type: 'line' },
    ]},
    options: {
      responsive: true,
      interaction: {
        mode: 'index',
        intersect: false,
      },
      onHover: (event) => {
        event.native.target.style.cursor = 'pointer';
      },
      scales: {
        x: { type: 'time', time: { unit: 'day' }, title: { display: true, text: t('date') }, ticks: { color: '#f5f5f5' } },
        yDamage: { type: 'linear', position: 'left', ticks: { color: '#f5f5f5', callback: v => formatDamage(v) }, title: { display: true, text: t('label_total_damage') } },
        yCount: { type: 'linear', position: 'right', grid: { drawOnChartArea: false }, ticks: { color: '#f5f5f5' }, title: { display: true, text: 'Count' } },
      },
      plugins: { 
        legend: { position: 'top' },
        tooltip: {
          mode: 'index',
          intersect: false,
          callbacks: {
            afterBody: () => {
              return '\n' + t('click_to_select');
            }
          }
        }
      },
      onClick: (evt, elements, chart) => {
        if (!events || isUpdatingFromClick) return;
        
        let dataIndex = -1;
        
        // If clicked on a data point
        if (elements && elements.length > 0) {
          dataIndex = elements[0].index;
        } else {
          // If clicked on axis or empty area, find nearest point
          const canvasPosition = Chart.helpers.getRelativePosition(evt, chart);
          const xValue = chart.scales.x.getValueForPixel(canvasPosition.x);
          
          if (xValue) {
            // Find closest date
            let minDiff = Infinity;
            series.forEach((s, idx) => {
              const diff = Math.abs(s.date.getTime() - xValue);
              if (diff < minDiff) {
                minDiff = diff;
                dataIndex = idx;
              }
            });
          }
        }
        
        if (dataIndex < 0 || dataIndex >= series.length) return;
        
        const selectedKey = series[dataIndex]?.dateStr;
        if (!selectedKey) return;
        const selectedEvent = events.find(e => (e.date || e.id) === selectedKey);
        if (selectedEvent) {
          isUpdatingFromClick = true;
          const idx = events.indexOf(selectedEvent);
          DataService.setSelectedEventIndex(idx);
          console.log('Clicked event:', selectedEvent);
          updateCurrentEventInfo(selectedEvent);
          updateRankingTable(selectedEvent);
          renderDamageChart(selectedEvent);
          isUpdatingFromClick = false;
        }
      },
    },
  });
}

function wireEventsTimeseriesToggles() {
  const t = I18nService.t;
  const getLabels = () => ({
    toggleDamage: t('label_total_damage'),
    toggleRallies: t('label_rallies'),
    togglePlayers: t('label_players_present'),
  });
  Object.keys(getLabels()).forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('change', () => {
      if (!eventsTimeseriesChart) return;
      const labels = getLabels();
      const label = labels[id];
      const ds = eventsTimeseriesChart.data.datasets.find(d => d.label === label);
      if (ds) { ds.hidden = !el.checked; eventsTimeseriesChart.update(); }
    });
  });
}

function renderPlayersOverTime(playerSeriesMap, topN) {
  const t = I18nService.t;
  const el = document.getElementById('players-over-time');
  if (!el) return;
  const ctx = el.getContext('2d');
  const entries = Array.from(playerSeriesMap.entries());
  entries.sort((a,b)=>b[1].totalAllTime - a[1].totalAllTime);
  const top = entries.slice(0, topN);
  const datasets = top.map(([id, meta], idx) => {
    const color = colorPalette(idx);
    const displayName = I18nService.playerName(id, { translit: true });
    return { label: displayName, data: meta.series, borderColor: color, backgroundColor: color + 'CC', tension: 0.2, fill: false, borderWidth: 3, pointRadius: 6, pointBorderWidth: 2, pointBorderColor: '#fff', pointBackgroundColor: color, pointHoverRadius: 8 };
  });
  if (playersOverTimeChart) playersOverTimeChart.destroy();
  playersOverTimeChart = new Chart(ctx, {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true,
      parsing: false,
      scales: {
        x: { type: 'time', time: { unit: 'day' }, title: { display: true, text: t('date') }, ticks: { color: '#f5f5f5' } },
        y: { title: { display: true, text: t('label_damage') }, ticks: { color: '#f5f5f5', callback: v => formatDamage(v) } },
      },
      plugins: { legend: { position: 'top' } },
    },
  });
}

function renderLeaderAnalytics(leaderId, maps) {
  const t = I18nService.t;
  const el = document.getElementById('leader-rallies-over-time');
  if (!el) return;
  const ctx = el.getContext('2d');
  const lineSeries = maps.perEventSum.get(leaderId) || [];
  const showRallies = document.getElementById('leaderShowRallies')?.checked;
  if (leaderChart) leaderChart.destroy();
  leaderChart = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [
        { label: t('label_sum_per_event'), data: lineSeries, borderColor: '#4BC0C0', backgroundColor: 'rgba(75,192,192,0.3)', parsing: false, tension: 0.2, borderWidth: 3, pointRadius: 6, pointBorderWidth: 2, pointBorderColor: '#fff', pointBackgroundColor: '#4BC0C0', pointHoverRadius: 8 },
        ...(showRallies ? [{ type: 'scatter', label: t('label_per_rally'), data: (maps.perRally.get(leaderId) || []), parsing: false, pointRadius: 4, backgroundColor: '#FF6384' }] : []),
      ],
    },
    options: {
      responsive: true,
      scales: {
        x: { type: 'time', time: { unit: 'day' }, title: { display: true, text: t('date') }, ticks: { color: '#f5f5f5' } },
        y: { title: { display: true, text: t('label_damage') }, ticks: { color: '#f5f5f5', callback: v => formatDamage(v) } },
      },
      plugins: { legend: { position: 'top' } },
    },
  });
}

function renderPlayerMax(playerId, participation) {
  const t = I18nService.t;
  const el = document.getElementById('player-max-rally');
  if (!el) return;
  const ctx = el.getContext('2d');
  const rows = participation.get(playerId) || [];
  const byEvent = new Map();
  for (const row of rows) {
    const key = row.eventId;
    if (!byEvent.has(key) || row.value > byEvent.get(key).value) byEvent.set(key, row);
  }
  const maxRows = Array.from(byEvent.values()).sort((a,b)=>a.date - b.date);
  const tbody = document.querySelector('#player-max-table tbody');
  if (tbody) {
    tbody.innerHTML = '';
    for (const r of maxRows) {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${r.date.toISOString().slice(0,10)}</td><td>${r.rallyId}</td><td>${r.leaderName || ''}</td><td>${formatDamage(r.value)}</td>`;
      tbody.appendChild(tr);
    }
  }
  const labels = maxRows.map(r => r.date);
  const values = maxRows.map(r => r.value);
  if (playerMaxChart) playerMaxChart.destroy();
  playerMaxChart = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ label: t('label_best_rally'), data: values, backgroundColor: '#36A2EB' }] },
    options: {
      scales: {
        x: { type: 'time', time: { unit: 'day' }, title: { display: true, text: t('date') }, ticks: { color: '#f5f5f5' } },
        y: { title: { display: true, text: t('label_damage') }, ticks: { color: '#f5f5f5', callback: v => formatDamage(v) } },
      },
      plugins: { legend: { display: false } },
    },
  });
}

function updateUIText() {
  const t = I18nService.t;
  const elements = {
    'page-title': t('title'),
    'page-subtitle': t('subtitle'),
    'language-label': t('language'),
    'nav-dashboard': t('menu_dashboard'),
    'manage-players-link': t('menu_manage_players'),
    'last-event-name-label': t('event_name'),
    'last-event-rallies-label': t('rallies'),
    'last-event-players-label': t('players'),
    'last-event-damage-label': t('alliance_total_damage'),
    'ranking-title': t('ranking'),
    'ranking-rank-header': t('rank'),
    'ranking-player-header': t('player'),
    'ranking-damage-header': t('damage'),
    'damage-chart-title': t('damage_chart'),
    'events-evolution-title': t('events_evolution'),
    'toggle-damage-label': t('toggle_damage'),
    'toggle-rallies-label': t('toggle_rallies'),
    'toggle-players-label': t('toggle_players'),
    'players-over-time-title': t('players_over_time'),
    'topN-label': t('top_n'),
    'leader-analytics-title': t('leader_analytics'),
    'leaderSelect-label': t('leader'),
    'leaderShowRallies-label': t('show_rallies'),
    'player-detail-title': t('player_detail'),
    'playerSelect-label': t('select_player'),
    'playerMaxRally-label': t('best_rally'),
    'player-detail-date-header': t('event_date'),
    'player-detail-rally-header': t('rally'),
    'player-detail-leader-header': t('rally_leader'),
    'player-detail-damage-header': t('damage'),
    'events-list-title': t('events_list'),
    'events-date-header': t('events_date'),
    'events-rallies-header': t('events_rallies'),
    'events-damage-header': t('events_damage'),
    'events-players-header': t('events_players'),
  };
  for (const [id, text] of Object.entries(elements)) {
    const el = document.getElementById(id);
    if (el && text) el.textContent = text;
  }
  const summaryDateEl = document.getElementById('summary-event-date');
  if (summaryDateEl && summaryDateEl.textContent) {
    const titleEl = document.getElementById('last-event-label');
    if (titleEl) titleEl.textContent = `${t('summary')} (${summaryDateEl.textContent})`;
  }
  const rankingTitleEl = document.getElementById('ranking-title');
  if (rankingTitleEl) {
    const currentDate = (document.getElementById('ranking-date')?.textContent) || '-';
    rankingTitleEl.innerHTML = `${t('ranking')} (<span id="ranking-date">${currentDate}</span>)`;
  }
  const damageTitleEl = document.getElementById('damage-chart-title');
  if (damageTitleEl) {
    const currentDate = (document.getElementById('damage-chart-date')?.textContent) || '-';
    damageTitleEl.innerHTML = `${t('damage_chart')} (<span id="damage-chart-date">${currentDate}</span>)`;
  }
}

async function initDashboard() {
  // Destroy existing charts
  if (damageChart) damageChart.destroy();
  if (eventsTimeseriesChart) eventsTimeseriesChart.destroy();
  if (playersOverTimeChart) playersOverTimeChart.destroy();
  if (leaderChart) leaderChart.destroy();
  if (playerMaxChart) playerMaxChart.destroy();

  await I18nService.ensurePlayersLoaded();
  await I18nService.ensureUiLoaded();
  updateUIText();

  const events = await DataService.fetchEvents();
  if (!events || events.length === 0) return;
  const last = DataService.findLastEvent(events);
  const selectedIndex = events.indexOf(last);
  DataService.setSelectedEventIndex(selectedIndex);

  updateCurrentEventInfo(last);
  updateRankingTable(last);
  renderDamageChart(last);

  const eventSeries = DataService.buildEventsSeries(events);
  renderLastEventsTable(events);
  renderEventsTimeseries(eventSeries, events);
  wireEventsTimeseriesToggles();

  const playerSeriesMap = DataService.buildPlayerSeriesMap(events);
  const topNSelect = document.getElementById('topNSelect');
  const renderTopN = () => {
    const n = parseInt(topNSelect?.value || '10', 10);
    renderPlayersOverTime(playerSeriesMap, n);
  };
  if (topNSelect) topNSelect.addEventListener('change', renderTopN);
  renderTopN();

  const leaderMaps = DataService.buildLeaderRallyMaps(events);
  const leaderSelect = document.getElementById('leaderSelect');
  if (leaderSelect) {
    const leaderEntries = Array.from(leaderMaps.leaderNames.entries()).sort((a,b)=> (a[1] > b[1] ? 1 : -1));
    leaderSelect.innerHTML = leaderEntries.map(([id,name]) => `<option value="${id}">${I18nService.playerName(id, { translit: true }) || name || id}</option>`).join('');
    const renderLeader = () => { const id = leaderSelect.value; renderLeaderAnalytics(id, leaderMaps); };
    leaderSelect.addEventListener('change', renderLeader);
    const leaderShow = document.getElementById('leaderShowRallies');
    if (leaderShow) leaderShow.addEventListener('change', renderLeader);
    if (leaderEntries.length > 0) { leaderSelect.value = leaderEntries[0][0]; renderLeader(); }
  }

  const participation = DataService.buildPlayerParticipation(events);
  const playersById = new Map();
  for (const ev of events) { for (const p of ev.players || []) { playersById.set(p.id || p.canonical_id || p.name, p.name); } }
  const playerSelect = document.getElementById('playerSelect');
  if (playerSelect) {
    const entries = Array.from(playersById.entries()).sort((a,b)=> (a[1] > b[1] ? 1 : -1));
    playerSelect.innerHTML = entries.map(([id,name]) => `<option value="${id}">${I18nService.playerName(id, { translit: true }) || name}</option>`).join('');
    const renderPlayer = () => { const id = playerSelect.value; renderPlayerMax(id, participation); };
    playerSelect.addEventListener('change', renderPlayer);
    if (entries.length > 0) { playerSelect.value = entries[0][0]; renderPlayer(); }
  }
}

async function initPlayersPage() {
  await I18nService.ensureUiLoaded();
  await I18nService.ensurePlayersLoaded();
  const t = I18nService.t;
  // Labels
  const navPlayers = document.getElementById('players-nav-link');
  const navDashboard = document.getElementById('nav-dashboard-players');
  const title = document.getElementById('players-title');
  const subtitle = document.getElementById('players-subtitle');
  const langLabel = document.getElementById('language-label-players');
  const searchInput = document.getElementById('search');
  const colId = document.getElementById('players-col-id');
  const colLang = document.getElementById('players-col-language');
  const colNames = document.getElementById('players-col-names');
  const colAliases = document.getElementById('players-col-aliases');
  const colStatus = document.getElementById('players-col-status');
  if (navDashboard) navDashboard.textContent = t('menu_dashboard');
  if (navPlayers) navPlayers.textContent = t('menu_manage_players');
  if (title) title.textContent = `${t('title')} – ${t('players')}`;
  if (subtitle) subtitle.textContent = t('players_subtitle');
  if (langLabel) langLabel.textContent = t('language') + ':';
  if (searchInput) searchInput.setAttribute('placeholder', t('players_search_placeholder'));
  if (colId) colId.textContent = t('players_col_id');
  if (colLang) colLang.textContent = t('players_col_language');
  if (colNames) colNames.textContent = t('players_col_names');
  if (colAliases) colAliases.textContent = t('players_col_aliases');
  if (colStatus) colStatus.textContent = t('players_col_status');
  document.title = `${t('title')} – ${t('players')}`;

  // Language select
  const langSel = document.getElementById('languageSelectPlayers');
  if (langSel) {
    langSel.value = I18nService.getLanguage();
    langSel.addEventListener('change', () => { I18nService.setLanguage(langSel.value); location.reload(); });
  }

  // Table rendering
  const tbody = document.querySelector('#players-table tbody');
  const store = DataService.playerTranslations || (await DataService.fetchPlayerTranslations());
  const all = Object.entries(store.players || {});
  const count = document.getElementById('count');
  function escapeHtml(s){ return (s||'').replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c])); }
  function renderRow(id, p){
    const names = p.names_by_language || {}; const aliases = p.aliases || []; const lang=p.language_detected||'';
    const namesPretty = Object.entries(names).filter(([,v])=>v).map(([k,v])=>`<span class="badge">${k}</span> ${escapeHtml(v)}`).join('<br/>');
    const aliasesPretty = aliases.map(a=>`<span class="badge">alias</span> ${escapeHtml(a)}`).join('<br/>') || '-';
    const status = p.pending_review ? '<span class="badge">pending</span>' : '<span class="badge">ok</span>';
    return `<tr><td>${escapeHtml(id)}</td><td>${escapeHtml(lang)}</td><td>${namesPretty||'-'}</td><td>${aliasesPretty}</td><td>${status}</td></tr>`;
  }
  const search = document.getElementById('search');
  function apply(){
    const q = (search?.value||'').toLowerCase();
    const rows = []; let n=0;
    for(const [id, p] of all){
      const hay = [id, (p.language_detected||''), ...(p.aliases||[]), ...Object.values(p.names_by_language||{})].join(' ').toLowerCase();
      if(!q || hay.includes(q)){ rows.push(renderRow(id,p)); n++; }
    }
    if (tbody) tbody.innerHTML = rows.join('');
    if (count) count.textContent = `${n} joueurs`;
  }
  if (search) search.addEventListener('input', apply);
  apply();
}

export const UIManager = {
  initDashboard,
  initPlayersPage,
  updateUIText,
};
