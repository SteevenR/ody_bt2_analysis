// Determine base path for data files
function getDataPath(filename) {
  // Check if we're in web/ subdirectory (localhost) or root (GitHub Pages)
  const pathSegments = window.location.pathname.split('/').filter(s => s);
  
  // If path contains 'web', we're at localhost or similar
  if (pathSegments.includes('web')) {
    return `../data/${filename}`;
  }
  
  // If path contains 'ody_bt2_analysis', we're on GitHub Pages
  if (pathSegments.includes('ody_bt2_analysis')) {
    return `./data/${filename}`;
  }
  
  // Fallback: try both
  return `../data/${filename}`;
}

async function loadData() {
  try {
    const response = await fetch(getDataPath("beartrap.json"));
    const data = await response.json();
    return data;
  } catch (err) {
    console.error("Erreur de chargement des données:", err);
    return { events: [] };
  }
}

async function loadTranslations() {
  try {
    const response = await fetch(getDataPath("player_translations.json"));
    const data = await response.json();
    return data;
  } catch (err) {
    console.error("Erreur de chargement des traductions:", err);
    return { players: {}, languages: {} };
  }
}

async function loadUITranslations() {
  try {
    const response = await fetch(getDataPath("ui_translations.json"));
    const data = await response.json();
    return data;
  } catch (err) {
    console.error("Erreur de chargement des traductions UI:", err);
    return {};
  }
}

// Variables globales
let currentLanguage = "en";
let translationsData = null;
let uiTranslations = null;

// Safe localStorage wrapper for Firefox privacy mode
function getSavedLanguage() {
  try {
    return localStorage.getItem("selectedLanguage") || "en";
  } catch (e) {
    // Firefox private mode blocks localStorage
    return "en";
  }
}

function saveLanguage(lang) {
  try {
    localStorage.setItem("selectedLanguage", lang);
  } catch (e) {
    // Firefox private mode blocks localStorage - ignore silently
  }
}

currentLanguage = getSavedLanguage();

// Track all charts for proper cleanup
let damageChart = null;
let eventsTimeseriesChart = null;
let playersOverTimeChart = null;
let leaderChart = null;
let playerMaxChart = null;

currentLanguage = getSavedLanguage();

function t(key) {
  if (!uiTranslations || !uiTranslations[currentLanguage]) {
    return key;
  }
  return uiTranslations[currentLanguage][key] || key;
}

function getPlayerNameInLanguage(playerId, language) {
  if (!translationsData || !translationsData.players) return playerId;
  const player = translationsData.players[playerId];
  if (!player) return playerId;
  
  const names = player.names_by_language || {};
  
  // Essayer la langue demandée
  if (language in names && names[language]) {
    const name = names[language];
    const trans = (player.transliterations_by_language || {})[language];
    if (trans && trans !== name) {
      return `${name} (${trans})`;
    }
    return name;
  }
  
  // Fallback sur la langue détectée
  const detected = player.language_detected || "en";
  if (detected in names && names[detected]) {
    return names[detected];
  }
  
  // Fallback sur n'importe quel nom non-vide
  for (const [lang, name] of Object.entries(names)) {
    if (name) return name;
  }
  
  return playerId;
}

function formatDamage(value) {
  if (value >= 1_000_000_000) {
    return (value / 1_000_000_000).toFixed(2) + "B";
  } else if (value >= 1_000_000) {
    return (value / 1_000_000).toFixed(2) + "M";
  } else if (value >= 1_000) {
    return (value / 1_000).toFixed(2) + "K";
  }
  return value.toString();
}

function parseDate(str) {
  // Expecting YYYY-MM-DD; fallback to Date.parse
  const d = new Date(str + 'T00:00:00');
  return isNaN(d.getTime()) ? new Date(Date.parse(str)) : d;
}

function colorPalette(i) {
  const colors = [
    '#4BC0C0', '#FF6384', '#36A2EB', '#FFCE56', '#9966FF',
    '#FF9F40', '#66BB6A', '#EF5350', '#29B6F6', '#AB47BC',
    '#D4E157', '#FF7043', '#EC407A', '#26A69A', '#8D6E63'
  ];
  return colors[i % colors.length];
}

function updateLastEventInfo(event) {
  document.getElementById("last-event-name").textContent = event.name;
  document.getElementById("last-event-date").textContent = event.date;
  document.getElementById("last-event-players-count").textContent =
    event.players.length;

  const totalDamage = event.players.reduce(
    (sum, p) => sum + (p.total_damage || 0),
    0
  );
  document.getElementById("last-event-total-damage").textContent =
    formatDamage(totalDamage);
}

function updateRankingTable(event) {
  const tbody = document.querySelector("#ranking-table tbody");
  tbody.innerHTML = "";

  event.players
    .slice()
    .sort((a, b) => b.total_damage - a.total_damage)
    .forEach((player, index) => {
      const tr = document.createElement("tr");

      const rankTd = document.createElement("td");
      rankTd.textContent = index + 1;

      const nameTd = document.createElement("td");
      const displayName = getPlayerNameInLanguage(player.id, currentLanguage);
      nameTd.textContent = displayName;

      const dmgTd = document.createElement("td");
      dmgTd.textContent = formatDamage(player.total_damage);

      tr.appendChild(rankTd);
      tr.appendChild(nameTd);
      tr.appendChild(dmgTd);

      tbody.appendChild(tr);
    });
}

function renderDamageChart(event) {
  const ctx = document.getElementById("damage-chart").getContext("2d");

  const sortedPlayers = event.players
    .slice()
    .sort((a, b) => b.total_damage - a.total_damage)
    .slice(0, 15); // top 15

  const labels = sortedPlayers.map((p) => p.name);
  const values = sortedPlayers.map((p) => p.total_damage);

  if (damageChart) damageChart.destroy();

  damageChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: t("label_damage"),
          data: values,
          backgroundColor: "rgba(75, 192, 192, 0.7)",
        },
      ],
    },
    options: {
      plugins: {
        legend: { display: false },
      },
      scales: {
        x: {
          title: { display: true, text: t('player') },
          ticks: {
            color: "#f5f5f5",
          },
        },
        y: {
          title: { display: true, text: t('label_damage') },
          ticks: {
            color: "#f5f5f5",
            callback: function (value) {
              return formatDamage(value);
            },
          },
        },
      },
    },
  });
}

// ===== Multi-event: Events time series =====
function buildEventsSeries(events) {
  const sorted = events.slice().sort((a, b) => (a.date < b.date ? -1 : 1));
  return sorted.map((ev) => ({
    date: parseDate(ev.date),
    dateStr: ev.date,
    totalDamage: ev.alliance_total_damage || 0,
    rallies: ev.rally_count_total || (ev.rallies ? ev.rallies.length : 0),
    playersCount: (ev.players && ev.players.length) || 0,
  }));
}

function renderEventsTimeseries(series) {
  const ctx = document.getElementById('events-timeseries').getContext('2d');
  // Use XY points to avoid 0..1 auto-scale issues on time axis
  const damagePoints = series.map(s => ({ x: s.date, y: s.totalDamage }));
  const ralliesPoints = series.map(s => ({ x: s.date, y: s.rallies }));
  const playersPoints = series.map(s => ({ x: s.date, y: s.playersCount }));

  if (eventsTimeseriesChart) {
    eventsTimeseriesChart.destroy();
  }

  eventsTimeseriesChart = new Chart(ctx, {
    data: {
      datasets: [
        {
          label: t('label_total_damage'),
          data: damagePoints,
          borderColor: '#4BC0C0',
          backgroundColor: 'rgba(75,192,192,0.2)',
          yAxisID: 'yDamage',
          tension: 0.2,
          type: 'line',
        },
        {
          label: t('label_rallies'),
          data: ralliesPoints,
          borderColor: '#FF6384',
          backgroundColor: 'rgba(255,99,132,0.2)',
          yAxisID: 'yCount',
          tension: 0.2,
          type: 'line',
        },
        {
          label: t('label_players_present'),
          data: playersPoints,
          borderColor: '#36A2EB',
          backgroundColor: 'rgba(54,162,235,0.2)',
          yAxisID: 'yCount',
          tension: 0.2,
          type: 'line',
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        x: {
          type: 'time',
          time: { unit: 'day' },
          title: { display: true, text: t('date') },
          ticks: { color: '#f5f5f5' },
        },
        yDamage: {
          type: 'linear',
          position: 'left',
          ticks: { color: '#f5f5f5', callback: (v) => formatDamage(v) },
          title: { display: true, text: t('label_total_damage') },
        },
        yCount: {
          type: 'linear',
          position: 'right',
          grid: { drawOnChartArea: false },
          ticks: { color: '#f5f5f5' },
          title: { display: true, text: 'Count' },
        },
      },
      plugins: { legend: { position: 'top' } },
    },
  });
}

function wireEventsTimeseriesToggles() {
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
      if (ds) {
        ds.hidden = !el.checked;
        eventsTimeseriesChart.update();
      }
    });
  });
}

// ===== Players over time (stacked Top N) =====
function buildPlayerSeriesMap(events) {
  // Map playerId -> { name, series: [{x: date, y: value}], totalAllTime }
  const map = new Map();
  const sorted = events.slice().sort((a, b) => (a.date < b.date ? -1 : 1));
  for (const ev of sorted) {
    const date = parseDate(ev.date);
    for (const p of ev.players || []) {
      const id = p.id || p.canonical_id || p.name;
      if (!map.has(id)) {
        map.set(id, { name: p.name, series: [], totalAllTime: 0 });
      }
      const entry = map.get(id);
      entry.series.push({ x: date, y: p.total_damage || 0 });
      entry.totalAllTime += p.total_damage || 0;
      // prefer most recent name
      entry.name = p.name || entry.name;
    }
  }
  return map;
}

function renderPlayersOverTime(playerSeriesMap, topN) {
  const ctx = document.getElementById('players-over-time').getContext('2d');

  // pick top N by totalAllTime
  const entries = Array.from(playerSeriesMap.entries());
  entries.sort((a, b) => b[1].totalAllTime - a[1].totalAllTime);
  const top = entries.slice(0, topN);

  // Build line datasets for top N players
  const datasets = top.map(([id, meta], idx) => {
    const color = colorPalette(idx);
    const displayName = getPlayerNameInLanguage(id, currentLanguage);
    return {
      label: displayName,
      data: meta.series,
      borderColor: color,
      backgroundColor: color + 'CC',
      tension: 0.2,
      fill: false,
      borderWidth: 3,
      pointRadius: 6,
      pointBorderWidth: 2,
      pointBorderColor: '#fff',
      pointBackgroundColor: color,
      pointHoverRadius: 8,
    };
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
        y: { title: { display: true, text: t('label_damage') }, ticks: { color: '#f5f5f5', callback: (v) => formatDamage(v) } },
      },
      plugins: { legend: { position: 'top' } },
    },
  });
}

// ===== Leader analytics =====
function buildLeaderRallyMaps(events) {
  // Returns { perEventSum: Map<leaderId, Array<{x:Date,y:number}>>, perRally: Map<leaderId, Array<{x:Date,y:number,label:string}>>, leaderNames: Map<leaderId,name> }
  const perEventSum = new Map();
  const perRally = new Map();
  const leaderNames = new Map();

  for (const ev of events) {
    const date = parseDate(ev.date);
    const nameToId = new Map((ev.players || []).map(p => [p.name, p.id]));
    for (const rally of ev.rallies || []) {
      const total = (rally.participants || []).reduce((s, p) => s + (p.damage || 0), 0);
      // find leader id from participants.is_leader; fallback name->id
      let leaderId = null;
      let leaderName = rally.leader || '';
      const lp = (rally.participants || []).find(p => p.is_leader);
      if (lp && (lp.canonical_id || lp.id)) {
        leaderId = lp.canonical_id || lp.id;
        leaderName = lp.name || leaderName;
      } else if (leaderName && nameToId.has(leaderName)) {
        leaderId = nameToId.get(leaderName);
      }
      if (!leaderId) continue;
      leaderNames.set(leaderId, leaderName || leaderNames.get(leaderId) || '');

      // per rally points
      if (!perRally.has(leaderId)) perRally.set(leaderId, []);
      perRally.get(leaderId).push({ x: date, y: total, label: rally.id });

      // per event sum
      if (!perEventSum.has(leaderId)) perEventSum.set(leaderId, []);
      const arr = perEventSum.get(leaderId);
      let pt = arr.find(pt => pt.x.getTime() === date.getTime());
      if (!pt) { pt = { x: date, y: 0 }; arr.push(pt); }
      pt.y += total;
    }
  }

  // sort by date
  for (const arr of perEventSum.values()) arr.sort((a, b) => a.x - b.x);
  for (const arr of perRally.values()) arr.sort((a, b) => a.x - b.x);
  return { perEventSum, perRally, leaderNames };
}

function renderLeaderAnalytics(leaderId, maps) {
  const ctx = document.getElementById('leader-rallies-over-time').getContext('2d');
  const lineSeries = maps.perEventSum.get(leaderId) || [];
  console.log('[DEBUG renderLeaderAnalytics] leaderId:', leaderId, 'lineSeries:', lineSeries);
  const showRallies = document.getElementById('leaderShowRallies')?.checked;

  if (leaderChart) leaderChart.destroy();
  leaderChart = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [
        {
          label: t('label_sum_per_event'),
          data: lineSeries,
          borderColor: '#4BC0C0',
          backgroundColor: 'rgba(75,192,192,0.3)',
          parsing: false,
          tension: 0.2,
          borderWidth: 3,
          pointRadius: 6,
          pointBorderWidth: 2,
          pointBorderColor: '#fff',
          pointBackgroundColor: '#4BC0C0',
          pointHoverRadius: 8,
        },
        ...(showRallies ? [{
          type: 'scatter',
          label: t('label_per_rally'),
          data: maps.perRally.get(leaderId) || [],
          parsing: false,
          pointRadius: 4,
          backgroundColor: '#FF6384',
        }] : []),
      ],
    },
    options: {
      responsive: true,
      scales: {
        x: { type: 'time', time: { unit: 'day' }, title: { display: true, text: t('date') }, ticks: { color: '#f5f5f5' } },
        y: { title: { display: true, text: t('label_damage') }, ticks: { color: '#f5f5f5', callback: (v) => formatDamage(v) } },
      },
      plugins: { legend: { position: 'top' } },
    },
  });
}

// ===== Player detail: max per event =====
function buildPlayerParticipation(events) {
  // Returns Map<playerId, Array<{date, eventId, rallyId, leaderName, value}>>
  const map = new Map();
  for (const ev of events) {
    const date = parseDate(ev.date);
    for (const rally of ev.rallies || []) {
      const leaderName = rally.leader || '';
      for (const p of rally.participants || []) {
        const id = p.canonical_id || p.id || p.name;
        if (!map.has(id)) map.set(id, []);
        map.get(id).push({ date, eventId: ev.id || ev.date, rallyId: rally.id, leaderName, value: p.damage || 0 });
      }
    }
  }
  // sort
  for (const arr of map.values()) arr.sort((a, b) => a.date - b.date);
  return map;
}

function renderPlayerMax(playerId, participation, playerNamesById) {
  const ctx = document.getElementById('player-max-rally').getContext('2d');
  const rows = participation.get(playerId) || [];
  // per event pick max
  const byEvent = new Map();
  for (const row of rows) {
    const key = row.eventId;
    if (!byEvent.has(key) || row.value > byEvent.get(key).value) {
      byEvent.set(key, row);
    }
  }
  const maxRows = Array.from(byEvent.values()).sort((a, b) => a.date - b.date);

  // table
  const tbody = document.querySelector('#player-max-table tbody');
  if (tbody) {
    tbody.innerHTML = '';
    for (const r of maxRows) {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${r.date.toISOString().slice(0,10)}</td><td>${r.rallyId}</td><td>${r.leaderName || ''}</td><td>${formatDamage(r.value)}</td>`;
      tbody.appendChild(tr);
    }
  }

  // chart
  const labels = maxRows.map(r => r.date);
  const values = maxRows.map(r => r.value);
  if (playerMaxChart) playerMaxChart.destroy();
  playerMaxChart = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ label: t('label_best_rally'), data: values, backgroundColor: '#36A2EB' }] },
    options: {
      scales: {
        x: { type: 'time', time: { unit: 'day' }, title: { display: true, text: t('date') }, ticks: { color: '#f5f5f5' } },
        y: { title: { display: true, text: t('label_damage') }, ticks: { color: '#f5f5f5', callback: (v) => formatDamage(v) } },
      },
      plugins: { legend: { display: false } },
    },
  });
}

async function init() {
  // Destroy all existing charts
  if (damageChart) damageChart.destroy();
  if (eventsTimeseriesChart) eventsTimeseriesChart.destroy();
  if (playersOverTimeChart) playersOverTimeChart.destroy();
  if (leaderChart) leaderChart.destroy();
  if (playerMaxChart) playerMaxChart.destroy();

  const data = await loadData();

  if (!data.events || data.events.length === 0) {
    console.warn("Aucun événement Bear Trap trouvé.");
    return;
  }

  const events = data.events;
  const lastEvent = events[events.length - 1];

  updateLastEventInfo(lastEvent);
  updateRankingTable(lastEvent);
  renderDamageChart(lastEvent);

  // Multi-event series
  const eventSeries = buildEventsSeries(events);
  renderEventsTimeseries(eventSeries);
  wireEventsTimeseriesToggles();

  // Players over time (Top N)
  const playerSeriesMap = buildPlayerSeriesMap(events);
  const topNSelect = document.getElementById('topNSelect');
  const renderTopN = () => {
    const n = parseInt(topNSelect?.value || '10', 10);
    renderPlayersOverTime(playerSeriesMap, n);
  };
  if (topNSelect) topNSelect.addEventListener('change', renderTopN);
  renderTopN();

  // Leader analytics
  const leaderMaps = buildLeaderRallyMaps(events);
  console.log('[DEBUG] leaderMaps.leaderNames:', leaderMaps.leaderNames);
  console.log('[DEBUG] leaderMaps.perEventSum:', leaderMaps.perEventSum);
  // Build list of leaders
  const leaderSelect = document.getElementById('leaderSelect');
  if (leaderSelect) {
    const leaderEntries = Array.from(leaderMaps.leaderNames.entries()).sort((a, b) => (a[1] > b[1] ? 1 : -1));
    console.log('[DEBUG] leaderEntries:', leaderEntries);
    leaderSelect.innerHTML = leaderEntries.map(([id, name]) => {
      const displayName = getPlayerNameInLanguage(id, currentLanguage) || name || id;
      return `<option value="${id}">${displayName}</option>`;
    }).join('');
    const renderLeader = () => {
      const id = leaderSelect.value;
      console.log('[DEBUG] Rendering leader:', id);
      renderLeaderAnalytics(id, leaderMaps);
    };
    leaderSelect.addEventListener('change', renderLeader);
    const leaderShow = document.getElementById('leaderShowRallies');
    if (leaderShow) leaderShow.addEventListener('change', renderLeader);
    if (leaderEntries.length > 0) {
      leaderSelect.value = leaderEntries[0][0];
      renderLeader();
    }
  }

  // Player detail
  const participation = buildPlayerParticipation(events);
  // Player list from all events
  const playersById = new Map();
  for (const ev of events) {
    for (const p of ev.players || []) {
      playersById.set(p.id || p.canonical_id || p.name, p.name);
    }
  }
  const playerSelect = document.getElementById('playerSelect');
  if (playerSelect) {
    const entries = Array.from(playersById.entries()).sort((a, b) => (a[1] > b[1] ? 1 : -1));
    playerSelect.innerHTML = entries.map(([id, name]) => {
      const displayName = getPlayerNameInLanguage(id, currentLanguage) || name;
      return `<option value="${id}">${displayName}</option>`;
    }).join('');
    const renderPlayer = () => {
      const id = playerSelect.value;
      renderPlayerMax(id, participation, playersById);
    };
    playerSelect.addEventListener('change', renderPlayer);
    if (entries.length > 0) {
      playerSelect.value = entries[0][0];
      renderPlayer();
    }
  }
}

function updateUIText() {
  // Mettre à jour tous les éléments texte UI basés sur les clés de traduction
  const elements = {
    // Titres et en-têtes
    "page-title": t("title"),
    "page-subtitle": t("subtitle"),
    "language-label": t("language"),
    "nav-dashboard": t("menu_dashboard"),
    "manage-players-link": t("menu_manage_players"),
    
    // Dernière section d'événement
    "last-event-label": t("last_event"),
    "last-event-name-label": t("event_name"),
    "last-event-date-label": t("date"),
    "last-event-players-label": t("players"),
    "last-event-damage-label": t("total_damage"),
    
    // Tableau de classement
    "ranking-title": t("ranking"),
    "ranking-rank-header": t("rank"),
    "ranking-player-header": t("player"),
    "ranking-damage-header": t("damage"),
    
    // Graphiques
    "damage-chart-title": t("damage_chart"),
    "events-evolution-title": t("events_evolution"),
    "toggle-damage-label": t("toggle_damage"),
    "toggle-rallies-label": t("toggle_rallies"),
    "toggle-players-label": t("toggle_players"),
    
    // Players over time
    "players-over-time-title": t("players_over_time"),
    "topN-label": t("top_n"),
    
    // Leader analytics
    "leader-analytics-title": t("leader_analytics"),
    "leaderSelect-label": t("leader"),
    "leaderShowRallies-label": t("show_rallies"),
    
    // Player detail
    "player-detail-title": t("player_detail"),
    "playerSelect-label": t("select_player"),
    "playerMaxRally-label": t("best_rally"),
    "player-detail-date-header": t("event_date"),
    "player-detail-rally-header": t("rally"),
    "player-detail-leader-header": t("rally_leader"),
    "player-detail-damage-header": t("damage"),
  };
  
  for (const [id, text] of Object.entries(elements)) {
    const el = document.getElementById(id);
    if (el && text) {
      el.textContent = text;
    }
  }

  document.title = `${t("title")} – ${t("menu_dashboard")}`;
}

document.addEventListener("DOMContentLoaded", async () => {
  // Charger les traductions d'abord
  translationsData = await loadTranslations();
  uiTranslations = await loadUITranslations();
  
  // Initialiser le sélecteur de langue
  const languageSelect = document.getElementById("languageSelect");
  if (languageSelect) {
    languageSelect.value = currentLanguage;
    languageSelect.addEventListener("change", (e) => {
      currentLanguage = e.target.value;
      saveLanguage(currentLanguage);
      // Mettre à jour les traductions UI
      updateUIText();
      // Recharger et redessiner l'interface
      init();
    });
  }
  
  // Mettre à jour les traductions UI dès le démarrage
  updateUIText();
  
  // Lancer l'initialisation
  init();
});