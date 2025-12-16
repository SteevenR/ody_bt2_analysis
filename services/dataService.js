// DataService: central data repository, caching, and derived series
// ES Module exporting a singleton-like object

export const DataService = (() => {
  const cache = new Map();
  let _events = null; // Array of events
  let _playerTranslations = null; // player_translations.json
  let _uiTranslations = null; // ui_translations.json
  let _selectedEventIndex = -1;

  function getDataPath(filename) {
    const pathSegments = window.location.pathname.split('/').filter(s => s);
    if (pathSegments.includes('web')) return `../data/${filename}`;
    if (pathSegments.includes('ody_bt2_analysis')) return `./data/${filename}`;
    return `../data/${filename}`;
  }

  async function fetchJson(filename) {
    const url = getDataPath(filename);
    if (cache.has(url)) return cache.get(url);
    const p = fetch(url).then(async (res) => {
      if (!res.ok) throw new Error(`Failed to fetch ${filename}: ${res.status}`);
      return res.json();
    }).catch((e) => {
      console.error(`[DataService] Error fetching ${filename}:`, e);
      return null;
    });
    cache.set(url, p);
    return p;
  }

  async function fetchEvents() {
    if (_events) return _events;
    const data = await fetchJson('beartrap.json');
    _events = (data && Array.isArray(data.events)) ? data.events : [];
    _events = buildPlayers(_events);
    return _events;
  }

  function buildPlayers(events) {
    for (let ev of events) {        
        const playersMap = new Map();
        // dictionary of player id and accumulated damage
        const playerDamageMap = new Map();

        for (const rally of ev.rallies || []) {
            for (const p of rally.participants || []) {
                const id = p.canonical_id;
                if (!playersMap.has(id)) {
                    playersMap.set(id, 
                        { 
                            id,
                            name: p.name || id,
                            total_damage: 0,
                        });
                }
                playerDamageMap.set(id, (playerDamageMap.get(id) || 0) + (p.damage || 0));
            }
        }
        // Set total damage for each player
        for (const [id, damage] of playerDamageMap.entries()) {
            const player = playersMap.get(id);
            if (player) {
                player.total_damage = damage;
            }
        }
        // Convert playersMap to array and assign to event
        ev.players = Array.from(playersMap.values());
    }

    return events;
  }

  async function fetchPlayerTranslations() {
    if (_playerTranslations) return _playerTranslations;
    const data = await fetchJson('player_translations.json');
    _playerTranslations = data || { players: {}, languages: {} };
    return _playerTranslations;
  }

  async function fetchUiTranslations() {
    if (_uiTranslations) return _uiTranslations;
    const data = await fetchJson('ui_translations.json');
    _uiTranslations = data || {};
    return _uiTranslations;
  }

  function findLastEvent(events) {
    if (!events || events.length === 0) return null;
    const now = new Date();
    let closestEvent = null;
    let minDiff = Infinity;
    for (const event of events) {
      const eventDate = new Date(event.date || event.id);
      if (isNaN(eventDate.getTime())) continue;
      const diff = now - eventDate;
      if (diff >= 0 && diff < minDiff) {
        minDiff = diff;
        closestEvent = event;
      }
    }
    if (!closestEvent) return events[events.length - 1];
    return closestEvent;
  }

  function parseDate(str) {
    const d = new Date(str + 'T00:00:00');
    return isNaN(d.getTime()) ? new Date(Date.parse(str)) : d;
  }

  // ===== Derived series builders =====
  function buildEventsSeries(events) {
    const sorted = events.slice().sort((a, b) => (a.date < b.date ? -1 : 1));
    return sorted.map((ev) => {
      let playerCount = 0;
      if (ev.players && ev.players.length > 0) {
        playerCount = ev.players.length;
      } else if (ev.rallies && ev.rallies.length > 0) {
        const playerIds = new Set();
        for (const rally of ev.rallies) {
          for (const p of rally.participants || []) {
            playerIds.add(p.canonical_id || p.id || p.name);
          }
        }
        playerCount = playerIds.size;
      }
      return {
        date: parseDate(ev.date),
        dateStr: ev.date,
        totalDamage: ev.alliance_total_damage || 0,
        rallies: ev.rally_count_total || (ev.rallies ? ev.rallies.length : 0),
        playersCount: playerCount,
      };
    });
  }

  function buildPlayerSeriesMap(events) {
    const map = new Map();
    const sorted = events.slice().sort((a, b) => (a.date < b.date ? -1 : 1));
    for (const ev of sorted) {
      const date = parseDate(ev.date);
      const players = ev.players || [];
      for (const p of players) {
        const id = p.id || p.canonical_id || p.name;
        if (!map.has(id)) {
          map.set(id, { name: p.name, series: [], totalAllTime: 0 });
        }
        const entry = map.get(id);
        entry.series.push({ x: date, y: p.total_damage || 0 });
        entry.totalAllTime += p.total_damage || 0;
        entry.name = p.name || entry.name;
      }
    }
    return map;
  }

  function buildLeaderRallyMaps(events) {
    const perEventSum = new Map();
    const perRally = new Map();
    const leaderNames = new Map();
    for (const ev of events) {
      const date = parseDate(ev.date);
      const nameToId = new Map((ev.players || []).map(p => [p.name, p.id]));
      for (const rally of ev.rallies || []) {
        const total = (rally.participants || []).reduce((s, p) => s + (p.damage || 0), 0);
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
        if (!perRally.has(leaderId)) perRally.set(leaderId, []);
        perRally.get(leaderId).push({ x: date, y: total, label: rally.id });
        if (!perEventSum.has(leaderId)) perEventSum.set(leaderId, []);
        const arr = perEventSum.get(leaderId);
        let pt = arr.find(pt => pt.x.getTime() === date.getTime());
        if (!pt) { pt = { x: date, y: 0 }; arr.push(pt); }
        pt.y += total;
      }
    }
    for (const arr of perEventSum.values()) arr.sort((a, b) => a.x - b.x);
    for (const arr of perRally.values()) arr.sort((a, b) => a.x - b.x);
    return { perEventSum, perRally, leaderNames };
  }

  function buildPlayerParticipation(events) {
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
    for (const arr of map.values()) arr.sort((a, b) => a.date - b.date);
    return map;
  }

  function setSelectedEventIndex(idx) { _selectedEventIndex = idx; }
  function getSelectedEventIndex() { return _selectedEventIndex; }

  return {
    getDataPath,
    fetchEvents,
    fetchPlayerTranslations,
    fetchUiTranslations,
    findLastEvent,
    buildEventsSeries,
    buildPlayerSeriesMap,
    buildLeaderRallyMaps,
    buildPlayerParticipation,
    setSelectedEventIndex,
    getSelectedEventIndex,
    // expose caches for read-only needs
    get events() { return _events; },
    get playerTranslations() { return _playerTranslations; },
    get uiTranslations() { return _uiTranslations; },
  };
})();
