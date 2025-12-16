// I18nService: language management, UI labels, player display names
import { DataService } from '../services/dataService.js';

export const I18nService = (() => {
  let _lang = 'en';
  let _ui = {};
  let _players = { players: {}, languages: {} };
  const _listeners = new Set();

  function getSavedLanguage() {
    try { return localStorage.getItem('selectedLanguage') || 'en'; } catch { return 'en'; }
  }
  function saveLanguage(lang) {
    try { localStorage.setItem('selectedLanguage', lang); } catch {}
  }

  _lang = getSavedLanguage();

  function setLanguage(lang) {
    _lang = lang || 'en';
    saveLanguage(_lang);
    for (const cb of _listeners) {
      try { cb(_lang); } catch {}
    }
  }
  function getLanguage() { return _lang; }
  function onLanguageChange(cb) { _listeners.add(cb); return () => _listeners.delete(cb); }

  function setUiTranslations(map) { _ui = map || {}; }
  function setPlayerTranslations(map) { _players = map || { players: {}, languages: {} }; }

  function t(key) {
    const table = _ui[_lang] || {};
    return table[key] || key;
  }

  function playerName(playerId, opts = {}) {
    const translit = opts.translit !== false; // default true
    const store = _players.players || {};
    const player = store[playerId];
    if (!player) return playerId;
    const names = player.names_by_language || {};
    if (_lang in names && names[_lang]) {
      const name = names[_lang];
      if (translit) {
        const trans = (player.transliterations_by_language || {})[_lang];
        if (trans && trans !== name) return `${name} (${trans})`;
      }
      return name;
    }
    const detected = player.language_detected || 'en';
    if (detected in names && names[detected]) return names[detected];
    for (const v of Object.values(names)) { if (v) return v; }
    return playerId;
  }

  async function ensureUiLoaded() {
    if (!DataService.uiTranslations) {
      const ui = await DataService.fetchUiTranslations();
      setUiTranslations(ui);
    }
  }
  async function ensurePlayersLoaded() {
    if (!DataService.playerTranslations) {
      const pt = await DataService.fetchPlayerTranslations();
      setPlayerTranslations(pt);
    }
  }

  return {
    setLanguage,
    getLanguage,
    onLanguageChange,
    setUiTranslations,
    setPlayerTranslations,
    t,
    playerName,
    ensureUiLoaded,
    ensurePlayersLoaded,
  };
})();
