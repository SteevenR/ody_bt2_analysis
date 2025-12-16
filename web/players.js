// Entry point for players page (players.html)
import { DataService } from './services/dataService.js';
import { I18nService } from './services/i18nService.js';
import { UIManager } from './managers/uiManager.js';

document.addEventListener('DOMContentLoaded', async () => {
  const [ui, pt] = await Promise.all([
    DataService.fetchUiTranslations(),
    DataService.fetchPlayerTranslations(),
  ]);
  I18nService.setUiTranslations(ui);
  I18nService.setPlayerTranslations(pt);
  await UIManager.initPlayersPage();
});
