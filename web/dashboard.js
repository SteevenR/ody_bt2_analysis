// Entry point for dashboard page (index.html)
import { DataService } from './services/dataService.js';
import { I18nService } from './services/i18nService.js';
import { UIManager } from './managers/uiManager.js';

document.addEventListener('DOMContentLoaded', async () => {
  // Load translations first
  const [ui, pt] = await Promise.all([
    DataService.fetchUiTranslations(),
    DataService.fetchPlayerTranslations(),
  ]);
  I18nService.setUiTranslations(ui);
  I18nService.setPlayerTranslations(pt);

  // Wire language selector
  const langSelect = document.getElementById('languageSelect');
  if (langSelect) {
    langSelect.value = I18nService.getLanguage();
    langSelect.addEventListener('change', async (e) => {
      I18nService.setLanguage(e.target.value);
      UIManager.updateUIText();
      // Re-init dashboard to re-render labels and datasets
      await UIManager.initDashboard();
    });
  }

  UIManager.updateUIText();
  await UIManager.initDashboard();
});
