/*
 * SillyTavern Advanced Memory & RLHF Plugin
 * Client-side only - communicates with Python backend via HTTP
 * Backend must be started separately via: py backend/start_backend.py
 */
(function () {
    'use strict';

    var PLUGIN_NAME = 'AdvancedMemoryRLHF';
    var BACKEND_URL = 'http://127.0.0.1:5125';

    var feedbackMode = false;
    var autoStoreEnabled = true;
    var ragInjectionEnabled = true;
    var pluginInitialized = false;
    var SETTINGS_KEY = 'amp_plugin_settings';
    var recentlyStored = {};
    var backendAvailable = false;

    // ===================== I18N SYSTEM =====================

    var i18n = {};
    var i18nFallback = {};
    var currentLanguage = 'en';
    var detectedLanguage = 'en';
    var languageOverride = '';
    var availableLanguages = [];
    var i18nPath = '';
    var i18nStats = { total: 0, translated: 0, missing: [] };

    // Hardcoded English fallback so UI is NEVER blank
    var BUILTIN_FALLBACK = {
        plugin_title: '🧠 Advanced Memory & RLHF',
        status_title: 'Status',
        status_backend: 'Backend',
        status_loading: '⏳ Loading...',
        status_connected: '🟢 Connected',
        status_disconnected: '🔴 Disconnected',
        status_device_detecting: 'Device: detecting...',
        status_vectordb: 'Vector DB: 0 MB',
        status_lora_none: 'LoRA: none',
        memory_title: '💾 Memory',
        memory_autostore_on: '🔄 Auto-Store ON',
        memory_autostore_off: '⏸ Auto-Store OFF',
        memory_rag_on: '📡 RAG ON',
        memory_rag_off: '📡 RAG OFF',
        memory_rag_max_chars: 'Max chars:',
        memory_rag_min_score: 'Min score:',
        memory_store_chat: '💾 Store Chat',
        memory_query: '🔍 Query',
        memory_query_placeholder: 'Search memories...',
        memory_query_search: '🔎 Search',
        memory_no_data: 'No chat data to store',
        memory_stored_success: '✅ Chat stored in memory',
        memory_store_failed: 'Store failed:',
        memory_no_results: 'No results found',
        memory_query_failed: 'Query failed:',
        browser_title: '📂 Memory Browser',
        browser_all_characters: '-- All Characters --',
        browser_browse: '📋 Browse',
        browser_history: '📜 History',
        browser_prev: '◀ Prev',
        browser_next: 'Next ▶',
        browser_page: 'Page',
        browser_delete_selected: '🗑 Delete Selected',
        browser_clear_character: '🗑 Clear Character',
        browser_clear_all: '⚠ Clear ALL Memory',
        browser_no_documents: 'No documents found',
        browser_select_all: 'Select all',
        browser_total: 'Total:',
        browser_documents: 'documents',
        browser_tag_auto: '[auto]',
        browser_tag_manual: '[manual]',
        browser_no_selected: 'No documents selected',
        browser_confirm_delete_selected: 'Delete {count} selected documents?',
        browser_deleted_success: 'Deleted {count} documents',
        browser_delete_failed: 'Delete failed:',
        browser_select_character_first: 'Select a character first',
        browser_confirm_delete_character: 'Delete all memories for {character}?',
        browser_deleted_character: 'Deleted {count} memories for {character}',
        browser_confirm_clear_all: '⚠ Delete ALL memory data? This cannot be undone!',
        browser_confirm_clear_all_2: 'Are you REALLY sure? ALL data will be lost!',
        browser_clear_failed: 'Clear failed:',
        browser_browse_failed: 'Browse failed:',
        browser_history_failed: 'History failed:',
        browser_no_history: 'No history entries',
        browser_recent_history: 'Recent History',
        browser_entries: 'entries',
        documents_title: '📄 Documents',
        documents_upload: '📁 Upload Files',
        documents_uploading: 'Uploading {filename} ({current}/{total})...',
        documents_upload_failed: 'Upload failed for {filename}:',
        documents_upload_complete: '✅ All uploads complete',
        rlhf_title: '⭐ RLHF Feedback',
        rlhf_toggle: '👍 Enable Feedback',
        rlhf_toggle_on: '✅ Feedback ON',
        rlhf_stats: '👍 {positive} | 👎 {negative} | ⭐ {excellent} | Total: {total}',
        rlhf_hint: 'When enabled, feedback buttons appear on AI messages.',
        rlhf_btn_good: '👍',
        rlhf_btn_bad: '👎',
        rlhf_btn_excellent: '⭐',
        rlhf_tooltip_good: 'Good response',
        rlhf_tooltip_bad: 'Bad response',
        rlhf_tooltip_excellent: 'Excellent response',
        rlhf_feedback_recorded: 'Recorded!',
        rlhf_feedback_failed: 'Feedback failed',
        training_base_model: 'Base Model:',
        training_model_placeholder: 'Path to local model (auto-detected from ST)...',
        training_model_hint: 'Local model path required for LoRA training. Use "Detect Model" or enter manually.',
        training_model_detecting: 'Model: click "Detect Model" to read from SillyTavern',
        training_model_detected: 'Model detected from SillyTavern',
        training_model_not_detected: 'Could not auto-detect model. Please enter path manually.',
        training_model_manual_hint: 'Enter the HuggingFace model ID or local path to your base model weights.',
        training_model_source: 'Source',
        training_model_required: 'Base model path is required for LoRA training!',
        training_model_label: 'Base Model',
        training_detect_model: 'Detect Model',
        training_cloud_warning: 'This looks like a cloud API model. LoRA training requires local model weights on disk. Continue anyway?',
        training_title: '🎓 Training',
        training_epochs: 'Epochs:',
        training_learning_rate: 'Learning Rate:',
        training_lora_rank: 'LoRA Rank:',
        training_batch_size: 'Batch Size:',
        training_start: '🚀 Start Training',
        training_cancel: '⏹ Cancel',
        training_confirm: 'Start LoRA training with current settings?',
        training_start_failed: 'Training failed:',
        training_cancelled: 'Training cancelled',
        training_cancel_failed: 'Cancel failed:',
        training_preparing: 'Preparing...',
        training_eta: 'ETA: {eta}',
        training_status: 'Epoch {current_epoch}/{total_epochs} | Step {current_step}/{total_steps} | Loss: {loss}',
        training_completed: '✅ Training completed!',
        training_failed: '❌ Training failed:',
        export_title: '📤 Export & Summary',
        export_summary: '📝 Generate Summary',
        export_format_txt: 'Plain Text (.txt)',
        export_format_json: 'JSON (.json)',
        export_format_pdf: 'PDF (.pdf)',
        export_format_xml: 'XML (.xml)',
        export_format_docx: 'Word (.docx)',
        export_format_odt: 'OpenDocument (.odt)',
        export_format_html: 'HTML (.html)',
        export_chat: '📥 Export Chat',
        export_print: '🖨 Print',
        export_success: '✅ Export ready!',
        export_failed: 'Export failed:',
        export_summary_failed: 'Summary failed:',
        export_print_title: 'Chat Export',
        export_print_exported: 'Exported:',
        modules_title: '🧩 Modules',
        modules_none: 'No modules available',
        modules_enabled: 'enabled',
        modules_disabled: 'disabled',
        modules_toggle_failed: 'Module toggle failed',
        chunks_title: '📦 Chunks & Versioning',
        chunks_model_status: 'Model:',
        chunks_model_none: 'none registered',
        chunks_model_path_placeholder: 'Model path or name...',
        chunks_register_model: '🔍 Detect/Register',
        chunks_overview: 'Chunks',
        chunks_pending: 'Pending',
        chunks_trained: 'Trained',
        chunks_failed: 'Failed',
        chunks_restored: 'Restored',
        chunks_restore_all: '♻️ Restore All',
        chunks_restore_success: 'Restored {chunks} chunks ({docs} documents)',
        chunks_restore_none: 'No restorable chunks',
        chunks_restorable: '♻️ {count} restorable chunks ({docs} documents)',
        chunks_lora_section: 'LoRA Management',
        chunks_known_models: 'Known Models',
        chunks_history_show: '📜 Show History',
        chunks_model_same: '(unchanged)',
        chunks_model_changed: '(CHANGED)',
        chunks_model_switch_warning: '⚠ Model changed! {unusable} LoRAs unusable, {restorable} chunks restorable ({docs} docs)',
        chunks_model_current_tag: '(current)',
        chunks_lora_compatible: 'Compatible:',
        chunks_lora_none: 'No compatible LoRAs',
        chunks_lora_unusable: 'Unusable:',
        chunks_lora_select: 'Select',
        chunks_lora_selected: 'LoRA {name} selected',
        chunks_lora_incompatible: '⚠ LoRA not compatible with current model',
        chunks_confirm_delete_lora: 'Delete LoRA {name}?',
        chunks_confirm_delete_lora_files: 'Also delete LoRA files from disk?',
        device_shared: '(shared)',
        i18n_using: 'Using: {language}',
        i18n_coverage: 'Coverage: {translated}/{total} ({percent}%)',
        console_initializing: 'Initializing...',
        console_handlers_registered: 'Event handlers registered',
        console_initialized: 'Plugin initialized successfully'
    };

    var pluginBasePath = (function() {
        try {
            var scripts = document.querySelectorAll('script[src]');
            for (var i = 0; i < scripts.length; i++) {
                if (scripts[i].src.indexOf('Advanced_Memory_and_RLHF') !== -1) {
                    var src = scripts[i].src;
                    return src.substring(0, src.lastIndexOf('/') + 1);
                }
            }
        } catch (e) { /* ignore */ }
        return '/scripts/extensions/third-party/Advanced_Memory_and_RLHF/';
    })();

    var i18nLocalPath = pluginBasePath + 'backend/data/i18n/';

    function t(key, replacements) {
        var text = null;
        // Try current language first
        if (i18n[key] && typeof i18n[key] === 'string' && i18n[key].trim() !== '') {
            text = i18n[key];
        }
        // Then loaded English fallback
        if (!text && i18nFallback[key] && typeof i18nFallback[key] === 'string' && i18nFallback[key].trim() !== '') {
            text = i18nFallback[key];
        }
        // Then hardcoded builtin fallback (NEVER returns empty)
        if (!text && BUILTIN_FALLBACK[key]) {
            text = BUILTIN_FALLBACK[key];
        }
        // Last resort: return key itself
        if (!text) text = key;

        if (replacements) {
            Object.keys(replacements).forEach(function(k) {
                text = text.replace(new RegExp('\\{' + k + '\\}', 'g'), String(replacements[k]));
            });
        }
        text = text.replace(/\{[a-z_]+\}/g, function(match) { return replacements ? '' : match; });
        return text;
    }

    // Safe attribute escaping for data attributes
    function escapeAttr(text) {
        if (!text) return '';
        return String(text).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/'/g, '&#39;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

	async function loadI18nFile(url) {
			try {
				var response = await fetch(url);
				if (!response.ok) {
					console.warn('[' + PLUGIN_NAME + '] i18n fetch failed: ' + url + ' (status ' + response.status + ')');
					throw new Error('Failed to load: ' + url + ' (status ' + response.status + ')');
				}
				var data = await response.json();

				// Handle two response formats:
				// Format 1 (backend API): { translations: {...}, meta: {...} }
				// Format 2 (raw file): { _meta: {...}, key1: "val1", ... }
				if (data.translations && typeof data.translations === 'object') {
					// Backend API format - already structured
					return { translations: data.translations, meta: data.meta || null };
				} else {
					// Raw file format - extract _meta
					var meta = data._meta || null;
					var translations = {};
					Object.keys(data).forEach(function(key) {
						if (key !== '_meta') {
							translations[key] = data[key];
						}
					});
					return { translations: translations, meta: meta };
				}
			} catch (e) {
				console.warn('[' + PLUGIN_NAME + '] loadI18nFile error for ' + url + ':', e.message);
				throw e;
			}
		}

    async function discoverAvailableLanguages() {
        availableLanguages = [];

        // Try backend API first
        try {
            console.debug('[' + PLUGIN_NAME + '] Discovering languages from backend...');
            var response = await fetch(BACKEND_URL + '/i18n');
            if (response.ok) {
                var info = await response.json();
                i18nPath = info.i18n_path || '';
                if (info.available_languages && info.available_languages.length > 0) {
                    availableLanguages = info.available_languages;
                    console.log('[' + PLUGIN_NAME + '] Discovered ' + availableLanguages.length + ' languages: ' + availableLanguages.map(function(l) { return l.code; }).join(', '));
                    return;
                }
            }
        } catch (e) {
            console.debug('[' + PLUGIN_NAME + '] Backend language discovery failed: ' + e.message);
        }

        // Fallback: probe local files
        var priorityCodes = ['en','de','fr','es','it','pt','nl','ru','uk','ja','ko','zh','pl','tr','ar','cs','sv','no','vi','hi','th','id'];
        var altPaths = [
            i18nLocalPath,
            '/scripts/extensions/third-party/Advanced_Memory_and_RLHF/backend/data/i18n/'
        ];

        for (var pathIdx = 0; pathIdx < altPaths.length; pathIdx++) {
            var found = false;
            try {
                var probePromises = priorityCodes.map(function(code) {
                    return fetch(altPaths[pathIdx] + code + '.json', { method: 'HEAD' })
                        .then(function(resp) { return { code: code, ok: resp.ok, path: altPaths[pathIdx] }; })
                        .catch(function() { return { code: code, ok: false }; });
                });

                var probeResults = await Promise.all(probePromises);
                var foundCodes = probeResults.filter(function(r) { return r.ok; });

                if (foundCodes.length > 0) {
                    found = true;
                    var metaPromises = foundCodes.map(function(r) {
                        return loadI18nFile(r.path + r.code + '.json')
                            .then(function(langFile) {
                                return {
                                    code: r.code,
                                    name: (langFile.meta && langFile.meta.language) ? langFile.meta.language : r.code,
                                    native_name: (langFile.meta && langFile.meta.native_name) ? langFile.meta.native_name : r.code
                                };
                            })
                            .catch(function() {
                                return { code: r.code, name: r.code, native_name: r.code };
                            });
                    });
                    availableLanguages = await Promise.all(metaPromises);
                    console.log('[' + PLUGIN_NAME + '] Discovered ' + availableLanguages.length + ' languages from local files');
                }
            } catch (e) { /* ignore */ }
            if (found) break;
        }

        var hasEnglish = availableLanguages.some(function(l) { return l.code === 'en'; });
        if (!hasEnglish) {
            availableLanguages.unshift({ code: 'en', name: 'English', native_name: 'English' });
        }
    }

    function detectSillyTavernLanguage() {
        var lang = 'en';
        try {
            var htmlLang = document.documentElement.lang;
            if (htmlLang && htmlLang.length >= 2) lang = htmlLang.substring(0, 2).toLowerCase();
            var stLocale = localStorage.getItem('language');
            if (stLocale) lang = stLocale.substring(0, 2).toLowerCase();
        } catch (e) { /* default */ }
        return lang;
    }

    function getEffectiveLanguage() {
        if (languageOverride && languageOverride !== '' && languageOverride !== 'auto') return languageOverride;
        return detectedLanguage;
    }

    async function loadLanguageData(langCode) {
        // Method 1: Try backend API (most reliable)
        try {
            var backendUrl = BACKEND_URL + '/i18n/' + langCode;
            console.debug('[' + PLUGIN_NAME + '] Trying backend i18n: ' + backendUrl);
            var result = await loadI18nFile(backendUrl);
            if (result && result.translations && Object.keys(result.translations).length > 0) {
                console.log('[' + PLUGIN_NAME + '] Loaded ' + langCode + ' from backend (' + Object.keys(result.translations).length + ' keys)');
                return result;
            }
        } catch (e) {
            console.debug('[' + PLUGIN_NAME + '] Backend i18n/' + langCode + ' failed: ' + e.message);
        }

        // Method 2: Try local file paths (SillyTavern static serving)
        var localPaths = [
            i18nLocalPath + langCode + '.json',
            '/scripts/extensions/third-party/Advanced_Memory_and_RLHF/backend/data/i18n/' + langCode + '.json'
        ];

        for (var p = 0; p < localPaths.length; p++) {
            try {
                console.debug('[' + PLUGIN_NAME + '] Trying local i18n: ' + localPaths[p]);
                var localResult = await loadI18nFile(localPaths[p]);
                if (localResult && localResult.translations && Object.keys(localResult.translations).length > 0) {
                    console.log('[' + PLUGIN_NAME + '] Loaded ' + langCode + ' from local (' + Object.keys(localResult.translations).length + ' keys)');
                    return localResult;
                }
            } catch (e) {
                console.debug('[' + PLUGIN_NAME + '] Local path failed: ' + localPaths[p]);
            }
        }

        console.warn('[' + PLUGIN_NAME + '] Could not load language "' + langCode + '" from any source');
        return null;
    }

    async function loadI18n() {
        detectedLanguage = detectSillyTavernLanguage();
        try {
            var saved = localStorage.getItem(SETTINGS_KEY);
            if (saved) {
                var parsed = JSON.parse(saved);
                if (parsed.languageOverride) languageOverride = parsed.languageOverride;
            }
        } catch (e) { /* ignore */ }
        currentLanguage = getEffectiveLanguage();

        // Start with builtin fallback so UI is never blank
        i18nFallback = Object.assign({}, BUILTIN_FALLBACK);
        console.log('[' + PLUGIN_NAME + '] Builtin fallback active (' + Object.keys(BUILTIN_FALLBACK).length + ' strings)');

        // Try to discover available languages
        try {
            await discoverAvailableLanguages();
        } catch (e) {
            console.warn('[' + PLUGIN_NAME + '] Language discovery failed:', e.message);
        }

        // Try to load full English from backend/file
        try {
            var enData = await loadLanguageData('en');
            if (enData && enData.translations) {
                var enCount = 0;
                Object.keys(enData.translations).forEach(function(key) {
                    if (enData.translations[key] && typeof enData.translations[key] === 'string' && enData.translations[key].trim() !== '') {
                        i18nFallback[key] = enData.translations[key];
                        enCount++;
                    }
                });
                console.log('[' + PLUGIN_NAME + '] English merged: ' + enCount + ' keys from file, ' + Object.keys(i18nFallback).length + ' total');
            }
        } catch (e) {
            console.warn('[' + PLUGIN_NAME + '] en.json load failed, using builtin only');
        }

        // Load target language if not English
        if (currentLanguage !== 'en') {
            console.log('[' + PLUGIN_NAME + '] Loading language: ' + currentLanguage);
            await applyLanguage(currentLanguage);
        } else {
            updateI18nStats();
        }
    }

    async function applyLanguage(langCode) {
        i18n = {};
        if (langCode === 'en') {
            currentLanguage = 'en';
            updateI18nStats();
            return true;
        }

        console.log('[' + PLUGIN_NAME + '] Applying language: ' + langCode);
        var langData = await loadLanguageData(langCode);

        if (langData && langData.translations) {
            var applied = 0;
            Object.keys(langData.translations).forEach(function(key) {
                var value = langData.translations[key];
                if (value && typeof value === 'string' && value.trim() !== '') {
                    i18n[key] = value;
                    applied++;
                }
            });
            currentLanguage = langCode;
            updateI18nStats();
            console.log('[' + PLUGIN_NAME + '] Applied ' + langCode + ': ' + applied + ' translated, ' + i18nStats.missing.length + ' missing (fallback covers them)');
            return true;
        }

        console.warn('[' + PLUGIN_NAME + '] No translation data for "' + langCode + '"');
        currentLanguage = langCode;
        updateI18nStats();
        return false;
    }

    function updateI18nStats() {
        var allKeys = Object.keys(i18nFallback);
        if (currentLanguage === 'en') { i18nStats = { total: allKeys.length, translated: allKeys.length, missing: [] }; return; }
        var missing = [];
        allKeys.forEach(function(key) {
            if (!i18n[key] || typeof i18n[key] !== 'string' || i18n[key].trim() === '') missing.push(key);
        });
        i18nStats = { total: allKeys.length, translated: allKeys.length - missing.length, missing: missing };
    }

    async function switchLanguage(langCode) {
        languageOverride = langCode;
        saveSettings();
        await applyLanguage(getEffectiveLanguage());
        rebuildUI();
    }

    function rebuildUI() {
        var existing = document.getElementById('memory-plugin-container');
        if (existing) existing.remove();
        injectUI();
        if (feedbackMode) { var fb = document.getElementById('amp-btn-toggle-feedback'); if (fb) fb.textContent = t('rlhf_toggle_on'); }
        if (!autoStoreEnabled) { var as = document.getElementById('amp-btn-toggle-autostore'); if (as) as.textContent = t('memory_autostore_off'); }
        if (!ragInjectionEnabled) { var rg = document.getElementById('amp-btn-toggle-rag'); if (rg) rg.textContent = t('memory_rag_off'); }
        if (backendAvailable) {
            setTimeout(updateStatus, 500);
            setTimeout(refreshCharacterList, 1000);
            setTimeout(updateChunkOverview, 1500);
        }
    }

    // ===================== SETTINGS =====================

    function loadSettings() {
        try { var s = localStorage.getItem(SETTINGS_KEY); if (s) return JSON.parse(s); } catch (e) { /* ignore */ }
        return { feedbackMode: false, autoStoreEnabled: true, ragInjectionEnabled: true, languageOverride: '' };
    }

    function saveSettings() {
        try {
            localStorage.setItem(SETTINGS_KEY, JSON.stringify({
                feedbackMode: feedbackMode, autoStoreEnabled: autoStoreEnabled,
                ragInjectionEnabled: ragInjectionEnabled, languageOverride: languageOverride
            }));
        } catch (e) { /* ignore */ }
    }

    // ===================== BACKEND CHECK =====================

    async function checkBackend() {
        var retries = 5;
        while (retries > 0) {
            try {
                var response = await fetch(BACKEND_URL + '/health');
                if (response.ok) { backendAvailable = true; return true; }
            } catch (e) { /* not ready */ }
            retries--;
            if (retries > 0) await new Promise(function(r) { setTimeout(r, 2000); });
        }
        backendAvailable = false;
        console.warn('[' + PLUGIN_NAME + '] Backend not reachable at ' + BACKEND_URL);
        return false;
    }

    // ===================== API =====================

    class BackendAPI {
        static async request(endpoint, method, body) {
            method = method || 'GET'; body = body || null;
            var options = { method: method, headers: { 'Content-Type': 'application/json' } };
            if (body) options.body = JSON.stringify(body);
            var response = await fetch(BACKEND_URL + endpoint, options);
            if (!response.ok) { var errText = await response.text(); throw new Error('API Error ' + response.status + ': ' + errText); }
            return await response.json();
        }
        static async uploadFile(endpoint, file) {
            var fd = new FormData(); fd.append('file', file);
            var r = await fetch(BACKEND_URL + endpoint, { method: 'POST', body: fd });
            if (!r.ok) throw new Error('Upload failed: ' + r.status);
            return await r.json();
        }
        static getStatus() { return this.request('/health'); }
        static getMemoryStatus() { return this.request('/memory/status'); }
        static storeMemory(data) { return this.request('/memory/store', 'POST', data); }
        static queryMemory(query, k) { return this.request('/memory/query', 'POST', { query: query, k: k || 5 }); }
        static browseMemory(character, offset, limit) {
            var p = '?offset=' + (offset||0) + '&limit=' + (limit||50);
            if (character) p += '&character=' + encodeURIComponent(character);
            return this.request('/memory/browse' + p);
        }
        static deleteMemory(data) { return this.request('/memory/delete', 'POST', data); }
        static clearAllMemory() { return this.request('/memory/clear-all', 'POST'); }
        static getCharacters() { return this.request('/memory/characters'); }
        static getMemoryHistory(limit) { return this.request('/memory/history?limit=' + (limit||100)); }
        static startTraining(config) { return this.request('/training/start', 'POST', config); }
        static getTrainingProgress() { return this.request('/training/progress'); }
        static cancelTraining() { return this.request('/training/cancel', 'POST'); }
        static getLoraStatus() { return this.request('/training/lora-status'); }
        static submitFeedback(data) { return this.request('/rlhf/feedback', 'POST', data); }
        static getFeedbackStats() { return this.request('/rlhf/stats'); }
        static exportChat(format, chatData) { return this.request('/documents/export', 'POST', { format: format, chat_data: chatData }); }
        static ingestDocument(file) { return this.uploadFile('/documents/ingest', file); }
        static generateSummary(chatData) { return this.request('/documents/summary', 'POST', { chat_data: chatData }); }
        static getDownloadUrl(filename) { return BACKEND_URL + '/documents/download/' + encodeURIComponent(filename); }
        static getModules() { return this.request('/modules/list'); }
        static toggleModule(name, enabled) { return this.request('/modules/toggle', 'POST', { name: name, enabled: enabled }); }
        static getDeviceInfo() { return this.request('/device/info'); }
    }

    // ===================== LOADING INDICATOR =====================

    function setLoading(elementId, loading) {
        var el = document.getElementById(elementId);
        if (!el) return;
        if (loading) {
            el.dataset.originalText = el.textContent;
            el.textContent = '⏳ ' + el.textContent;
            el.disabled = true;
            el.style.opacity = '0.6';
        } else {
            if (el.dataset.originalText) el.textContent = el.dataset.originalText;
            el.disabled = false;
            el.style.opacity = '1';
        }
    }

    // ===================== DEDUPLICATION =====================

    function computeContentHash(text) {
        var hash = 0;
        for (var i = 0; i < text.length; i++) {
            var chr = text.charCodeAt(i);
            hash = ((hash << 5) - hash) + chr;
            hash |= 0;
        }
        return hash.toString(36);
    }

    // ===================== LANGUAGE SELECTOR =====================

    function buildLanguageSelector() {
        var effectiveLang = getEffectiveLanguage();
        var isOverridden = languageOverride && languageOverride !== '' && languageOverride !== 'auto';

        var html = '<div id="amp-i18n-section" class="amp-section">' +
            '<h4>🌐 Language Override</h4>' +
            '<div class="amp-info-text" style="margin-bottom:5px;">Plugin language (independent of SillyTavern)</div>' +
            '<div class="amp-button-row" style="align-items:center;">' +
                '<label style="white-space:nowrap;">Language:</label>' +
                '<select id="amp-language-override" class="text_pole" style="flex:1;">' +
                    '<option value="auto"' + (!isOverridden ? ' selected' : '') + '>Auto-detect (ST: ' + detectedLanguage + ')</option>';

        for (var i = 0; i < availableLanguages.length; i++) {
            var lang = availableLanguages[i];
            var selected = (isOverridden && languageOverride === lang.code) ? ' selected' : '';
            var displayName = lang.name;
            if (lang.native_name && lang.native_name !== lang.name) displayName = lang.name + ' (' + lang.native_name + ')';
            html += '<option value="' + escapeAttr(lang.code) + '"' + selected + '>' + escapeHtml(displayName) + ' [' + escapeHtml(lang.code) + ']</option>';
        }

        html += '</select></div>';
        html += '<div class="amp-info-text" style="margin-top:5px;">';
        html += isOverridden ? 'Active: <b>' + escapeHtml(effectiveLang) + '</b> (override)' : 'Active: <b>' + escapeHtml(effectiveLang) + '</b> (auto)';
        html += '</div>';
        html += '<div class="amp-info-text" style="margin-top:3px;font-style:italic;color:#aaa;">' + t('i18n_using', {language: effectiveLang}) + '</div>';

        if (effectiveLang !== 'en' && i18nStats.total > 0) {
            var percent = Math.round((i18nStats.translated / i18nStats.total) * 100);
            var color = percent === 100 ? '#88cc88' : (percent >= 80 ? '#cccc44' : '#cc8844');
            html += '<div class="amp-info-text" style="margin-top:3px;color:' + color + ';">' + t('i18n_coverage', {translated: i18nStats.translated, total: i18nStats.total, percent: percent}) + '</div>';
            if (i18nStats.missing.length > 0 && i18nStats.missing.length <= 10) {
                html += '<div class="amp-info-text" style="margin-top:3px;color:#cc8844;font-size:0.85em;">Missing: ' + i18nStats.missing.join(', ') + '</div>';
            } else if (i18nStats.missing.length > 10) {
                html += '<div class="amp-info-text" style="margin-top:3px;color:#cc8844;font-size:0.85em;">Missing: ' + i18nStats.missing.length + ' strings</div>';
            }
        }
        if (effectiveLang !== 'en' && Object.keys(i18n).length === 0) {
            html += '<div class="amp-info-text" style="color:#ffaa00;margin-top:3px;">⚠ No translation for "' + escapeHtml(effectiveLang) + '". Using English.</div>';
        }
        if (effectiveLang === 'en') {
            html += '<div class="amp-info-text" style="margin-top:3px;color:#88cc88;">✓ English — all ' + i18nStats.total + ' strings</div>';
        }

        html += '<button id="amp-btn-i18n-folder" class="menu_button" style="margin-top:5px;">📂 Open i18n folder</button></div>';
        return html;
    }

    // ===================== CHUNK SYSTEM UI =====================

    function buildChunkSection() {
        return '<div id="amp-chunks-section" class="amp-section">' +
            '<h4>' + t('chunks_title') + '</h4>' +
            '<div id="amp-chunk-model-area">' +
                '<div id="amp-chunk-model-status" class="amp-info-text">' + t('chunks_model_status') + ' ' + t('chunks_model_none') + '</div>' +
                '<div class="amp-button-row" style="margin-top:5px;">' +
                    '<input type="text" id="amp-model-path" placeholder="' + escapeAttr(t('chunks_model_path_placeholder')) + '" class="text_pole" style="flex:1;">' +
                    '<button id="amp-btn-detect-model" class="menu_button">' + t('chunks_register_model') + '</button>' +
                '</div>' +
                '<div id="amp-model-switch-warning" style="display:none;background:#443300;border:1px solid #886600;border-radius:4px;padding:8px;margin-top:5px;"></div>' +
            '</div>' +
            '<div id="amp-chunk-overview" class="amp-info-text" style="margin-top:8px;">' + t('chunks_overview') + ': ...</div>' +
            '<div id="amp-restore-area" style="margin-top:8px;">' +
                '<div id="amp-restore-info" class="amp-info-text"></div>' +
                '<div class="amp-button-row" style="margin-top:5px;"><button id="amp-btn-restore-all" class="menu_button" style="display:none;">' + t('chunks_restore_all') + '</button></div>' +
                '<div id="amp-restore-list" style="display:none;max-height:200px;overflow-y:auto;margin-top:5px;"></div>' +
            '</div>' +
            '<div id="amp-lora-manage-area" style="margin-top:8px;">' +
                '<h5 style="margin:5px 0;">' + t('chunks_lora_section') + '</h5>' +
                '<div id="amp-lora-compatible-list"></div>' +
                '<div id="amp-lora-unusable-list" style="margin-top:5px;"></div>' +
            '</div>' +
            '<div id="amp-known-models-area" style="margin-top:8px;">' +
                '<h5 style="margin:5px 0;">' + t('chunks_known_models') + '</h5>' +
                '<div id="amp-known-models-list"></div>' +
            '</div>' +
            '<div style="margin-top:8px;">' +
                '<button id="amp-btn-chunk-history" class="menu_button">' + t('chunks_history_show') + '</button>' +
                '<div id="amp-chunk-history-list" style="display:none;max-height:300px;overflow-y:auto;margin-top:5px;"></div>' +
            '</div>' +
        '</div>';
    }

    // ===================== UI INJECTION =====================

    function injectUI() {
        var extensionsPanel = document.getElementById('extensions_settings');
        if (!extensionsPanel) { setTimeout(injectUI, 1000); return; }

        var c = document.createElement('div');
        c.id = 'memory-plugin-container';
        c.innerHTML =
            '<div class="inline-drawer">' +
                '<div class="inline-drawer-toggle inline-drawer-header">' +
                    '<b>' + t('plugin_title') + '</b>' +
                    '<div class="inline-drawer-icon fa-solid fa-circle-chevron-down down"></div>' +
                '</div>' +
                '<div class="inline-drawer-content">' +

                    '<div id="amp-status-section" class="amp-section">' +
                        '<h4>' + t('status_title') + '</h4>' +
                        '<div id="amp-backend-status">' + t('status_backend') + ': <span id="amp-backend-dot">' + t('status_loading') + '</span></div>' +
                        '<div id="amp-device-info" class="amp-info-text">' + t('status_device_detecting') + '</div>' +
                        '<div id="amp-memory-size" class="amp-info-text">' + t('status_vectordb') + '</div>' +
                        '<div id="amp-lora-status" class="amp-info-text">' + t('status_lora_none') + '</div>' +
                    '</div>' +

                    '<div id="amp-memory-section" class="amp-section">' +
                        '<h4>' + t('memory_title') + '</h4>' +
                        '<div class="amp-button-row">' +
                            '<button id="amp-btn-toggle-autostore" class="menu_button">' + t('memory_autostore_on') + '</button>' +
                            '<button id="amp-btn-toggle-rag" class="menu_button">' + t('memory_rag_on') + '</button>' +
                        '</div>' +
                        '<div class="amp-button-row" style="margin-top:5px;">' +
                            '<label style="display:flex;align-items:center;gap:5px;width:100%;">' + t('memory_rag_max_chars') + ' <input type="number" id="amp-rag-max-chars" value="1500" min="200" max="5000" step="100" class="text_pole narrow" style="width:80px;"></label>' +
                            '<label style="display:flex;align-items:center;gap:5px;width:100%;">' + t('memory_rag_min_score') + ' <input type="number" id="amp-rag-min-score" value="0.3" min="0.1" max="0.9" step="0.05" class="text_pole narrow" style="width:80px;"></label>' +
                        '</div>' +
                        '<div class="amp-button-row" style="margin-top:5px;">' +
                            '<button id="amp-btn-store-chat" class="menu_button">' + t('memory_store_chat') + '</button>' +
                            '<button id="amp-btn-query-memory" class="menu_button">' + t('memory_query') + '</button>' +
                        '</div>' +
                        '<div id="amp-query-input-area" style="display:none;">' +
                            '<textarea id="amp-query-text" rows="2" placeholder="' + escapeAttr(t('memory_query_placeholder')) + '" class="text_pole"></textarea>' +
                            '<button id="amp-btn-run-query" class="menu_button">' + t('memory_query_search') + '</button>' +
                            '<div id="amp-query-results" class="amp-results-box"></div>' +
                        '</div>' +
                    '</div>' +

                    '<div id="amp-browser-section" class="amp-section">' +
                        '<h4>' + t('browser_title') + '</h4>' +
                        '<div class="amp-button-row">' +
                            '<select id="amp-browse-character" class="text_pole" style="flex:1;"><option value="">' + t('browser_all_characters') + '</option></select>' +
                            '<button id="amp-btn-browse" class="menu_button">' + t('browser_browse') + '</button>' +
                            '<button id="amp-btn-browse-history" class="menu_button">' + t('browser_history') + '</button>' +
                        '</div>' +
                        '<div id="amp-browse-results" style="display:none;max-height:400px;overflow-y:auto;margin-top:5px;">' +
                            '<div id="amp-browse-list"></div>' +
                            '<div class="amp-button-row" style="margin-top:5px;">' +
                                '<button id="amp-btn-browse-prev" class="menu_button" disabled>' + t('browser_prev') + '</button>' +
                                '<span id="amp-browse-page-info" style="padding:5px;">' + t('browser_page') + ' 1</span>' +
                                '<button id="amp-btn-browse-next" class="menu_button">' + t('browser_next') + '</button>' +
                            '</div>' +
                        '</div>' +
                        '<div class="amp-button-row" style="margin-top:5px;">' +
                            '<button id="amp-btn-delete-selected" class="menu_button" style="display:none;">' + t('browser_delete_selected') + '</button>' +
                            '<button id="amp-btn-clear-character" class="menu_button">' + t('browser_clear_character') + '</button>' +
                            '<button id="amp-btn-clear-all" class="menu_button" style="color:#ff4444;">' + t('browser_clear_all') + '</button>' +
                        '</div>' +
                    '</div>' +

                    '<div id="amp-documents-section" class="amp-section">' +
                        '<h4>' + t('documents_title') + '</h4>' +
                        '<div class="amp-button-row">' +
                            '<label class="menu_button" for="amp-file-upload">' + t('documents_upload') + '</label>' +
                            '<input type="file" id="amp-file-upload" accept=".txt,.json,.pdf,.xml,.doc,.docx,.odt,.ods,.odp,.xls,.xlsx,.pptx,.png,.jpg,.jpeg,.gif,.webp" multiple style="display:none;">' +
                        '</div>' +
                        '<div id="amp-upload-progress" style="display:none;"><div class="amp-progress-bar"><div class="amp-progress-fill" id="amp-upload-fill"></div></div><span id="amp-upload-text">...</span></div>' +
                    '</div>' +

                    '<div id="amp-rlhf-section" class="amp-section">' +
                        '<h4>' + t('rlhf_title') + '</h4>' +
                        '<div class="amp-button-row"><button id="amp-btn-toggle-feedback" class="menu_button">' + t('rlhf_toggle') + '</button></div>' +
                        '<div id="amp-feedback-stats" class="amp-info-text">' + t('rlhf_stats', {positive:0,negative:0,excellent:0,total:0}) + '</div>' +
                        '<p class="amp-hint">' + t('rlhf_hint') + '</p>' +
                    '</div>' +

                    '<div id="amp-training-section" class="amp-section">' +
                        '<h4>' + t('training_title') + '</h4>' +
                        '<div id="amp-training-model-info" class="amp-info-text" style="margin-bottom:8px;padding:5px;border:1px solid #444;border-radius:4px;">' +
                            '<div id="amp-detected-model" style="color:#aaa;">' + t('training_model_detecting') + '</div>' +
                        '</div>' +
                        '<div class="amp-training-config">' +
                            '<label>' + t('training_base_model') + ' <input type="text" id="amp-train-base-model" placeholder="' + escapeAttr(t('training_model_placeholder')) + '" class="text_pole" style="width:100%;"></label>' +
                            '<div class="amp-info-text" style="font-size:0.85em;color:#888;margin-top:2px;">' + t('training_model_hint') + '</div>' +
                            '<label>' + t('training_epochs') + ' <input type="number" id="amp-train-epochs" value="3" min="1" max="50" class="text_pole narrow"></label>' +
                            '<label>' + t('training_learning_rate') + ' <input type="text" id="amp-train-lr" value="2e-4" class="text_pole narrow"></label>' +
                            '<label>' + t('training_lora_rank') + ' <input type="number" id="amp-train-rank" value="16" min="4" max="128" class="text_pole narrow"></label>' +
                            '<label>' + t('training_batch_size') + ' <input type="number" id="amp-train-batch" value="4" min="1" max="32" class="text_pole narrow"></label>' +
                        '</div>' +
                        '<div class="amp-button-row">' +
                            '<button id="amp-btn-detect-st-model" class="menu_button">🔍 ' + t('training_detect_model') + '</button>' +
                            '<button id="amp-btn-start-training" class="menu_button">' + t('training_start') + '</button>' +
                            '<button id="amp-btn-cancel-training" class="menu_button" style="display:none;">' + t('training_cancel') + '</button>' +
                        '</div>' +
                        '<div id="amp-training-progress" style="display:none;"><div class="amp-progress-bar"><div class="amp-progress-fill" id="amp-train-fill"></div></div><div id="amp-train-status-text">' + t('training_preparing') + '</div><div id="amp-train-eta">' + t('training_eta', {eta:'...'}) + '</div></div>' +
                    '</div>' +

                    '<div id="amp-export-section" class="amp-section">' +
                        '<h4>' + t('export_title') + '</h4>' +
                        '<div class="amp-button-row"><button id="amp-btn-summary" class="menu_button">' + t('export_summary') + '</button></div>' +
                        '<select id="amp-export-format" class="text_pole">' +
                            '<option value="txt">' + t('export_format_txt') + '</option>' +
                            '<option value="json">' + t('export_format_json') + '</option>' +
                            '<option value="pdf">' + t('export_format_pdf') + '</option>' +
                            '<option value="xml">' + t('export_format_xml') + '</option>' +
                            '<option value="docx">' + t('export_format_docx') + '</option>' +
                            '<option value="odt">' + t('export_format_odt') + '</option>' +
                            '<option value="html">' + t('export_format_html') + '</option>' +
                        '</select>' +
                        '<div class="amp-button-row">' +
                            '<button id="amp-btn-export" class="menu_button">' + t('export_chat') + '</button>' +
                            '<button id="amp-btn-print" class="menu_button">' + t('export_print') + '</button>' +
                        '</div>' +
                        '<div id="amp-summary-output" class="amp-results-box" style="display:none;"></div>' +
                    '</div>' +

                    '<div id="amp-modules-section" class="amp-section">' +
                        '<h4>' + t('modules_title') + '</h4>' +
                        '<div id="amp-modules-list"></div>' +
                    '</div>' +

                    buildChunkSection() +
                    buildLanguageSelector() +

                '</div>' +
            '</div>';

        extensionsPanel.appendChild(c);
    }

    // ===================== EVENT BINDING =====================

    function bindEvents() {
        document.addEventListener('click', function(e) {
            var target = e.target;

            if (target.id === 'amp-btn-toggle-feedback' || target.closest('#amp-btn-toggle-feedback')) {
                feedbackMode = !feedbackMode;
                var btn = document.getElementById('amp-btn-toggle-feedback');
                if (btn) btn.textContent = feedbackMode ? t('rlhf_toggle_on') : t('rlhf_toggle');
                saveSettings(); toggleFeedbackButtons(feedbackMode); return;
            }
            if (target.id === 'amp-btn-toggle-autostore' || target.closest('#amp-btn-toggle-autostore')) {
                autoStoreEnabled = !autoStoreEnabled;
                var asB = document.getElementById('amp-btn-toggle-autostore');
                if (asB) asB.textContent = autoStoreEnabled ? t('memory_autostore_on') : t('memory_autostore_off');
                saveSettings(); return;
            }
            if (target.id === 'amp-btn-toggle-rag' || target.closest('#amp-btn-toggle-rag')) {
                ragInjectionEnabled = !ragInjectionEnabled;
                var rgB = document.getElementById('amp-btn-toggle-rag');
                if (rgB) rgB.textContent = ragInjectionEnabled ? t('memory_rag_on') : t('memory_rag_off');
                saveSettings(); return;
            }

            if (target.id === 'amp-btn-store-chat' || target.closest('#amp-btn-store-chat')) {
                (async function() {
                    var chatData = extractCurrentChat();
                    if (!chatData || chatData.messages.length === 0) { showToast(t('memory_no_data'), 'warning'); return; }
                    setLoading('amp-btn-store-chat', true);
                    try {
                        await BackendAPI.storeMemory(chatData);
                        try {
                            await BackendAPI.request('/chunks/create', 'POST', {
                                documents: chatData.messages.map(function(m) {
                                    return { role: m.role, name: m.name, content: m.content, text: m.content };
                                }),
                                character: chatData.character,
                                metadata: { source: 'manual_store', timestamp: chatData.timestamp }
                            });
                        } catch (chunkErr) { console.debug('[' + PLUGIN_NAME + '] Chunk creation skipped:', chunkErr.message); }
                        showToast(t('memory_stored_success'), 'success');
                        updateStatus();
                    } catch (err) { showToast(t('memory_store_failed') + ' ' + err.message, 'error'); }
                    finally { setLoading('amp-btn-store-chat', false); }
                })(); return;
            }

            if (target.id === 'amp-btn-query-memory' || target.closest('#amp-btn-query-memory')) {
                var area = document.getElementById('amp-query-input-area');
                if (area) area.style.display = area.style.display === 'none' ? 'block' : 'none'; return;
            }

            if (target.id === 'amp-btn-run-query' || target.closest('#amp-btn-run-query')) {
                (async function() {
                    var query = document.getElementById('amp-query-text');
                    if (!query || !query.value.trim()) return;
                    setLoading('amp-btn-run-query', true);
                    try {
                        var results = await BackendAPI.queryMemory(query.value.trim());
                        var rd = document.getElementById('amp-query-results');
                        if (rd) {
                            if (results.results && results.results.length > 0) {
                                rd.innerHTML = results.results.map(function(r, i) {
                                    return '<div class="amp-result-item"><b>#' + (i+1) + '</b> (score: ' + r.score.toFixed(3) + ')<br>' + escapeHtml(r.text.substring(0,300)) + '...</div>';
                                }).join('');
                            } else { rd.innerHTML = '<i>' + t('memory_no_results') + '</i>'; }
                        }
                    } catch (err) { showToast(t('memory_query_failed') + ' ' + err.message, 'error'); }
                    finally { setLoading('amp-btn-run-query', false); }
                })(); return;
            }

            if (target.id === 'amp-btn-browse' || target.closest('#amp-btn-browse')) {
                (async function() {
                    var cs = document.getElementById('amp-browse-character');
                    try { var r = await BackendAPI.browseMemory(cs ? cs.value : '', 0, 50); displayBrowseResults(r, 0); }
                    catch (err) { showToast(t('browser_browse_failed') + ' ' + err.message, 'error'); }
                })(); return;
            }
            if (target.id === 'amp-btn-browse-history' || target.closest('#amp-btn-browse-history')) {
                (async function() {
                    try { var r = await BackendAPI.getMemoryHistory(100); displayHistoryResults(r.history); }
                    catch (err) { showToast(t('browser_history_failed') + ' ' + err.message, 'error'); }
                })(); return;
            }
            if (target.id === 'amp-btn-browse-next' || target.closest('#amp-btn-browse-next')) {
                (async function() {
                    var pi = document.getElementById('amp-browse-page-info');
                    var off = parseInt(pi.dataset.offset || '0'); var cs = document.getElementById('amp-browse-character');
                    try { var r = await BackendAPI.browseMemory(cs ? cs.value : '', off + 50, 50); displayBrowseResults(r, off + 50); }
                    catch (err) { showToast(t('browser_browse_failed') + ' ' + err.message, 'error'); }
                })(); return;
            }
            if (target.id === 'amp-btn-browse-prev' || target.closest('#amp-btn-browse-prev')) {
                (async function() {
                    var pi = document.getElementById('amp-browse-page-info');
                    var off = Math.max(0, parseInt(pi.dataset.offset || '0') - 50); var cs = document.getElementById('amp-browse-character');
                    try { var r = await BackendAPI.browseMemory(cs ? cs.value : '', off, 50); displayBrowseResults(r, off); }
                    catch (err) { showToast(t('browser_browse_failed') + ' ' + err.message, 'error'); }
                })(); return;
            }

            if (target.id === 'amp-btn-delete-selected' || target.closest('#amp-btn-delete-selected')) {
                var cbs = document.querySelectorAll('.amp-doc-checkbox:checked');
                var ids = []; cbs.forEach(function(cb) { ids.push(cb.dataset.docId); });
                if (ids.length === 0) { showToast(t('browser_no_selected'), 'warning'); return; }
                if (!confirm(t('browser_confirm_delete_selected', {count: ids.length}))) return;
                (async function() {
                    setLoading('amp-btn-delete-selected', true);
                    try { var r = await BackendAPI.deleteMemory({ doc_ids: ids }); showToast(t('browser_deleted_success', {count: r.deleted}), 'success'); document.getElementById('amp-btn-browse').click(); updateStatus(); }
                    catch (err) { showToast(t('browser_delete_failed') + ' ' + err.message, 'error'); }
                    finally { setLoading('amp-btn-delete-selected', false); }
                })(); return;
            }

            if (target.id === 'amp-btn-clear-character' || target.closest('#amp-btn-clear-character')) {
                var cSel = document.getElementById('amp-browse-character'); var cName = cSel ? cSel.value : '';
                if (!cName) { showToast(t('browser_select_character_first'), 'warning'); return; }
                if (!confirm(t('browser_confirm_delete_character', {character: cName}))) return;
                (async function() {
                    try { var r = await BackendAPI.deleteMemory({ character: cName }); showToast(t('browser_deleted_character', {count: r.deleted, character: cName}), 'success'); updateStatus(); refreshCharacterList(); }
                    catch (err) { showToast(t('browser_delete_failed') + ' ' + err.message, 'error'); }
                })(); return;
            }

            if (target.id === 'amp-btn-clear-all' || target.closest('#amp-btn-clear-all')) {
                if (!confirm(t('browser_confirm_clear_all'))) return;
                if (!confirm(t('browser_confirm_clear_all_2'))) return;
                (async function() {
                    try { var r = await BackendAPI.clearAllMemory(); showToast(r.message, 'success'); updateStatus(); refreshCharacterList(); var br = document.getElementById('amp-browse-results'); if (br) br.style.display = 'none'; }
                    catch (err) { showToast(t('browser_clear_failed') + ' ' + err.message, 'error'); }
                })(); return;
            }

            if (target.id === 'amp-select-all-docs') {
                var allC = target.checked; document.querySelectorAll('.amp-doc-checkbox').forEach(function(cb) { cb.checked = allC; });
                var dB = document.getElementById('amp-btn-delete-selected');
                if (dB) { var cc = document.querySelectorAll('.amp-doc-checkbox:checked').length; dB.style.display = cc > 0 ? 'inline-block' : 'none'; dB.textContent = t('browser_delete_selected') + ' (' + cc + ')'; }
                return;
            }

            if (target.id === 'amp-btn-detect-st-model' || target.closest('#amp-btn-detect-st-model')) {
                (async function() {
                    var modelInfo = detectSillyTavernModel();
                    var modelInput = document.getElementById('amp-train-base-model');
                    var modelDisplay = document.getElementById('amp-detected-model');

                    if (modelInfo.model_path) {
                        if (modelInput) modelInput.value = modelInfo.model_path;
                        if (modelDisplay) {
                            modelDisplay.innerHTML =
                                '<span style="color:#88cc88;">✓ ' + t('training_model_detected') + '</span><br>' +
                                '<b>' + escapeHtml(modelInfo.model_name || modelInfo.model_path) + '</b><br>' +
                                '<span style="color:#aaa;font-size:0.85em;">' +
                                t('training_model_source') + ': ' + escapeHtml(modelInfo.source) +
                                (modelInfo.api_type ? ' | API: ' + escapeHtml(modelInfo.api_type) : '') +
                                '</span>';
                        }
                    } else {
                        if (modelDisplay) {
                            modelDisplay.innerHTML =
                                '<span style="color:#ffaa00;">⚠ ' + t('training_model_not_detected') + '</span><br>' +
                                '<span style="color:#aaa;font-size:0.85em;">' + t('training_model_manual_hint') + '</span>';
                        }
                        showToast(t('training_model_not_detected'), 'warning');
                    }
                })(); return;
            }

            if (target.id === 'amp-btn-start-training' || target.closest('#amp-btn-start-training')) {
                (async function() {
                    var baseModel = document.getElementById('amp-train-base-model');
                    var modelPath = baseModel ? baseModel.value.trim() : '';

                    // Validate model path
                    if (!modelPath) {
                        // Try auto-detect
                        var detected = detectSillyTavernModel();
                        if (detected.model_path) {
                            modelPath = detected.model_path;
                            if (baseModel) baseModel.value = modelPath;
                        } else {
                            showToast(t('training_model_required'), 'error');
                            return;
                        }
                    }

                    // Warn if it looks like a cloud API model
                    var cloudPatterns = ['gpt-', 'claude-', 'gemini-', 'palm-', 'command-'];
                    var isCloud = cloudPatterns.some(function(p) { return modelPath.toLowerCase().indexOf(p) !== -1; });
                    if (isCloud) {
                        if (!confirm(t('training_cloud_warning'))) return;
                    }

                    var config = {
                        base_model: modelPath,
                        epochs: parseInt(document.getElementById('amp-train-epochs').value || '3'),
                        learning_rate: parseFloat(document.getElementById('amp-train-lr').value || '2e-4'),
                        lora_rank: parseInt(document.getElementById('amp-train-rank').value || '16'),
                        batch_size: parseInt(document.getElementById('amp-train-batch').value || '4')
                    };

                    if (!confirm(t('training_confirm') + '\n\n' + t('training_model_label') + ': ' + modelPath)) return;

                    try {
                        // Mark untrained chunks as training
                        try {
                            var untrained = await BackendAPI.request('/chunks/untrained');
                            if (untrained.chunks) {
                                for (var uc = 0; uc < untrained.chunks.length; uc++) {
                                    try { await BackendAPI.request('/chunks/' + untrained.chunks[uc].chunk_id, 'GET'); } catch(e) { /* ignore */ }
                                }
                            }
                        } catch (ce) { /* chunk system optional */ }

                        await BackendAPI.startTraining(config);
                        var p = document.getElementById('amp-training-progress');
                        var s = document.getElementById('amp-btn-start-training');
                        var cv = document.getElementById('amp-btn-cancel-training');
                        if (p) p.style.display = 'block';
                        if (s) s.style.display = 'none';
                        if (cv) cv.style.display = 'inline-block';
                        pollTrainingProgress();
                    } catch (err) { showToast(t('training_start_failed') + ' ' + err.message, 'error'); }
                })(); return;
            }

            if (target.id === 'amp-btn-cancel-training' || target.closest('#amp-btn-cancel-training')) {
                (async function() {
                    try { await BackendAPI.cancelTraining(); showToast(t('training_cancelled'), 'warning'); resetTrainingUI(); }
                    catch (err) { showToast(t('training_cancel_failed') + ' ' + err.message, 'error'); }
                })(); return;
            }

            if (target.id === 'amp-btn-export' || target.closest('#amp-btn-export')) {
                (async function() {
                    var fmt = document.getElementById('amp-export-format'); var chatData = extractCurrentChat();
                    setLoading('amp-btn-export', true);
                    try {
                        var r = await BackendAPI.exportChat(fmt ? fmt.value : 'txt', chatData);
                        if (r.download_url) { var a = document.createElement('a'); a.href = BACKEND_URL + r.download_url; a.download = r.filename || 'export'; document.body.appendChild(a); a.click(); document.body.removeChild(a); }
                        else if (r.filename) { var a2 = document.createElement('a'); a2.href = BackendAPI.getDownloadUrl(r.filename); a2.download = r.filename; document.body.appendChild(a2); a2.click(); document.body.removeChild(a2); }
                        showToast(t('export_success'), 'success');
                    } catch (err) { showToast(t('export_failed') + ' ' + err.message, 'error'); }
                    finally { setLoading('amp-btn-export', false); }
                })(); return;
            }

            if (target.id === 'amp-btn-print' || target.closest('#amp-btn-print')) {
                var cd = extractCurrentChat(); var pw = window.open('', '_blank');
                pw.document.write('<html><head><title>' + escapeHtml(t('export_print_title')) + '</title><style>body{font-family:Arial,sans-serif;max-width:800px;margin:0 auto;padding:20px}.message{margin:10px 0;padding:10px;border-radius:8px}.user{background:#e3f2fd}.assistant{background:#f5f5f5}.name{font-weight:bold;margin-bottom:5px}</style></head><body><h1>' + escapeHtml(t('export_print_title')) + ' - ' + escapeHtml(cd.character || 'Unknown') + '</h1><p>' + escapeHtml(t('export_print_exported')) + ' ' + new Date().toLocaleString() + '</p><hr>' + cd.messages.map(function(m) { return '<div class="message ' + escapeAttr(m.role) + '"><div class="name">' + escapeHtml(m.name || m.role) + '</div><div>' + escapeHtml(m.content) + '</div></div>'; }).join('') + '</body></html>');
                pw.document.close(); pw.print(); return;
            }

            if (target.id === 'amp-btn-summary' || target.closest('#amp-btn-summary')) {
                (async function() {
                    setLoading('amp-btn-summary', true);
                    try { var r = await BackendAPI.generateSummary(extractCurrentChat()); var out = document.getElementById('amp-summary-output'); if (out) { out.style.display = 'block'; out.innerHTML = '<b>Summary:</b><br>' + escapeHtml(r.summary); } }
                    catch (err) { showToast(t('export_summary_failed') + ' ' + err.message, 'error'); }
                    finally { setLoading('amp-btn-summary', false); }
                })(); return;
            }

            if (target.id === 'amp-btn-i18n-folder' || target.closest('#amp-btn-i18n-folder')) {
                showToast('i18n: ' + (i18nPath || i18nLocalPath), 'info'); return;
            }
        });

        document.addEventListener('change', function(e) {
            if (e.target.id === 'amp-language-override') {
                var sel = e.target.value;
                (async function() { showToast('Switching language...', 'info'); await switchLanguage(sel === 'auto' ? '' : sel); showToast('Language switched!', 'success'); })();
                return;
            }
            if (e.target.closest('#amp-file-upload')) {
                var files = e.target.files; if (!files.length) return;
                (async function() {
                    var pD = document.getElementById('amp-upload-progress'); var fD = document.getElementById('amp-upload-fill'); var tD = document.getElementById('amp-upload-text');
                    if (pD) pD.style.display = 'block';
                    for (var i = 0; i < files.length; i++) {
                        if (fD) fD.style.width = ((i / files.length) * 100).toFixed(0) + '%';
                        if (tD) tD.textContent = t('documents_uploading', {filename: files[i].name, current: i+1, total: files.length});
                        try { await BackendAPI.ingestDocument(files[i]); }
                        catch (err) { showToast(t('documents_upload_failed', {filename: files[i].name}) + ' ' + err.message, 'error'); }
                    }
                    if (fD) fD.style.width = '100%'; if (tD) tD.textContent = t('documents_upload_complete');
                    setTimeout(function() { if (pD) pD.style.display = 'none'; }, 3000);
                    updateStatus(); e.target.value = '';
                })(); return;
            }
            var toggle = e.target.closest('.amp-module-toggle');
            if (toggle) {
                var mn = toggle.dataset.module; var en = toggle.checked;
                (async function() {
                    try { await BackendAPI.toggleModule(mn, en); showToast(mn + ' ' + (en ? t('modules_enabled') : t('modules_disabled')), 'info'); }
                    catch (err) { showToast(t('modules_toggle_failed'), 'error'); toggle.checked = !en; }
                })(); return;
            }
            if (e.target.closest('.amp-doc-checkbox')) {
                var dBtn = document.getElementById('amp-btn-delete-selected');
                if (dBtn) { var cnt = document.querySelectorAll('.amp-doc-checkbox:checked').length; dBtn.style.display = cnt > 0 ? 'inline-block' : 'none'; dBtn.textContent = t('browser_delete_selected') + ' (' + cnt + ')'; }
            }
        });
    }

    // ===================== CHUNK EVENT BINDING =====================

    function bindChunkEvents() {
        document.addEventListener('click', function(e) {
            var target = e.target;

            if (target.id === 'amp-btn-detect-model' || target.closest('#amp-btn-detect-model')) {
                (async function() {
                    var pi = document.getElementById('amp-model-path');
                    var mp = pi ? pi.value.trim() : '';
                    if (!mp) { showToast('Please enter a model path', 'warning'); return; }
                    if (mp.indexOf('..') !== -1) { showToast('Invalid path: ".." not allowed', 'error'); return; }
                    setLoading('amp-btn-detect-model', true);
                    try {
                        var r = await BackendAPI.request('/chunks/model/switch', 'POST', { model_path: mp });
                        handleModelSwitchResult(r); updateChunkOverview();
                    } catch (err) { showToast('Model detection failed: ' + err.message, 'error'); }
                    finally { setLoading('amp-btn-detect-model', false); }
                })(); return;
            }

            if (target.id === 'amp-btn-restore-all' || target.closest('#amp-btn-restore-all')) {
                (async function() {
                    setLoading('amp-btn-restore-all', true);
                    try {
                        var r = await BackendAPI.request('/chunks/restore', 'POST', {});
                        if (r.documents && r.documents.length > 0) {
                            await BackendAPI.storeMemory({
                                character: 'restored',
                                messages: r.documents.map(function(d) { return { role: d.role || 'assistant', name: d.name || '', content: d.content || d.text || '', restored: true }; }),
                                auto_stored: false
                            });
                        }
                        showToast(t('chunks_restore_success', { chunks: r.restored_chunks.length, docs: r.total_documents }), 'success');
                        updateChunkOverview(); updateStatus();
                    } catch (err) { showToast('Restore failed: ' + err.message, 'error'); }
                    finally { setLoading('amp-btn-restore-all', false); }
                })(); return;
            }

            if (target.id === 'amp-btn-chunk-history' || target.closest('#amp-btn-chunk-history')) {
                (async function() {
                    var hd = document.getElementById('amp-chunk-history-list');
                    if (!hd) return;
                    if (hd.style.display !== 'none') { hd.style.display = 'none'; return; }
                    try {
                        var r = await BackendAPI.request('/chunks/history?limit=50');
                        if (r.history && r.history.length > 0) {
                            hd.innerHTML = r.history.reverse().map(function(entry) {
                                var ts = new Date(entry.timestamp).toLocaleString();
                                var op = entry.operation || 'unknown';
                                var details = '';
                                if (entry.details) { details = typeof entry.details === 'string' ? entry.details : Object.keys(entry.details).map(function(k) { return k + ': ' + entry.details[k]; }).join(' | '); }
                                var oc = op.includes('fail') ? '#ff6666' : op.includes('restore') ? '#66ccff' : op.includes('switch') ? '#ffcc44' : '#88cc88';
                                return '<div style="border-left:3px solid ' + oc + ';padding:3px 8px;margin:2px 0;font-size:0.85em;"><span style="color:#aaa;">' + escapeHtml(ts) + '</span> <span style="color:' + oc + ';font-weight:bold;">' + escapeHtml(op) + '</span>' + (details ? '<br><span style="color:#999;">' + escapeHtml(details) + '</span>' : '') + '</div>';
                            }).join('');
                        } else { hd.innerHTML = '<i>No history</i>'; }
                        hd.style.display = 'block';
                    } catch (err) { showToast('History failed: ' + err.message, 'error'); }
                })(); return;
            }

            if (target.id === 'amp-btn-auto-restore' || target.closest('#amp-btn-auto-restore')) {
                var rb = document.getElementById('amp-btn-restore-all'); if (rb) rb.click();
                var wd = document.getElementById('amp-model-switch-warning'); if (wd) wd.style.display = 'none';
                return;
            }

            var selBtn = target.classList.contains('amp-lora-select-btn') ? target : target.closest('.amp-lora-select-btn');
            if (selBtn) {
                var lid = selBtn.dataset.loraId;
                (async function() {
                    try {
                        var r = await BackendAPI.request('/chunks/lora/select', 'POST', { lora_id: lid });
                        showToast(r.compatible ? t('chunks_lora_selected', {name: lid}) : t('chunks_lora_incompatible'), r.compatible ? 'success' : 'warning');
                    } catch (err) { showToast('LoRA selection failed: ' + err.message, 'error'); }
                })(); return;
            }

            var delLBtn = target.classList.contains('amp-lora-delete-btn') ? target : target.closest('.amp-lora-delete-btn');
            if (delLBtn) {
                var dlid = delLBtn.dataset.loraId; var dln = delLBtn.dataset.loraName || dlid;
                if (!confirm(t('chunks_confirm_delete_lora', {name: dln}))) return;
                var delFiles = confirm(t('chunks_confirm_delete_lora_files'));
                (async function() {
                    try { await BackendAPI.request('/chunks/lora/' + encodeURIComponent(dlid) + '/delete', 'POST', { delete_files: delFiles }); showToast('LoRA deleted', 'success'); updateChunkOverview(); }
                    catch (err) { showToast('Delete failed: ' + err.message, 'error'); }
                })(); return;
            }

            var restBtn = target.classList.contains('amp-chunk-restore-btn') ? target : target.closest('.amp-chunk-restore-btn');
            if (restBtn) {
                var rcid = restBtn.dataset.chunkId;
                restBtn.disabled = true; restBtn.textContent = '⏳';
                (async function() {
                    try {
                        var r = await BackendAPI.request('/chunks/restore', 'POST', { chunk_ids: [rcid] });
                        if (r.documents && r.documents.length > 0) {
                            await BackendAPI.storeMemory({
                                character: 'restored',
                                messages: r.documents.map(function(d) { return { role: d.role || 'assistant', name: d.name || '', content: d.content || d.text || '', restored: true }; }),
                                auto_stored: false
                            });
                        }
                        showToast('Chunk restored (' + r.total_documents + ' docs)', 'success');
                        updateChunkOverview(); updateStatus();
                    } catch (err) { showToast('Restore failed: ' + err.message, 'error'); restBtn.disabled = false; restBtn.textContent = '♻️'; }
                })(); return;
            }
        });
    }

    // ===================== CHUNK UI UPDATES =====================

    function handleModelSwitchResult(result) {
        var sd = document.getElementById('amp-chunk-model-status');
        var wd = document.getElementById('amp-model-switch-warning');
        if (!result.changed) {
            if (sd) sd.innerHTML = t('chunks_model_status') + ' <b>' + escapeHtml(result.model ? result.model.friendly_name : '') + '</b> ' + t('chunks_model_same');
            if (wd) wd.style.display = 'none';
            return;
        }
        if (sd) sd.innerHTML = t('chunks_model_status') + ' <b>' + escapeHtml(result.model_info ? result.model_info.friendly_name : result.new_model) + '</b> <span style="color:#ffcc00;">' + t('chunks_model_changed') + '</span>';
        if (wd) {
            var wh = '<div style="color:#ffcc00;font-weight:bold;">' + t('chunks_model_switch_warning', { unusable: result.unusable_lora_count||0, restorable: result.restorable_chunk_count||0, docs: result.restorable_document_count||0 }) + '</div>';
            if (result.restorable_chunk_count > 0) wh += '<div style="margin-top:5px;"><button id="amp-btn-auto-restore" class="menu_button" style="background:#446600;">♻️ ' + t('chunks_restore_all') + ' (' + (result.restorable_document_count||0) + ' docs)</button></div>';
            wd.innerHTML = wh; wd.style.display = 'block';
        }
    }

    async function updateChunkOverview() {
        try {
            var results = await Promise.allSettled([
                BackendAPI.request('/chunks/overview'),
                BackendAPI.request('/chunks/restorable'),
                BackendAPI.request('/chunks/lora/compatible'),
                BackendAPI.request('/chunks/lora/list'),
                BackendAPI.request('/chunks/model/known')
            ]);

            var overview = results[0].status === 'fulfilled' ? results[0].value : null;
            var restorable = results[1].status === 'fulfilled' ? results[1].value : null;
            var compatLoras = results[2].status === 'fulfilled' ? results[2].value : null;
            var allLoras = results[3].status === 'fulfilled' ? results[3].value : null;
            var knownModels = results[4].status === 'fulfilled' ? results[4].value : null;

            if (overview) {
                var od = document.getElementById('amp-chunk-overview');
                if (od) {
                    var st = overview.chunks_by_status || {};
                    od.innerHTML = t('chunks_overview') + ': ' + t('chunks_pending') + ': ' + (st.pending||0) + ' | ' + t('chunks_trained') + ': ' + (st.trained||0) + ' | ' + t('chunks_failed') + ': ' + (st.failed||0) + ' | ' + t('chunks_restored') + ': ' + (st.restored||0) + ' | LoRAs: ' + (overview.compatible_loras||0) + '/' + (overview.total_loras||0);
                }
                var ms = document.getElementById('amp-chunk-model-status');
                if (ms && overview.current_model_name) ms.innerHTML = t('chunks_model_status') + ' <b>' + escapeHtml(overview.current_model_name) + '</b>';
            }

            if (restorable) {
                var ri = document.getElementById('amp-restore-info'); var rab = document.getElementById('amp-btn-restore-all'); var rl = document.getElementById('amp-restore-list');
                if (ri) {
                    if (restorable.total > 0) {
                        ri.innerHTML = '♻️ ' + t('chunks_restorable', { count: restorable.total, docs: restorable.total_documents });
                        if (rab) rab.style.display = 'inline-block';
                        if (rl) {
                            rl.innerHTML = restorable.chunks.map(function(ch) {
                                var reason = ch.restore_reason === 'model_mismatch' ? '(model mismatch)' : ch.restore_reason === 'failed' ? '(failed)' : '';
                                return '<div style="display:flex;justify-content:space-between;align-items:center;padding:3px 5px;border-bottom:1px solid #333;"><span>' + escapeHtml(ch.chunk_id) + ' - ' + (ch.document_count||0) + ' docs <span style="color:#aaa;font-size:0.85em;">' + reason + '</span></span><button class="amp-chunk-restore-btn menu_button" data-chunk-id="' + escapeAttr(ch.chunk_id) + '" style="padding:2px 8px;">♻️</button></div>';
                            }).join('');
                            rl.style.display = 'block';
                        }
                    } else {
                        ri.innerHTML = t('chunks_restore_none');
                        if (rab) rab.style.display = 'none'; if (rl) rl.style.display = 'none';
                    }
                }
            }

            if (compatLoras) {
                var cl = document.getElementById('amp-lora-compatible-list');
                if (cl) {
                    if (compatLoras.loras && compatLoras.loras.length > 0) {
                        cl.innerHTML = '<div style="font-size:0.9em;color:#88cc88;margin-bottom:3px;">' + t('chunks_lora_compatible') + ' ' + compatLoras.loras.length + '</div>' +
                            compatLoras.loras.map(function(l) {
                                return '<div style="display:flex;justify-content:space-between;align-items:center;padding:3px 5px;border:1px solid #444;border-radius:4px;margin:2px 0;">' +
                                    '<div>' + (l.status === 'active' ? '🟢' : '🟡') + ' <b>' + escapeHtml(l.lora_id) + '</b><span style="color:#aaa;font-size:0.85em;"> | ' + escapeHtml(l.model_name || '') + ' | ' + (l.created_at || '').substring(0,10) + '</span></div>' +
                                    '<div style="display:flex;gap:3px;"><button class="amp-lora-select-btn menu_button" data-lora-id="' + escapeAttr(l.lora_id) + '" style="padding:2px 6px;">' + t('chunks_lora_select') + '</button>' +
                                    '<button class="amp-lora-delete-btn menu_button" data-lora-id="' + escapeAttr(l.lora_id) + '" data-lora-name="' + escapeAttr(l.lora_id) + '" style="padding:2px 6px;color:#ff6666;">🗑️</button></div></div>';
                            }).join('');
                    } else { cl.innerHTML = '<div style="color:#888;font-size:0.9em;">' + t('chunks_lora_none') + '</div>'; }
                }
            }

            if (allLoras) {
                var ul = document.getElementById('amp-lora-unusable-list');
                if (ul) {
                    var unusable = (allLoras.loras || []).filter(function(l) { return l.status === 'unusable'; });
                    if (unusable.length > 0) {
                        ul.innerHTML = '<div style="font-size:0.9em;color:#cc8844;margin-bottom:3px;">' + t('chunks_lora_unusable') + ' ' + unusable.length + '</div>' +
                            unusable.map(function(l) {
                                return '<div style="display:flex;justify-content:space-between;align-items:center;padding:3px 5px;border:1px solid #553300;border-radius:4px;margin:2px 0;opacity:0.7;">' +
                                    '<div>⚠️ <b>' + escapeHtml(l.lora_id) + '</b><span style="color:#aaa;font-size:0.85em;"> | ' + escapeHtml(l.model_name || 'unknown') + '</span></div>' +
                                    '<button class="amp-lora-delete-btn menu_button" data-lora-id="' + escapeAttr(l.lora_id) + '" data-lora-name="' + escapeAttr(l.lora_id) + '" style="padding:2px 6px;color:#ff6666;">🗑️</button></div>';
                            }).join('');
                    } else { ul.innerHTML = ''; }
                }
            }

            if (knownModels) {
                var kml = document.getElementById('amp-known-models-list');
                if (kml) {
                    if (knownModels.models && knownModels.models.length > 0) {
                        kml.innerHTML = knownModels.models.map(function(m) {
                            var ct = m.is_current ? ' <span style="color:#88cc88;">' + t('chunks_model_current_tag') + '</span>' : '';
                            return '<div style="padding:3px 5px;border-left:3px solid ' + (m.is_current ? '#88cc88' : '#555') + ';margin:2px 0;"><b>' + escapeHtml(m.name) + '</b>' + ct + '<span style="color:#aaa;font-size:0.85em;"> | ' + escapeHtml(m.model_type || 'unknown') + ' | LoRAs: ' + (m.lora_count||0) + ' | Used: ' + (m.times_used||0) + 'x</span></div>';
                        }).join('');
                    } else { kml.innerHTML = '<i style="color:#888;">No models registered</i>'; }
                }
            }
        } catch (e) {
            console.debug('[' + PLUGIN_NAME + '] Chunk overview error:', e.message);
        }
    }

    // ===================== BROWSER HELPERS =====================

    function displayBrowseResults(result, offset) {
        var container = document.getElementById('amp-browse-results'); var list = document.getElementById('amp-browse-list');
        var pageInfo = document.getElementById('amp-browse-page-info'); var prevBtn = document.getElementById('amp-btn-browse-prev');
        var nextBtn = document.getElementById('amp-btn-browse-next'); var deleteBtn = document.getElementById('amp-btn-delete-selected');
        if (!container || !list) return;
        container.style.display = 'block'; if (prevBtn) prevBtn.style.display = ''; if (nextBtn) nextBtn.style.display = '';

        if (!result.documents || result.documents.length === 0) { list.innerHTML = '<i>' + t('browser_no_documents') + '</i>'; if (deleteBtn) deleteBtn.style.display = 'none'; return; }

        var html = '<div style="margin-bottom:5px;"><label><input type="checkbox" id="amp-select-all-docs"> ' + t('browser_select_all') + '</label> | ' + t('browser_total') + ' ' + result.total + ' ' + t('browser_documents') + '</div>';
        result.documents.forEach(function(doc) {
            var tag = doc.auto_stored ? ' <span style="color:#888;font-size:0.8em;">' + t('browser_tag_auto') + '</span>' : ' <span style="color:#4CAF50;font-size:0.8em;">' + t('browser_tag_manual') + '</span>';
            html += '<div style="border:1px solid #444;border-radius:4px;padding:5px;margin:3px 0;"><label style="display:flex;align-items:flex-start;gap:5px;"><input type="checkbox" class="amp-doc-checkbox" data-doc-id="' + escapeAttr(doc.id) + '"><div style="flex:1;"><div style="font-size:0.85em;color:#aaa;">' + escapeHtml(doc.timestamp_formatted) + ' | ' + escapeHtml(doc.character) + ' | ' + escapeHtml(doc.role) + tag + '</div><div style="font-size:0.9em;margin-top:2px;">' + escapeHtml(doc.text_preview) + '</div></div></label></div>';
        });
        list.innerHTML = html;
        var pg = Math.floor(offset/50)+1; var tp = Math.ceil(result.total/50);
        if (pageInfo) { pageInfo.textContent = t('browser_page') + ' ' + pg + '/' + tp; pageInfo.dataset.offset = offset; }
        if (prevBtn) prevBtn.disabled = offset <= 0;
        if (nextBtn) nextBtn.disabled = !result.has_more;
        if (deleteBtn) deleteBtn.style.display = 'none';
    }

    function displayHistoryResults(history) {
        var container = document.getElementById('amp-browse-results'); var list = document.getElementById('amp-browse-list');
        if (!container || !list) return; container.style.display = 'block';
        var pb = document.getElementById('amp-btn-browse-prev'); var nb = document.getElementById('amp-btn-browse-next'); var pi = document.getElementById('amp-browse-page-info');
        if (pb) pb.style.display = 'none'; if (nb) nb.style.display = 'none'; if (pi) pi.textContent = '';
        if (!history || history.length === 0) { list.innerHTML = '<i>' + t('browser_no_history') + '</i>'; return; }

        var html = '<div style="margin-bottom:5px;"><b>' + t('browser_recent_history') + ' (' + history.length + ' ' + t('browser_entries') + ')</b></div>';
        history.forEach(function(entry) {
            var tag = entry.auto_stored ? ' <span style="color:#888;font-size:0.8em;">' + t('browser_tag_auto') + '</span>' : ' <span style="color:#4CAF50;font-size:0.8em;">' + t('browser_tag_manual') + '</span>';
            var rc = entry.role === 'user' ? '#64B5F6' : '#81C784';
            html += '<div style="border-left:3px solid ' + rc + ';padding:3px 8px;margin:3px 0;"><div style="font-size:0.85em;color:#aaa;">' + escapeHtml(entry.timestamp_formatted) + ' | ' + escapeHtml(entry.character) + ' | <span style="color:' + rc + ';">' + escapeHtml(entry.name) + ' (' + escapeHtml(entry.role) + ')</span>' + tag + '</div><div style="font-size:0.9em;margin-top:2px;">' + escapeHtml(entry.text_preview) + '</div></div>';
        });
        list.innerHTML = html;
    }

    async function refreshCharacterList() {
        try {
            var chars = await BackendAPI.getCharacters(); var sel = document.getElementById('amp-browse-character');
            if (!sel) return; var cv = sel.value;
            sel.innerHTML = '<option value="">' + t('browser_all_characters') + '</option>';
            chars.characters.forEach(function(c) { sel.innerHTML += '<option value="' + escapeAttr(c.name) + '">' + escapeHtml(c.name) + ' (' + c.total + ')</option>'; });
            sel.value = cv;
        } catch (e) { /* ignore */ }
    }

    // ===================== FEEDBACK =====================

    function toggleFeedbackButtons(enable) {
        document.querySelectorAll('.mes').forEach(function(msg) {
            var existing = msg.querySelector('.amp-feedback-btns');
            if (enable) {
                if (existing) return;
                if (msg.getAttribute('is_user') === 'true' || msg.getAttribute('is_system') === 'true') return;
                var mesId = msg.getAttribute('mesid'); if (!mesId) return;
                var bc = document.createElement('div'); bc.className = 'amp-feedback-btns'; bc.style.cssText = 'display:flex;gap:5px;margin-top:5px;padding:3px;';
                bc.innerHTML = '<button class="amp-fb-btn amp-fb-positive menu_button" title="' + escapeAttr(t('rlhf_tooltip_good')) + '" style="padding:2px 8px;">' + t('rlhf_btn_good') + '</button><button class="amp-fb-btn amp-fb-negative menu_button" title="' + escapeAttr(t('rlhf_tooltip_bad')) + '" style="padding:2px 8px;">' + t('rlhf_btn_bad') + '</button><button class="amp-fb-btn amp-fb-excellent menu_button" title="' + escapeAttr(t('rlhf_tooltip_excellent')) + '" style="padding:2px 8px;">' + t('rlhf_btn_excellent') + '</button>';
                bc.querySelector('.amp-fb-positive').addEventListener('click', function(e) { e.stopPropagation(); submitMessageFeedback(mesId, 'positive', msg); });
                bc.querySelector('.amp-fb-negative').addEventListener('click', function(e) { e.stopPropagation(); submitMessageFeedback(mesId, 'negative', msg); });
                bc.querySelector('.amp-fb-excellent').addEventListener('click', function(e) { e.stopPropagation(); submitMessageFeedback(mesId, 'excellent', msg); });
                (msg.querySelector('.mes_block') || msg.querySelector('.mes_text') || msg).appendChild(bc);
            } else { if (existing) existing.remove(); }
        });
    }

    function submitMessageFeedback(messageId, rating, msgElement) {
        var mt = msgElement.querySelector('.mes_text'); var txt = mt ? mt.textContent : '';
        var prev = msgElement.previousElementSibling; var pt = prev ? prev.querySelector('.mes_text') : null; var prompt = pt ? pt.textContent : '';
        (async function() {
            try {
                await BackendAPI.submitFeedback({ message_id: messageId, rating: rating, response_text: txt, prompt_text: prompt, character: getCurrentCharacter(), timestamp: Date.now() });
                var btns = msgElement.querySelector('.amp-feedback-btns');
                if (btns) { var icon = rating === 'positive' ? t('rlhf_btn_good') : rating === 'negative' ? t('rlhf_btn_bad') : t('rlhf_btn_excellent'); btns.innerHTML = '<span class="amp-fb-submitted">' + icon + ' ' + t('rlhf_feedback_recorded') + '</span>'; }
                updateFeedbackStats();
            } catch (err) { showToast(t('rlhf_feedback_failed'), 'error'); }
        })();
    }

    // ===================== TRAINING =====================

    var trainingPollInterval = null;

    function pollTrainingProgress() {
        if (trainingPollInterval) clearInterval(trainingPollInterval);
        trainingPollInterval = setInterval(function() {
            (async function() {
                try {
                    var p = await BackendAPI.getTrainingProgress();
                    var f = document.getElementById('amp-train-fill'); var s = document.getElementById('amp-train-status-text'); var et = document.getElementById('amp-train-eta');
                    var pct = p.percentage || p.progress || 0;
                    if (f) f.style.width = pct + '%';
                    if (s) s.textContent = t('training_status', { current_epoch: p.current_epoch||0, total_epochs: p.total_epochs||0, current_step: p.current_step||0, total_steps: p.total_steps||0, loss: (p.loss||0).toFixed(4) });
                    if (et) et.textContent = t('training_eta', {eta: p.eta || '...'});
                    if (p.status === 'completed') { clearInterval(trainingPollInterval); showToast(t('training_completed'), 'success'); resetTrainingUI(); updateStatus(); updateChunkOverview(); }
                    else if (p.status === 'failed') { clearInterval(trainingPollInterval); showToast(t('training_failed') + ' ' + (p.error || ''), 'error'); resetTrainingUI(); }
                } catch (err) { /* poll error */ }
            })();
        }, 2000);
    }

    function resetTrainingUI() {
        if (trainingPollInterval) clearInterval(trainingPollInterval);
        var p = document.getElementById('amp-training-progress'); var s = document.getElementById('amp-btn-start-training'); var c = document.getElementById('amp-btn-cancel-training');
        if (p) p.style.display = 'none'; if (s) s.style.display = 'inline-block'; if (c) c.style.display = 'none';
    }

    // ===================== CHAT EXTRACTION =====================

    function extractCurrentChat() {
        var messages = [];
        document.querySelectorAll('.mes').forEach(function(msg) {
            var isUser = msg.getAttribute('is_user') === 'true'; var isSys = msg.getAttribute('is_system') === 'true';
            var nEl = msg.querySelector('.ch_name'); var tEl = msg.querySelector('.mes_text'); var mesId = msg.getAttribute('mesid');
            var images = []; msg.querySelectorAll('.mes_img, .mes_text img').forEach(function(img) { images.push(img.src || img.getAttribute('data-src') || ''); });
            messages.push({ id: mesId, role: isSys ? 'system' : (isUser ? 'user' : 'assistant'), name: nEl ? nEl.textContent.trim() : '', content: tEl ? tEl.textContent.trim() : '', images: images, is_system: isSys });
        });
        return { character: getCurrentCharacter(), messages: messages, timestamp: Date.now() };
    }

    function getCurrentCharacter() {
        try { if (typeof SillyTavern !== 'undefined' && SillyTavern.getContext) { var ctx = SillyTavern.getContext(); return ctx.name2 || ctx.characterId || 'Unknown'; } } catch (e) { /* */ }
        var n = document.querySelector('#rm_button_selected_ch .ch_name') || document.querySelector('.selected_chat_block .ch_name');
        return n ? n.textContent.trim() : 'Unknown';
    }

    // ===================== STATUS =====================

    async function updateStatus() {
        if (document.hidden) return;

        try {
            var status = await BackendAPI.getMemoryStatus();
            var dot = document.getElementById('amp-backend-dot'); if (dot) dot.textContent = t('status_connected');
            var mem = document.getElementById('amp-memory-size');
            if (mem) mem.textContent = t('status_vectordb').replace('0 MB', (status.total_size_mb||0).toFixed(1) + ' MB') + ' | ' + (status.document_count||0) + ' ' + t('browser_documents');
            backendAvailable = true;
        } catch (e) {
            var dot2 = document.getElementById('amp-backend-dot'); if (dot2) dot2.textContent = t('status_disconnected');
            backendAvailable = false;
        }

        try {
            var di = await BackendAPI.getDeviceInfo();
            var dt = 'Device: ' + di.device_name + ' (' + di.device_type + ')';
            if (di.gpu_memory_gb) dt += ' | ' + di.gpu_memory_gb + ' GB';
            if (di.is_shared_memory) dt += ' ' + t('device_shared');
            var de = document.getElementById('amp-device-info'); if (de) de.textContent = dt;
        } catch (e) { /* */ }

        try {
            var li = await BackendAPI.getLoraStatus();
            var le = document.getElementById('amp-lora-status');
            if (le) le.textContent = 'LoRA: ' + (li.available ? li.model_name : 'none') + (li.auto_loaded ? ' (loaded)' : '');
        } catch (e) { var le2 = document.getElementById('amp-lora-status'); if (le2) le2.textContent = t('status_lora_none'); }

        updateFeedbackStats();
        updateModulesList();
    }

    async function updateFeedbackStats() {
        try {
            var s = await BackendAPI.getFeedbackStats();
            var el = document.getElementById('amp-feedback-stats');
            if (el) el.textContent = t('rlhf_stats', { positive: s.positive||0, negative: s.negative||0, excellent: s.excellent||0, total: s.total||0 });
        } catch (e) { /* */ }
    }

    async function updateModulesList() {
        try {
            var m = await BackendAPI.getModules(); var c = document.getElementById('amp-modules-list');
            if (!c) return;
            if (!m.modules || !m.modules.length) { c.innerHTML = '<i>' + t('modules_none') + '</i>'; return; }
            c.innerHTML = m.modules.map(function(mod) {
                return '<div class="amp-module-item"><label><input type="checkbox" class="amp-module-toggle" data-module="' + escapeAttr(mod.name) + '"' + (mod.enabled ? ' checked' : '') + '><b>' + escapeHtml(mod.display_name || mod.name) + '</b></label><span class="amp-module-desc">' + escapeHtml(mod.description || '') + '</span></div>';
            }).join('');
        } catch (e) { /* */ }
    }

    // ===================== AUTO-MEMORY =====================

    function interceptChatMessages() {
        var chat = document.getElementById('chat');
        if (!chat) { setTimeout(interceptChatMessages, 2000); return; }
        var observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                mutation.addedNodes.forEach(function(node) {
                    if (node.nodeType === 1 && node.classList && node.classList.contains('mes')) {
                        autoStoreMessage(node);
                        if (feedbackMode) setTimeout(function() { toggleFeedbackButtons(true); }, 100);
                    }
                });
            });
        });
        observer.observe(chat, { childList: true });
    }

    function autoStoreMessage(msgNode) {
        if (!autoStoreEnabled || !backendAvailable) return;
        var isUser = msgNode.getAttribute('is_user') === 'true';
        var tEl = msgNode.querySelector('.mes_text'); var text = tEl ? tEl.textContent.trim() : '';
        var nEl = msgNode.querySelector('.ch_name'); var name = nEl ? nEl.textContent.trim() : '';
        var mesId = msgNode.getAttribute('mesid');
        if (!text || !mesId || text.length < 20) return;
        if (recentlyStored[mesId]) return;

        var contentHash = computeContentHash(text);
        var dedupKey = mesId + '_' + contentHash;
        if (recentlyStored[dedupKey]) return;

        setTimeout(function() {
            var ftEl = msgNode.querySelector('.mes_text'); var finalText = ftEl ? ftEl.textContent.trim() : '';
            if (!finalText || finalText.length < 20) return;

            var finalHash = computeContentHash(finalText);
            var finalDedupKey = mesId + '_' + finalHash;
            if (recentlyStored[finalDedupKey]) return;

            recentlyStored[mesId] = true;
            recentlyStored[finalDedupKey] = true;

            var keys = Object.keys(recentlyStored);
            if (keys.length > 200) { for (var i = 0; i < 100; i++) delete recentlyStored[keys[i]]; }

            (async function() {
                try {
                    await BackendAPI.storeMemory({ character: getCurrentCharacter(), messages: [{ role: isUser ? 'user' : 'assistant', name: name, content: finalText, images: [], timestamp: Date.now() }], auto_stored: true });

                    try {
                        await BackendAPI.request('/chunks/create', 'POST', {
                            documents: [{ role: isUser ? 'user' : 'assistant', name: name, content: finalText, text: finalText }],
                            character: getCurrentCharacter(),
                            metadata: { source: 'auto_store', mesid: mesId }
                        });
                    } catch (ce) { /* chunk system optional */ }

                    console.debug('[' + PLUGIN_NAME + '] Auto-stored mesid ' + mesId);
                } catch (e) { console.debug('[' + PLUGIN_NAME + '] Auto-store failed:', e.message); }
            })();
        }, 3000);
    }

    // ===================== RAG INJECTION =====================

    function interceptPromptGeneration() {
        if (typeof eventSource !== 'undefined') {
            try {
                eventSource.on('generate_before_combine_prompts', function(data) {
                    if (!ragInjectionEnabled || !backendAvailable) return;
                    (async function() {
                        try {
                            var userMsgs = data.messages ? data.messages.filter(function(m) { return m.role === 'user'; }) : [];
                            var lastMsg = userMsgs[userMsgs.length - 1]; if (!lastMsg) return;

                            var results = await BackendAPI.queryMemory(lastMsg.content, 3);
                            if (results.results && results.results.length > 0) {
                                var msEl = document.getElementById('amp-rag-min-score'); var minScore = msEl ? parseFloat(msEl.value) || 0.3 : 0.3;
                                var mcEl = document.getElementById('amp-rag-max-chars'); var maxChars = mcEl ? parseInt(mcEl.value) || 1500 : 1500;

                                var relevant = results.results.filter(function(r) { return r.score >= minScore; });
                                if (relevant.length === 0) return;

                                var totalChars = 0; var selected = [];
                                for (var i = 0; i < relevant.length; i++) {
                                    var rText = relevant[i].text;
                                    if (totalChars + rText.length > maxChars) { var rem = maxChars - totalChars; if (rem > 100) selected.push(rText.substring(0, rem) + '...'); break; }
                                    selected.push(rText); totalChars += rText.length;
                                }
                                if (selected.length === 0) return;

                                var memCtx = selected.join('\n---\n');
                                var prompt = '[OOC: Background Memory - The following are relevant memories from previous interactions. Use these ONLY as passive background knowledge to maintain consistency. Do NOT directly reference, quote, summarize, or respond to these memories unless the user explicitly asks about them. Do NOT extend your response length because of this context. Respond naturally to the user\'s latest message only, at your normal response length.\n\n' + memCtx + '\nEnd of background memory. Respond ONLY to the latest message.]';

                                if (data.systemPrompt) data.systemPrompt += '\n' + prompt;
                                console.debug('[' + PLUGIN_NAME + '] Injected ' + selected.length + ' memories (' + totalChars + ' chars)');
                            }
                        } catch (e) { console.debug('[' + PLUGIN_NAME + '] RAG skipped:', e.message); }
                    })();
                });
            } catch (e) { console.debug('[' + PLUGIN_NAME + '] Event hook not available.'); }
        }
    }

    // ===================== UTILITY =====================

    function showToast(message, type) {
        type = type || 'info';
        if (typeof toastr !== 'undefined') { if (toastr[type]) toastr[type](message); else toastr.info(message); return; }
        var toast = document.createElement('div'); toast.className = 'amp-toast amp-toast-' + type; toast.textContent = message;
        document.body.appendChild(toast);
        setTimeout(function() { toast.classList.add('amp-toast-show'); }, 10);
        setTimeout(function() { toast.classList.remove('amp-toast-show'); setTimeout(function() { toast.remove(); }, 300); }, 3000);
    }

    function escapeHtml(text) { var d = document.createElement('div'); d.textContent = text; return d.innerHTML; }

    // ===================== VISIBILITY-AWARE POLLING =====================

    var statusInterval = null;
    var chunkInterval = null;

    function startPolling() {
        if (statusInterval) clearInterval(statusInterval);
        if (chunkInterval) clearInterval(chunkInterval);
        statusInterval = setInterval(updateStatus, 30000);
        chunkInterval = setInterval(function() { if (!document.hidden) updateChunkOverview(); }, 60000);
    }

    document.addEventListener('visibilitychange', function() {
        if (!document.hidden && backendAvailable) {
            updateStatus();
            updateChunkOverview();
        }
    });

    // ===================== INITIALIZATION =====================

    async function init() {
        if (pluginInitialized) return;
        pluginInitialized = true;

        await loadI18n();
        console.log('[' + PLUGIN_NAME + '] ' + t('console_initializing'));

        var backendReady = await checkBackend();

        injectUI();
        bindEvents();
        bindChunkEvents();
        console.log('[' + PLUGIN_NAME + '] ' + t('console_handlers_registered'));

        var settings = loadSettings();
        if (settings.feedbackMode) { feedbackMode = true; var fb = document.getElementById('amp-btn-toggle-feedback'); if (fb) fb.textContent = t('rlhf_toggle_on'); setTimeout(function() { toggleFeedbackButtons(true); }, 3000); }
        if (settings.autoStoreEnabled === false) { autoStoreEnabled = false; var as = document.getElementById('amp-btn-toggle-autostore'); if (as) as.textContent = t('memory_autostore_off'); }
        if (settings.ragInjectionEnabled === false) { ragInjectionEnabled = false; var rg = document.getElementById('amp-btn-toggle-rag'); if (rg) rg.textContent = t('memory_rag_off'); }

        interceptChatMessages();
        interceptPromptGeneration();

        if (backendReady) {
            setTimeout(updateStatus, 1000);
            setTimeout(refreshCharacterList, 2000);
            setTimeout(updateChunkOverview, 3000);
        }

        startPolling();
        console.log('[' + PLUGIN_NAME + '] ' + t('console_initialized'));
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() { setTimeout(init, 2000); });
    } else {
        setTimeout(init, 2000);
    }
})();