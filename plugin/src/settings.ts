import { App, PluginSettingTab, Setting } from "obsidian";
import type JournalGraphPlugin from "../main";

export type JournalGraphSettings = {
  /** Vault-relative folder holding journal notes (used by Timeline + export). */
  journalFolder: string;
  /** Full URL of the extraction endpoint. */
  serverUrl: string;
  /** If true, Timeline/export walk every markdown note in the vault. */
  treatAllAsJournal: boolean;
  /** Re-extract the current note a few seconds after typing stops. */
  autoRefresh: boolean;
  autoRefreshDelayMs: number;
};

export const DEFAULT_SETTINGS: JournalGraphSettings = {
  journalFolder: "journal",
  serverUrl: "http://localhost:8000/extract",
  treatAllAsJournal: false,
  autoRefresh: true,
  autoRefreshDelayMs: 1500,
};

export class JournalGraphSettingTab extends PluginSettingTab {
  constructor(
    app: App,
    private plugin: JournalGraphPlugin,
  ) {
    super(app, plugin);
  }

  display(): void {
    const { containerEl } = this;
    containerEl.empty();

    new Setting(containerEl)
      .setName("Journal folder")
      .setDesc(
        'Vault folder that Timeline and the therapist export read from. Ignored when "Treat all notes as journal entries" is on.',
      )
      .addText((t) =>
        t
          .setPlaceholder(DEFAULT_SETTINGS.journalFolder)
          .setValue(this.plugin.settings.journalFolder)
          .onChange(async (v) => {
            this.plugin.settings.journalFolder = v.trim().replace(/\/+$/, "");
            await this.plugin.saveSettings();
          }),
      );

    new Setting(containerEl)
      .setName("Extraction server URL")
      .setDesc("Endpoint that receives POST {text} and returns the graph.")
      .addText((t) =>
        t
          .setPlaceholder(DEFAULT_SETTINGS.serverUrl)
          .setValue(this.plugin.settings.serverUrl)
          .onChange(async (v) => {
            this.plugin.settings.serverUrl = v.trim() || DEFAULT_SETTINGS.serverUrl;
            await this.plugin.saveSettings();
          }),
      );

    new Setting(containerEl)
      .setName("Treat all notes in vault as journal entries")
      .setDesc("Timeline and export include every markdown note, not just the journal folder.")
      .addToggle((t) =>
        t.setValue(this.plugin.settings.treatAllAsJournal).onChange(async (v) => {
          this.plugin.settings.treatAllAsJournal = v;
          await this.plugin.saveSettings();
        }),
      );

    new Setting(containerEl)
      .setName("Auto-refresh current note")
      .setDesc("Re-extract the note shown in the panel a few seconds after typing stops.")
      .addToggle((t) =>
        t.setValue(this.plugin.settings.autoRefresh).onChange(async (v) => {
          this.plugin.settings.autoRefresh = v;
          await this.plugin.saveSettings();
        }),
      );

    new Setting(containerEl)
      .setName("Auto-refresh delay (seconds)")
      .setDesc("How long to wait after the last keystroke before re-extracting.")
      .addSlider((s) =>
        s
          .setLimits(0.5, 10, 0.5)
          .setValue(this.plugin.settings.autoRefreshDelayMs / 1000)
          .setDynamicTooltip()
          .onChange(async (v) => {
            this.plugin.settings.autoRefreshDelayMs = v * 1000;
            await this.plugin.saveSettings();
          }),
      );
  }
}
