import { App, Modal, Setting } from "obsidian";

/**
 * Progress dialog for the therapist export. Closing it by any path (button,
 * Esc, backdrop click) counts as cancel unless markFinished() ran first.
 */
export class ExportProgressModal extends Modal {
  private cancelled = false;
  private finished = false;
  private statusEl!: HTMLElement;
  private fillEl!: HTMLElement;
  onCancel?: () => void;

  constructor(app: App) {
    super(app);
  }

  onOpen() {
    this.titleEl.setText("Export therapist summary");
    this.contentEl.addClass("jg-export-modal");
    this.statusEl = this.contentEl.createDiv({ cls: "jg-export-status", text: "Preparing…" });
    const bar = this.contentEl.createDiv({ cls: "jg-export-bar" });
    this.fillEl = bar.createDiv({ cls: "jg-export-bar-fill" });
    new Setting(this.contentEl).addButton((b) =>
      b.setButtonText("Cancel").onClick(() => this.close()),
    );
  }

  setProgress(done: number, total: number) {
    this.statusEl?.setText(`Day ${done} of ${total}…`);
    if (this.fillEl) this.fillEl.style.width = `${Math.round((100 * done) / Math.max(1, total))}%`;
  }

  markFinished() {
    this.finished = true;
    this.close();
  }

  isCancelled() {
    return this.cancelled;
  }

  onClose() {
    if (!this.finished && !this.cancelled) {
      this.cancelled = true;
      this.onCancel?.();
    }
    this.contentEl.empty();
  }
}
