import { Chart, registerables, ChartConfiguration } from "chart.js/auto";
import { JournalDay } from "./types";
import { ThemeColors, withAlpha } from "./theme";

Chart.register(...registerables);

export class MoodChart {
  private chart?: Chart;
  private canvas?: HTMLCanvasElement;

  constructor(private host: HTMLElement) {}

  render(days: JournalDay[], selectedIndex: number, theme: ThemeColors) {
    this.host.empty();
    if (!days.length) {
      const note = this.host.createDiv({ cls: "jg-mood-empty" });
      note.setText("No dated notes to plot.");
      return;
    }

    this.canvas = this.host.createEl("canvas");
    this.canvas.height = 80;

    const line = withAlpha(theme.accent, 0.85);
    const fill = withAlpha(theme.accent, 0.15);
    const point = withAlpha(theme.textMuted, 0.8);

    const labels = days.map((d) => d.date.slice(5)); // MM-DD
    const values = days.map((d) => d.mood_score);
    const colors = days.map((_, i) => (i === selectedIndex ? theme.accent : point));
    const radii = days.map((_, i) => (i === selectedIndex ? 6 : 3));

    const config: ChartConfiguration = {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "mood",
            data: values as (number | null)[],
            spanGaps: true,
            borderColor: line,
            backgroundColor: fill,
            tension: 0.25,
            pointBackgroundColor: colors,
            pointRadius: radii,
            pointHoverRadius: 7,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const i = ctx.dataIndex;
                const d = days[i];
                const m = d.mood_score;
                return `${d.date} · mood ${m ?? "—"}`;
              },
            },
          },
        },
        scales: {
          y: {
            min: 0,
            max: 10,
            ticks: { stepSize: 2, color: theme.textMuted },
            grid: { color: withAlpha(theme.textMuted, 0.12) },
          },
          x: {
            grid: { display: false },
            ticks: { autoSkip: true, maxRotation: 0, color: theme.textMuted },
          },
        },
      },
    };

    this.chart?.destroy();
    this.chart = new Chart(this.canvas, config);
  }

  destroy() {
    this.chart?.destroy();
    this.chart = undefined;
    this.host.empty();
  }
}
