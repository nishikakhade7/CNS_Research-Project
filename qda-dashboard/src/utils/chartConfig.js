import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  RadialLinearScale,
  Filler,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  RadialLinearScale,
  Filler
);

export const defaultChartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      labels: {
        color: '#94a3c8',
        font: {
          family: "'Inter', sans-serif",
          size: 12,
        },
      },
    },
    tooltip: {
      backgroundColor: '#1a1f35',
      titleColor: '#f0f4ff',
      bodyColor: '#94a3c8',
      borderColor: 'rgba(99, 179, 237, 0.3)',
      borderWidth: 1,
      padding: 12,
      cornerRadius: 8,
      boxPadding: 4,
      bodyFont: {
        family: "'JetBrains Mono', monospace",
        size: 12,
      },
      titleFont: {
        family: "'Inter', sans-serif",
        size: 13,
        weight: 'bold',
      },
    },
  },
  scales: {
    x: {
      grid: {
        color: 'rgba(99, 115, 171, 0.08)',
      },
      ticks: {
        color: '#64748b',
        font: {
          family: "'Inter', sans-serif",
          size: 11,
        },
      },
    },
    y: {
      grid: {
        color: 'rgba(99, 115, 171, 0.08)',
      },
      ticks: {
        color: '#64748b',
        font: {
          family: "'JetBrains Mono', monospace",
          size: 11,
        },
      },
    },
  },
};
