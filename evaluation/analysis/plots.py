"""Dependency-free SVG comparison chart for the final report."""
from __future__ import annotations

from pathlib import Path


def write_recall_chart(summary, target: Path):
    labels = list(summary)
    width, height, margin = 900, 420, 70
    plot_width = width - 2 * margin
    bar_width = plot_width / max(1, len(labels)) * .6
    colors = ('#2563eb', '#7c3aed', '#059669', '#dc2626')
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="450" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">Recall@5 ablation</text>',
             f'<line x1="{margin}" y1="350" x2="{width-margin}" y2="350" stroke="#334155"/>']
    for index, label in enumerate(labels):
        value = summary[label]['recall_at_5']['mean'] or 0
        x = margin + (index + .2) * plot_width / len(labels)
        h = value * 280
        parts += [f'<rect x="{x:.1f}" y="{350-h:.1f}" width="{bar_width:.1f}" height="{h:.1f}" fill="{colors[index % len(colors)]}"/>',
                  f'<text x="{x+bar_width/2:.1f}" y="{340-h:.1f}" text-anchor="middle" font-family="sans-serif" font-size="14">{value:.1%}</text>',
                  f'<text x="{x+bar_width/2:.1f}" y="375" text-anchor="middle" font-family="sans-serif" font-size="12">{label.replace("_", "+")}</text>']
    parts.append('</svg>')
    target.write_text('\n'.join(parts) + '\n')
