"""
Generate all figures for the LaTeX report (Chapter 4: System Design).
Outputs: report_figures/ directory with publication-quality plots.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ── Paths ───────────────────────────────────────────────────────
BASE = Path(__file__).parent
JSON_DIR = BASE / "OksidatifStress" / "json_files"
SPERM_DS = BASE / "SpermDataset"
OUT_DIR = BASE / "report_figures"
OUT_DIR.mkdir(exist_ok=True)

# ── Style ───────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

CLASS_NAMES = ['DEG', 'NH', 'SH', 'MH', 'BH']
CLASS_FULL = {
    'DEG': 'Degenerative',
    'NH':  'Normal Head',
    'SH':  'Small Head',
    'MH':  'Medium Head',
    'BH':  'Big Head',
}
COLORS = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6']


def gather_stats():
    """Parse all JSON annotations to gather comprehensive statistics."""
    class_counts = Counter()
    objects_per_image = []
    patient_counts = Counter()
    bbox_areas = []
    class_areas = {c: [] for c in CLASS_NAMES}

    for jf in sorted(JSON_DIR.glob('*.json')):
        data = json.loads(jf.read_text())
        img_w = data.get('imageWidth', 1280)
        img_h = data.get('imageHeight', 720)

        # Patient
        patient = ''
        for ch in jf.stem:
            if ch.isalpha():
                patient += ch
            else:
                break
        patient_counts[patient] += 1

        obj_count = 0
        for shape in data.get('shapes', []):
            lbl = shape.get('label', '')
            if lbl not in CLASS_NAMES:
                continue
            class_counts[lbl] += 1
            obj_count += 1
            pts = shape.get('points', [])
            if len(pts) >= 3:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                bbox_areas.append(area)
                class_areas[lbl].append(area)
        objects_per_image.append(obj_count)

    return class_counts, objects_per_image, patient_counts, bbox_areas, class_areas


def gather_split_stats():
    """Gather per-split class distributions from YOLO labels."""
    split_stats = {}
    for split in ['train', 'val', 'test']:
        label_dir = SPERM_DS / split / "labels"
        counts = Counter()
        for lf in sorted(label_dir.glob('*.txt')):
            content = lf.read_text().strip()
            if not content:
                continue
            for line in content.split('\n'):
                parts = line.strip().split()
                if parts:
                    counts[int(parts[0])] += 1
        split_stats[split] = counts
    return split_stats


# ════════════════════════════════════════════════════════════════
# Figure 1: Overall Class Distribution (Bar + Pie)
# ════════════════════════════════════════════════════════════════
def fig1_class_distribution(class_counts):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    counts = [class_counts[c] for c in CLASS_NAMES]
    total = sum(counts)

    # Bar chart
    bars = ax1.bar(CLASS_NAMES, counts, color=COLORS, edgecolor='white', linewidth=1.2)
    for bar, cnt in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                 f'{cnt}\n({cnt/total*100:.1f}%)', ha='center', va='bottom', fontsize=9)
    ax1.set_xlabel('Morphology Class')
    ax1.set_ylabel('Number of Instances')
    ax1.set_title('(a) Instance Count per Class')
    ax1.set_ylim(0, max(counts) * 1.2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Pie chart
    labels_pie = [f'{c}\n({cnt/total*100:.1f}%)' for c, cnt in zip(CLASS_NAMES, counts)]
    wedges, texts = ax2.pie(counts, labels=labels_pie, colors=COLORS,
                            startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    ax2.set_title('(b) Class Proportion')

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'fig_class_distribution.pdf')
    fig.savefig(OUT_DIR / 'fig_class_distribution.png')
    plt.close(fig)
    print("[OK] fig_class_distribution")


# ════════════════════════════════════════════════════════════════
# Figure 2: Train / Val / Test Split Comparison
# ════════════════════════════════════════════════════════════════
def fig2_split_comparison(split_stats):
    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(CLASS_NAMES))
    width = 0.25
    split_colors = {'train': '#2ecc71', 'val': '#3498db', 'test': '#e74c3c'}

    for i, (split, color) in enumerate(split_colors.items()):
        counts = [split_stats[split].get(j, 0) for j in range(5)]
        bars = ax.bar(x + i * width, counts, width, label=f'{split.capitalize()}',
                      color=color, edgecolor='white', linewidth=0.8, alpha=0.85)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    str(cnt), ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Morphology Class')
    ax.set_ylabel('Number of Instances')
    ax.set_title('Class Distribution Across Dataset Splits')
    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASS_NAMES)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'fig_split_comparison.pdf')
    fig.savefig(OUT_DIR / 'fig_split_comparison.png')
    plt.close(fig)
    print("[OK] fig_split_comparison")


# ════════════════════════════════════════════════════════════════
# Figure 3: Objects per Image Histogram
# ════════════════════════════════════════════════════════════════
def fig3_objects_per_image(objects_per_image):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(objects_per_image, bins=30, color='#3498db', edgecolor='white',
            linewidth=0.8, alpha=0.85)
    ax.axvline(np.mean(objects_per_image), color='#e74c3c', linestyle='--',
               linewidth=2, label=f'Mean = {np.mean(objects_per_image):.1f}')
    ax.axvline(np.median(objects_per_image), color='#f39c12', linestyle='-.',
               linewidth=2, label=f'Median = {np.median(objects_per_image):.1f}')
    ax.set_xlabel('Number of Sperm Cells per Image')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Annotated Objects per Image')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'fig_objects_per_image.pdf')
    fig.savefig(OUT_DIR / 'fig_objects_per_image.png')
    plt.close(fig)
    print("[OK] fig_objects_per_image")


# ════════════════════════════════════════════════════════════════
# Figure 4: Object Size Distribution (Box Plot per Class)
# ════════════════════════════════════════════════════════════════
def fig4_object_sizes(class_areas):
    fig, ax = plt.subplots(figsize=(9, 5))

    data = [np.sqrt(class_areas[c]) for c in CLASS_NAMES]  # sqrt for linear dimension
    bp = ax.boxplot(data, labels=CLASS_NAMES, patch_artist=True, notch=True,
                    medianprops={'color': 'black', 'linewidth': 1.5})
    for patch, color in zip(bp['boxes'], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Morphology Class')
    ax.set_ylabel('Object Size (√area, pixels)')
    ax.set_title('Bounding Box Size Distribution per Class')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'fig_object_sizes.pdf')
    fig.savefig(OUT_DIR / 'fig_object_sizes.png')
    plt.close(fig)
    print("[OK] fig_object_sizes")


# ════════════════════════════════════════════════════════════════
# Figure 5: Patient Contribution
# ════════════════════════════════════════════════════════════════
def fig5_patient_contribution(patient_counts):
    fig, ax = plt.subplots(figsize=(9, 5))

    patients = sorted(patient_counts.keys())
    counts = [patient_counts[p] for p in patients]
    colors_p = plt.cm.Set3(np.linspace(0, 1, len(patients)))

    bars = ax.barh(patients, counts, color=colors_p, edgecolor='white', linewidth=0.8)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{cnt} ({cnt/sum(counts)*100:.1f}%)', va='center', fontsize=9)

    ax.set_xlabel('Number of Images')
    ax.set_ylabel('Patient ID')
    ax.set_title('Image Contribution per Patient')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, max(counts) * 1.3)

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'fig_patient_contribution.pdf')
    fig.savefig(OUT_DIR / 'fig_patient_contribution.png')
    plt.close(fig)
    print("[OK] fig_patient_contribution")


# ════════════════════════════════════════════════════════════════
# Figure 6: Dataset Split Summary Table (as image)
# ════════════════════════════════════════════════════════════════
def fig6_split_table(split_stats):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')

    split_imgs = {'train': 272, 'val': 58, 'test': 59}
    
    cell_text = []
    for split in ['train', 'val', 'test']:
        row = [split.capitalize(), str(split_imgs[split])]
        total = sum(split_stats[split].get(j, 0) for j in range(5))
        for j in range(5):
            cnt = split_stats[split].get(j, 0)
            row.append(f'{cnt}')
        row.append(str(total))
        cell_text.append(row)

    # Totals row
    total_row = ['Total', str(sum(split_imgs.values()))]
    grand_total = 0
    for j in range(5):
        s = sum(split_stats[split].get(j, 0) for split in ['train', 'val', 'test'])
        total_row.append(str(s))
        grand_total += s
    total_row.append(str(grand_total))
    cell_text.append(total_row)

    col_labels = ['Split', 'Images'] + CLASS_NAMES + ['Total']

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#34495e')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Style totals row
    for j in range(len(col_labels)):
        table[len(cell_text), j].set_facecolor('#ecf0f1')
        table[len(cell_text), j].set_text_props(fontweight='bold')

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'fig_split_table.pdf')
    fig.savefig(OUT_DIR / 'fig_split_table.png')
    plt.close(fig)
    print("[OK] fig_split_table")


# ════════════════════════════════════════════════════════════════
# Figure 7: Pipeline Architecture Diagram
# ════════════════════════════════════════════════════════════════
def fig7_pipeline_architecture():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Proposed System Architecture', fontsize=14, fontweight='bold', pad=20)

    def draw_box(x, y, w, h, text, color, fontsize=9, textcolor='white'):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=textcolor,
                multialignment='center')

    def draw_arrow(x1, y1, x2, y2, color='#7f8c8d'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle='->', color=color, lw=2))

    # Input
    draw_box(0.5, 6.2, 3, 1.2, 'Input\nMicroscope Image\n(1280×720)', '#2c3e50')

    # Pipeline A: YOLOv12m
    draw_box(5, 6.2, 3.5, 1.2, 'Pipeline A\nYOLOv12m-seg', '#2980b9', fontsize=10)
    draw_arrow(3.5, 6.8, 5, 6.8)

    draw_box(9.5, 6.8, 2, 0.6, 'Detection +\nClassification', '#27ae60', fontsize=8)
    draw_arrow(8.5, 6.8, 9.5, 7.1)

    draw_box(9.5, 6.0, 2, 0.6, 'Instance\nSegmentation', '#27ae60', fontsize=8)
    draw_arrow(8.5, 6.6, 9.5, 6.3)

    # Pipeline B: DeepLabV3+ → Classifiers
    draw_box(5, 3.5, 3.5, 1.2, 'Pipeline B\nDeepLabV3+', '#8e44ad', fontsize=10)
    draw_arrow(2.0, 6.2, 2.0, 4.7)
    draw_arrow(2.0, 4.7, 5.0, 4.1)

    draw_box(9.5, 4.3, 2, 0.6, 'Sperm\nDetection', '#e67e22', fontsize=8)
    draw_arrow(8.5, 4.1, 9.5, 4.6)

    # Classifiers
    draw_box(9.5, 3.3, 2, 0.6, 'Cropped\nSperm ROIs', '#e67e22', fontsize=8)
    draw_arrow(8.5, 3.9, 9.5, 3.6)

    draw_box(5, 1.5, 3, 0.9, 'EfficientNet-B7\nClassifier', '#c0392b', fontsize=9)
    draw_arrow(10.5, 3.3, 10.5, 2.8)
    draw_arrow(10.5, 2.8, 8.0, 2.0)

    draw_box(9.5, 1.5, 3, 0.9, 'ConvNeXt\nClassifier', '#d35400', fontsize=9)
    draw_arrow(10.5, 2.8, 11.0, 2.4)

    # Final output
    draw_box(5, 0.2, 7.5, 0.8, 'Morphology Classification: DEG | NH | SH | MH | BH', '#2c3e50', fontsize=10)
    draw_arrow(6.5, 1.5, 6.5, 1.0)
    draw_arrow(11.0, 1.5, 11.0, 1.0)
    draw_arrow(12.0, 6.8, 12.5, 6.8)

    # Output box for Pipeline A
    draw_box(12.0, 5.8, 1.5, 1.5, 'Final\nOutput A', '#16a085', fontsize=9)
    draw_arrow(11.5, 6.8, 12.0, 6.8)

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'fig_pipeline_architecture.pdf')
    fig.savefig(OUT_DIR / 'fig_pipeline_architecture.png')
    plt.close(fig)
    print("[OK] fig_pipeline_architecture")


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Gathering statistics...")
    class_counts, objects_per_image, patient_counts, bbox_areas, class_areas = gather_stats()
    split_stats = gather_split_stats()

    print("\nGenerating figures...")
    fig1_class_distribution(class_counts)
    fig2_split_comparison(split_stats)
    fig3_objects_per_image(objects_per_image)
    fig4_object_sizes(class_areas)
    fig5_patient_contribution(patient_counts)
    fig6_split_table(split_stats)
    fig7_pipeline_architecture()

    print(f"\nAll figures saved to: {OUT_DIR.resolve()}")
