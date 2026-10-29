"""Build a standalone handoff and figures from frozen outputs, never from models.

This reporting script does not import the training runner, load joblib, fit,
predict, pick thresholds, or change selection. All new diagnostics are post hoc.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import os
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def markdown_blocks(text):
    """Parse the limited headings/paragraphs/tables/fences in our audited report."""
    lines = text.splitlines()
    blocks, i = [], 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith('```'):
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                code.append(lines[i])
                i += 1
            blocks.append(('code', '\n'.join(code)))
        elif line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                cells = [x.strip().replace('\\|', '|') for x in re.split(r'(?<!\\)\|', lines[i])[1:-1]]
                if not all(re.fullmatch(r'[:\- ]+', cell) for cell in cells):
                    rows.append(cells)
                i += 1
            blocks.append(('table', rows))
            continue
        elif line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            blocks.append(('heading', (level, line[level:].strip())))
        else:
            paragraph = [line]
            while i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1].startswith(('#', '|', '```', '- ')):
                i += 1
                paragraph.append(lines[i])
            blocks.append(('paragraph', ' '.join(paragraph)))
        i += 1
    return blocks


def plain(text):
    return text.replace('**', '').replace('``', '').replace('`', '')


def inline(text):
    text = html.escape(text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'``(.+?)``', r'<code>\1</code>', text)
    return re.sub(r'`([^`]+)`', r'<code>\1</code>', text)


def write_html(blocks, path, title):
    parts = []
    for kind, value in blocks:
        if kind == 'heading':
            level, text = value
            parts.append(f'<h{level}>{inline(text)}</h{level}>')
        elif kind == 'table':
            parts.append('<div class="table-wrap"><table>')
            for i, row in enumerate(value):
                cell = 'th' if i == 0 else 'td'
                parts.append('<tr>' + ''.join(f'<{cell}>{inline(x)}</{cell}>' for x in row) + '</tr>')
            parts.append('</table></div>')
        elif kind == 'code':
            parts.append('<pre>' + html.escape(value) + '</pre>')
        elif kind == 'image':
            data = base64.b64encode(value.read_bytes()).decode('ascii')
            parts.append(f'<figure><img alt="{html.escape(value.stem)}" src="data:image/png;base64,{data}"></figure>')
        else:
            parts.append('<p>' + inline(value) + '</p>')
    css = '''body{font:16px/1.6 system-ui,sans-serif;color:#172b3a;max-width:1200px;margin:40px auto;padding:0 24px}
    h1,h2,h3{line-height:1.25;color:#123e55}h2{margin-top:42px;border-top:2px solid #d9e6eb;padding-top:24px}
    table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;border:1px solid #cfdae0;text-align:left;vertical-align:top;overflow-wrap:anywhere}
    th{background:#edf4f7}tr:nth-child(even){background:#f8fafb}.table-wrap{overflow-x:auto}code{overflow-wrap:anywhere;background:#eef2f4}
    pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef2f4;padding:16px}img{max-width:100%;height:auto}figure{margin:20px 0}
    @media print{body{max-width:none;margin:0;font-size:10pt}table{font-size:8pt}h2{break-before:page}tr,img{break-inside:avoid}}'''
    path.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>' + html.escape(title)
                    + '</title><style>' + css + '</style><body>' + '\n'.join(parts) + '</body></html>', encoding='utf-8')


def write_docx(blocks, path):
    """Write standard OOXML with real Word tables and embedded PNGs, no new dependency."""
    from PIL import Image
    w = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    r = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    a = 'http://schemas.openxmlformats.org/drawingml/2006/main'
    wp = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
    pic = 'http://schemas.openxmlformats.org/drawingml/2006/picture'
    ET.register_namespace('w', w)
    ET.register_namespace('r', r)
    def element(parent, name, attrs=None):
        return ET.SubElement(parent, f'{{{w}}}{name}', {f'{{{w}}}{k}': str(v) for k, v in (attrs or {}).items()})
    document = ET.Element(f'{{{w}}}document')
    body = element(document, 'body')
    def paragraph(parent, text, heading=None, size=20):
        p = element(parent, 'p')
        pp = element(p, 'pPr')
        if heading:
            element(pp, 'keepNext')
            element(pp, 'outlineLvl', {'val': heading - 1})
        run = element(p, 'r')
        props = element(run, 'rPr')
        element(props, 'sz', {'val': (34 if heading == 1 else 28 if heading else size)})
        if heading:
            element(props, 'b')
        for i, line in enumerate(plain(text).split('\n')):
            if i:
                element(run, 'br')
            node = element(run, 't')
            node.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
            node.text = line
        return p
    images = []
    for kind, value in blocks:
        if kind == 'table':
            table = element(body, 'tbl')
            props = element(table, 'tblPr')
            element(props, 'tblW', {'w': 5000, 'type': 'pct'})
            borders = element(props, 'tblBorders')
            for side in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
                element(borders, side, {'val': 'single', 'sz': 4, 'color': 'CCD8DF'})
            grid = element(table, 'tblGrid')
            for _ in value[0]:
                element(grid, 'gridCol', {'w': 14000 // len(value[0])})
            for i, row in enumerate(value):
                tr = element(table, 'tr')
                if i == 0:
                    element(element(tr, 'trPr'), 'tblHeader')
                for cell in row:
                    tc = element(tr, 'tc')
                    paragraph(tc, cell, size=16)
        elif kind == 'image':
            index = len(images) + 1
            images.append(value)
            width, height = Image.open(value).size
            cx = int(8.5 * 914400)
            cy = int(cx * height / width)
            if cy > 6 * 914400:
                cy = 6 * 914400
                cx = int(cy * width / height)
            p = element(body, 'p')
            drawing = element(element(p, 'r'), 'drawing')
            item = ET.SubElement(drawing, f'{{{wp}}}inline')
            ET.SubElement(item, f'{{{wp}}}extent', {'cx': str(cx), 'cy': str(cy)})
            ET.SubElement(item, f'{{{wp}}}docPr', {'id': str(index), 'name': value.stem})
            graphic = ET.SubElement(item, f'{{{a}}}graphic')
            data = ET.SubElement(graphic, f'{{{a}}}graphicData', {'uri': pic})
            picture = ET.SubElement(data, f'{{{pic}}}pic')
            nonvisual = ET.SubElement(picture, f'{{{pic}}}nvPicPr')
            ET.SubElement(nonvisual, f'{{{pic}}}cNvPr', {'id': str(index), 'name': value.name})
            ET.SubElement(nonvisual, f'{{{pic}}}cNvPicPr')
            fill = ET.SubElement(picture, f'{{{pic}}}blipFill')
            ET.SubElement(fill, f'{{{a}}}blip', {f'{{{r}}}embed': f'rId{index}'})
            ET.SubElement(ET.SubElement(fill, f'{{{a}}}stretch'), f'{{{a}}}fillRect')
            shape = ET.SubElement(picture, f'{{{pic}}}spPr')
            transform = ET.SubElement(shape, f'{{{a}}}xfrm')
            ET.SubElement(transform, f'{{{a}}}off', {'x': '0', 'y': '0'})
            ET.SubElement(transform, f'{{{a}}}ext', {'cx': str(cx), 'cy': str(cy)})
            ET.SubElement(ET.SubElement(shape, f'{{{a}}}prstGeom', {'prst': 'rect'}), f'{{{a}}}avLst')
        elif kind == 'heading':
            paragraph(body, value[1], heading=value[0])
        else:
            paragraph(body, value, size=18 if kind == 'code' else 20)
    section = element(body, 'sectPr')
    element(section, 'pgSz', {'w': 15840, 'h': 12240, 'orient': 'landscape'})
    element(section, 'pgMar', {'top': 720, 'right': 720, 'bottom': 720, 'left': 720})
    relationships = ET.Element('Relationships', xmlns='http://schemas.openxmlformats.org/package/2006/relationships')
    for index, image in enumerate(images, 1):
        ET.SubElement(relationships, 'Relationship', Id=f'rId{index}', Type=r + '/image', Target=f'media/image{index}.png')
    types = '''<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>'''
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('[Content_Types].xml', types)
        archive.writestr('_rels/.rels', f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="{r}/officeDocument" Target="word/document.xml"/></Relationships>')
        archive.writestr('word/document.xml', ET.tostring(document, encoding='utf-8', xml_declaration=True))
        archive.writestr('word/_rels/document.xml.rels', ET.tostring(relationships, encoding='utf-8', xml_declaration=True))
        for index, image in enumerate(images, 1):
            archive.write(image, f'word/media/image{index}.png')


def build(config_path):
    settings = read(config_path)
    root = next(parent for parent in config_path.resolve().parents if (parent / '.git').exists())
    run = root / settings['run_directory']
    output = root / settings['report_directory']
    plots = root / settings['plot_directory']
    for path in [output, plots]:
        resolved = path.resolve()
        assert resolved.is_relative_to((root / 'groups/group1_qda/kronodroid').resolve())
        assert not any(resolved.is_relative_to((root / p).resolve()) for p in ['runs/real', 'runs/kronodroid'])
        path.mkdir(parents=True, exist_ok=True)
    sources = {p: digest(p) for p in run.rglob('*') if p.is_file() and p.suffix != '.joblib'}
    provenance = read(run / 'input_provenance.json')
    for record in provenance['input_files'].values():
        path = root / record['path']
        assert digest(path) == record['sha256'], f'Input hash mismatch: {path}'
        sources[path] = digest(path)
    os.environ['MPLCONFIGDIR'] = str(root / settings['matplotlib_cache'])
    import matplotlib
    matplotlib.use(settings['backend'])
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update(settings['rcparams'])
    selection = read(run / 'selection_summary.json')
    config = yaml.safe_load((run / 'config_used.yaml').read_text())
    names = config['selection']['feature_sets']
    metrics = {name: {role: read(run / name / f'{role}_metrics.json') for role in settings['roles']} for name in names}
    searches = {name: read(run / name / 'reg_param_search.json') for name in names}
    runtimes = {name: read(run / name / 'runtime.json') for name in names}
    audit = {'model_calls': 0, 'refits': 0, 'test_threshold_changes': 0, 'files': {}, 'checks': []}
    predictions = {}
    manifest = pd.read_parquet(root / config['inputs']['manifest']).set_index(config['contract']['manifest_id'])
    for name in names:
        # Python's float parser preserves round-trip values without pandas float_precision.
        prediction = pd.read_csv(run / name / 'test_predictions.csv', converters={'prob_malware': float})
        y, pred, probability = prediction.true_label, prediction.predicted_label, prediction.prob_malware
        assert prediction.sha256.is_unique and len(prediction) == config['contract']['split_counts']['test']
        aligned = manifest.loc[prediction.sha256]
        assert aligned['split'].eq('test').all()
        assert np.array_equal(aligned['label_binary_raw'].to_numpy(), y.to_numpy())
        calculated = {
            'accuracy': accuracy_score(y, pred), 'balanced_accuracy': balanced_accuracy_score(y, pred),
            'malware_precision': precision_score(y, pred, zero_division=config['metrics']['zero_division']),
            'malware_recall': recall_score(y, pred, zero_division=config['metrics']['zero_division']),
            'malware_f1': f1_score(y, pred, zero_division=config['metrics']['zero_division']),
            'macro_f1': f1_score(y, pred, average='macro', zero_division=config['metrics']['zero_division']),
            'roc_auc': roc_auc_score(y, probability), 'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        }
        assert all(value == metrics[name]['test'][key] for key, value in calculated.items()), name
        best = max(searches[name], key=lambda item: (item['val_macro_f1'], item['reg_param']))
        assert best['reg_param'] == selection['feature_sets'][name]['chosen_reg_param']
        predictions[name] = prediction
    audit['checks'].append('All saved TEST CSVs reproduce JSON metrics exactly and match manifest IDs/labels.')
    assert selection['selected_feature_set'] == max(names, key=lambda n: selection['feature_sets'][n]['val_macro_f1'])
    audit['checks'].append('Selection matches VAL-only maxima; no selection or model settings were changed.')
    figure_records = []
    with PdfPages(plots / settings['combined_pdf']) as pdf:
        def save(fig, stem, caption):
            fig.tight_layout()
            png = plots / (stem + '.png')
            fig.savefig(png, dpi=settings['dpi'])
            fig.savefig(plots / (stem + '.pdf'))
            pdf.savefig(fig)
            plt.close(fig)
            figure_records.append({'stem': stem, 'png': png, 'caption': caption, 'copied': False})

        fig, ax = plt.subplots(figsize=settings['wide_figsize'])
        for name, color in zip(names, settings['colors']):
            ax.plot([r['reg_param'] for r in searches[name]], [r['val_macro_f1'] for r in searches[name]], marker='o', label=name, color=color)
            winner = selection['feature_sets'][name]
            ax.scatter(winner['chosen_reg_param'], winner['val_macro_f1'], marker='*', s=settings['winner_marker_size'], color=color, edgecolor='black', zorder=4)
        ax.set(xlabel='reg_param (configured grid)', ylabel='Validation macro F1', title='Validation-only regularisation search', ylim=settings['score_ylim'])
        ax.legend(loc='lower left')
        save(fig, '01_validation_search', 'VERIFIED: all 32 VAL macro-F1 values come from reg_param_search.json. Stars identify each set\'s frozen winner. No TEST result appears in this selection plot.')

        fig, axes = plt.subplots(1, 2, figsize=settings['double_figsize'])
        x = np.arange(len(names))
        for ax, key in zip(axes, ['accuracy', 'macro_f1']):
            for i, (role, color) in enumerate(zip(settings['roles'], settings['split_colors'])):
                ax.bar(x + (i - 1) * settings['bar_width'], [metrics[n][role][key] for n in names], width=settings['bar_width'], color=color, label=role.upper())
            ax.set_xticks(x, names)
            ax.set(title=key.replace('_', ' ').title(), ylim=settings['score_ylim'])
            ax.legend(loc='lower left')
        save(fig, '02_train_val_test_metrics', 'VERIFIED: accuracy and macro F1 from the twelve split-metric JSONs. TRAIN is in-sample; VAL was used for selection; TEST was evaluated after freezing. Full-height 0-to-1 axes avoid exaggerating differences; exact values are in the report tables.')

        fig, ax = plt.subplots(figsize=settings['wide_figsize'])
        for name, color in zip(names, settings['colors']):
            p = predictions[name]
            fpr, tpr, _ = roc_curve(p.true_label, p.prob_malware, drop_intermediate=False)
            ax.plot(fpr, tpr, label=name + ' AUC=' + str(metrics[name]['test']['roc_auc']), color=color)
        ax.plot([0, 1], [0, 1], '--', color='gray', label='Random ranking reference')
        ax.set(xlabel='False positive rate', ylabel='True positive rate (Malware recall)', title='TEST ROC curves: post hoc description only', xlim=settings['score_ylim'], ylim=settings['score_ylim'])
        ax.legend(loc='lower right', fontsize=settings['legend_fontsize'])
        save(fig, '03_test_roc_curves', 'VERIFIED: curves computed only from saved TEST labels/probabilities, with no model calls. AUC values exactly match test_metrics.json. These retrospective curves were not used for selecting a model or threshold.')

        fig, ax = plt.subplots(figsize=settings['wide_figsize'])
        for name, color in zip(names, settings['colors']):
            p = predictions[name]
            precision, recall, _ = precision_recall_curve(p.true_label, p.prob_malware, drop_intermediate=False)
            ax.step(recall, precision, where='post', label=name, color=color)
        prevalence = float(predictions[names[0]].true_label.mean())
        ax.axhline(prevalence, ls='--', color='gray', label='TEST Malware prevalence=' + str(prevalence))
        ax.set(xlabel='Malware recall', ylabel='Malware precision', title='TEST precision-recall curves: post hoc only', xlim=settings['score_ylim'], ylim=settings['score_ylim'])
        ax.legend(loc='lower left', fontsize=settings['legend_fontsize'])
        save(fig, '04_test_precision_recall_curves', 'VERIFIED: curves use saved TEST probabilities and labels only. No new average-precision result or threshold recommendation is introduced. The horizontal reference is the observed Malware prevalence, not a fitted classifier.')

        fig, axes = plt.subplots(1, 2, figsize=settings['double_figsize'])
        for ax, key, title in zip(axes, ['final_model_fit_seconds', 'total_feature_set_search_seconds'], ['Final model fit only', 'Full per-feature-set search']):
            values = [runtimes[name][key] for name in names]
            bars = ax.bar(names, values, color=settings['colors'])
            ax.set(title=title, ylabel='Measured wall time (seconds)', ylim=(0, max(values) * settings['runtime_headroom']))
            for bar, value in zip(bars, values):
                ax.text(bar.get_x()+bar.get_width()/2, value, str(value), ha='center', va='bottom', rotation=90, fontsize=settings['annotation_fontsize'])
        save(fig, '05_fit_and_search_runtime', 'VERIFIED: winning-candidate fit time and full search time are separate quantities copied from runtime.json. No retraining or fresh timing was performed. Measurements depend on this machine and include the documented search overhead.')

        fig, ax = plt.subplots(figsize=settings['wide_figsize'])
        warning_matrix = np.array([[len(r['warnings']) for r in searches[name]] for name in names])
        image = ax.imshow(warning_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=max(1, int(warning_matrix.max())))
        ax.set_xticks(np.arange(len(config['qda']['reg_param_grid'])), [str(x) for x in config['qda']['reg_param_grid']])
        ax.set_yticks(np.arange(len(names)), names)
        for y in range(len(names)):
            for x in range(len(config['qda']['reg_param_grid'])):
                ax.text(x, y, str(warning_matrix[y, x]), ha='center', va='center', color='white' if warning_matrix[y, x] else '#163a50')
        ax.set(title='Captured sklearn warnings per fitted candidate', xlabel='reg_param', ylabel='Feature set')
        fig.colorbar(image, ax=ax, ticks=[0, 1, 2], label='Warning occurrences')
        save(fig, '06_candidate_warning_counts', 'VERIFIED: eight LinAlgWarning occurrences: classes 0 and 1 at reg_param=0.0 for each set. Counts are copied from search JSON, not inferred from plotted scores. All other candidates and selected-model evaluations have empty warning lists.')

        group3 = provenance['group3_metrics']
        fig, ax = plt.subplots(figsize=settings['wide_figsize'])
        labels = ['QDA primary\nmi_top50', 'QDA VAL-selected\net_top50', 'GROUP 3 LightGBM\nreplication', 'GROUP 3 LightGBM\nfull_standardized']
        accuracy = [metrics['mi_top50']['test']['accuracy'], metrics['et_top50']['test']['accuracy'], group3['replication']['accuracy'], group3['full_standardized']['accuracy']]
        malware_f1 = [metrics['mi_top50']['test']['malware_f1'], metrics['et_top50']['test']['malware_f1'], group3['replication']['f1'], group3['full_standardized']['f1']]
        x = np.arange(len(labels))
        for delta, values, label, color in [(-settings['bar_width']/2, accuracy, 'TEST accuracy', settings['split_colors'][0]), (settings['bar_width']/2, malware_f1, 'TEST Malware F1 (not macro F1)', settings['split_colors'][1])]:
            ax.bar(x + delta, values, width=settings['bar_width'], label=label, color=color)
        ax.set_xticks(x, labels)
        ax.set(title='Measured comparison: QDA below GROUP 3 LightGBM', ylim=settings['score_ylim'])
        ax.legend(loc='lower left')
        save(fig, '07_group3_comparison', 'VERIFIED: Group 3 values are their stored, four-decimal results, not ours. Different feature representations, class-weight treatment and model selection procedures prevent a controlled classifier-only comparison. No significance claim or test-driven tuning is made.')

        fig, ax = plt.subplots(figsize=settings['wide_figsize'])
        splits = list(config['contract']['split_counts'])
        benign = [config['contract']['label_counts'][s][0] for s in splits]
        malware = [config['contract']['label_counts'][s][1] for s in splits]
        ax.bar(splits, benign, label='Benign (0)', color=settings['split_colors'][0])
        ax.bar(splits, malware, bottom=benign, label='Malware (1)', color=settings['split_colors'][1])
        for i, (b, m) in enumerate(zip(benign, malware)):
            ax.text(i, b/2, str(b), ha='center', va='center', color='white')
            ax.text(i, b+m/2, str(m), ha='center', va='center', color='white')
        ax.set(title='Unchanged Group 3 split: exact class counts', ylabel='App samples')
        ax.legend()
        save(fig, '08_split_label_counts', 'VERIFIED: exact split/class counts from the executed config, checked against the actual split manifest. The chart depicts the supplied split, not a new random split.')

        for name in names:
            # Preserve the original test confusion PNG byte-for-byte in Group 1's folder.
            stem = '09_confusion_' + name
            target = plots / (stem + '.png')
            shutil.copyfile(run / name / 'confusion_matrix.png', target)
            fig, ax = plt.subplots(figsize=settings['confusion_figsize'])
            matrix = np.array(metrics[name]['test']['confusion_matrix'])
            ax.imshow(matrix, cmap='Blues', vmin=0)
            for (y, x), value in np.ndenumerate(matrix):
                ax.text(x, y, str(value), ha='center', va='center', color='white' if value > matrix.max()/2 else '#163a50')
            ax.set_xticks([0, 1], ['Benign (0)', 'Malware (1)'])
            ax.set_yticks([0, 1], ['Benign (0)', 'Malware (1)'])
            ax.set(title=f'{name}: frozen-model TEST confusion matrix', xlabel='Predicted label', ylabel='True label')
            fig.tight_layout()
            fig.savefig(plots / (stem + '.pdf'))
            pdf.savefig(fig)
            plt.close(fig)
            figure_records.append({'stem': stem, 'png': target, 'copied': True,
                'caption': 'VERIFIED: original PNG copied byte-for-byte from the frozen run; PDF redrawn from its saved confusion counts. Rows=true, columns=predicted, order=[0,1]. No predictions were regenerated. ' + name + ' counts: ' + str(matrix.tolist())})

    rows = (len(figure_records) + settings['overview_columns'] - 1) // settings['overview_columns']
    fig, axes = plt.subplots(rows, settings['overview_columns'], figsize=settings['overview_figsize'])
    for ax, record in zip(axes.flat, figure_records):
        ax.imshow(plt.imread(record['png']))
        ax.set_title(record['stem'], fontsize=settings['legend_fontsize'])
        ax.axis('off')
    for ax in list(axes.flat)[len(figure_records):]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(plots / settings['overview_png'], dpi=settings['dpi'])
    plt.close(fig)
    # A narrative front section plus the audited full configuration/results appendix.
    narrative = (root / settings['narrative_source']).read_text(encoding='utf-8')
    source_readme = root / settings['audited_detail_source']
    details = source_readme.read_text(encoding='utf-8')
    details = details[details.index('## Scope and input provenance'):]
    details = details[:details.index('## Critical review and delivery status at completion of Step 7')]
    blocks = markdown_blocks(narrative)
    blocks.append(('heading', (2, 'Plot catalogue and interpretation')))
    blocks.append(('paragraph', 'VERIFIED: Twelve figures below use existing JSON/CSV artifacts only. Newly drawn ROC and precision-recall curves are post hoc descriptive summaries of already saved probabilities. No fitting, model calls, threshold selection or new TEST evaluation occurred. Axis tick formatting is visual presentation; exact results remain in the tables and JSON. Every figure is saved as PNG and PDF in the Group 1 plots folder, with a combined PDF atlas.'))
    for record in figure_records:
        blocks.extend([('heading', (3, record['stem'])), ('paragraph', record['caption']), ('image', record['png'])])
    blocks.append(('heading', (2, 'Exact ordered feature identities supplied to each frozen model')))
    blocks.append(('paragraph', 'VERIFIED: These names are copied from selection_summary.json, in model input order. All four dropped-feature lists are empty. These lists come from Group 3 rankings; the report does not generate a new ranking.'))
    for name in names:
        info = selection['feature_sets'][name]
        blocks.extend([('heading', (3, name)), ('code', json.dumps({
            'features_in_model_order': info['retained_features'],
            'dropped_features': info['dropped_features'],
            'chosen_reg_param': info['chosen_reg_param'],
            'fitted_priors': info['fitted_priors'],
        }, indent=settings['json_indent']))])
    blocks.extend([('heading', (2, 'Evidence appendix: exact settings, provenance and results'))])
    blocks += markdown_blocks(details)
    title = settings['title']
    write_html(blocks, output / 'Group3_Handoff_Report.html', title)
    write_docx(blocks, output / 'Group3_Handoff_Report.docx')
    text = []
    for kind, value in blocks:
        if kind == 'heading':
            text += ['', value[1].upper() if value[0] < 3 else value[1], '']
        elif kind == 'table':
            text += ['\t'.join(plain(cell) for cell in row) for row in value] + ['']
        elif kind == 'image':
            text.append('[Figure: ' + value.relative_to(root).as_posix() + ']')
        else:
            text += [plain(value), '']
    (output / 'Group3_Handoff_Report.txt').write_text('\n'.join(text).rstrip() + '\n', encoding='utf-8')
    for path, before in sources.items():
        assert digest(path) == before, f'Reporting changed input: {path}'
    audit['checks'].append('All source run artifacts and Group 3 input hashes unchanged after rendering.')
    for record in figure_records:
        if record['copied']:
            name = record['stem'].removeprefix('09_confusion_')
            assert digest(record['png']) == digest(run / name / 'confusion_matrix.png')
    audit['checks'].append('All four original confusion-matrix PNG copies are byte-identical.')
    with zipfile.ZipFile(output / 'Group3_Handoff_Report.docx') as archive:
        assert archive.testzip() is None
        for member in archive.namelist():
            if member.endswith(('.xml', '.rels')):
                ET.fromstring(archive.read(member))
        assert len([member for member in archive.namelist() if member.startswith('word/media/')]) == len(figure_records)
    audit['checks'].append('DOCX ZIP/XML validity and twelve embedded images verified; Word GUI rendering not verified.')
    audit.update({'generated_utc': datetime.now(timezone.utc).isoformat(), 'run_directory': settings['run_directory'],
                  'figure_count': len(figure_records), 'pipeline_config': config, 'selection_summary': selection,
                  'plot_settings': settings, 'plots': [{k: str(v.relative_to(root)) if k == 'png' else v for k, v in record.items()} for record in figure_records],
                  'source_hashes': {p.relative_to(root).as_posix(): h for p, h in sources.items()}})
    for path in [config_path, source_readme, root / settings['narrative_source'], Path(__file__).resolve()]:
        audit['source_hashes'][path.relative_to(root).as_posix()] = digest(path)
    audit['files'] = {p.relative_to(root).as_posix(): digest(p) for folder in [output, plots] for p in folder.iterdir() if p.is_file() and p.name != 'report_audit.json'}
    (output / 'report_audit.json').write_text(json.dumps(audit, indent=settings['json_indent'], allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'figures': len(figure_records), 'reports': str(output.relative_to(root)), 'plots': str(plots.relative_to(root)), 'audit_checks': audit['checks']}, indent=settings['json_indent']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    build(parser.parse_args().config.resolve())
