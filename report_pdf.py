"""
report_pdf.py — Membuat 4 laporan hasil analisis sentimen dalam format PDF,
siap dicetak, hitam-putih, lengkap dengan kop surat, tanggal, dan kolom
tanda tangan.

4 Laporan:
  1. Laporan Dataset
  2. Laporan Hasil Training & Evaluasi Model
  3. Laporan Confusion Matrix
  4. Laporan Analisis Kata
"""

import os
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    HRFlowable, Flowable,
)
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
from reportlab.graphics.shapes import Drawing, Circle, String
from reportlab.graphics import renderPDF

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, 'storage')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

BLACK = colors.black
GRAY = colors.HexColor('#555555')
LINE_GRAY = colors.HexColor('#999999')

BULAN_ID = [
    'Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
    'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember',
]


def tanggal_indonesia(dt):
    return f'{dt.day} {BULAN_ID[dt.month - 1]} {dt.year}'


# =====================================================================
# KOP SURAT (letterhead) — monogram "RM" + identitas sistem, hitam-putih
# =====================================================================
class MonogramRM(Flowable):
    """Lingkaran monogram 'RM' sederhana, digambar vektor (bukan logo asli
    Ramayana) — dipakai sebagai elemen identitas kop surat."""

    def __init__(self, size=52):
        Flowable.__init__(self)
        self.size = size
        self.width = size
        self.height = size

    def draw(self):
        d = Drawing(self.size, self.size)
        r = self.size / 2
        d.add(Circle(r, r, r - 1.5, strokeColor=BLACK, strokeWidth=1.3, fillColor=None))
        d.add(String(r, r - 6, 'RM', fontName='Helvetica-Bold', fontSize=15,
                      fillColor=BLACK, textAnchor='middle'))
        renderPDF.draw(d, self.canv, 0, 0)


def kop_surat(styles):
    title_style = ParagraphStyle('KopTitle', parent=styles['Normal'],
                                  fontName='Helvetica-Bold', fontSize=14,
                                  textColor=BLACK, leading=17)
    sub_style = ParagraphStyle('KopSub', parent=styles['Normal'],
                                fontSize=10, textColor=GRAY, leading=13)

    logo_path = os.path.join(STATIC_DIR, 'logo-ramayana.png')
    if os.path.exists(logo_path):
        logo_img = Image(logo_path)
        logo_img.drawWidth = 80
        logo_img.drawHeight = 52
    else:
        logo_img = MonogramRM(52)

    header_table = Table(
        [[
            logo_img,
            [
                Paragraph('Sistem Analisis Sentimen Ramayana', title_style),
                Paragraph('Sistem Analisis Sentimen Ulasan Google Maps', sub_style),
                Paragraph('Ramayana Pasar Minggu, Jakarta Selatan', sub_style),
            ],
        ]],
        colWidths=[70, 9280 / 20],
    )
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (0, 0), 0),
        ('LEFTPADDING', (1, 0), (1, 0), 12),
    ]))

    elements = [
        header_table,
        Spacer(1, 8),
        HRFlowable(width='100%', color=BLACK, thickness=1.4, spaceAfter=2),
        HRFlowable(width='100%', color=BLACK, thickness=0.5, spaceAfter=14),
    ]
    return elements


# =====================================================================
# Helper umum
# =====================================================================
def _table_style(header=False):
    cmds = [
        ('FONTSIZE', (0, 0), (-1, -1), 9.5),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LINEBELOW', (0, 0), (-1, -1), 0.5, LINE_GRAY),
        ('TEXTCOLOR', (0, 0), (0, -1), GRAY),
        ('GRID', (0, 0), (-1, -1), 0.4, LINE_GRAY) if header else (None,),
    ]
    cmds = [c for c in cmds if c[0] is not None]
    if header:
        cmds += [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E8E8E8')),
            ('TEXTCOLOR', (0, 0), (-1, 0), BLACK),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ]
    return TableStyle(cmds)


def _add_image(story, relative_path, width, styles):
    if not relative_path:
        story.append(Paragraph('(grafik tidak tersedia)', styles['Normal']))
        return
    full_path = os.path.join(STATIC_DIR, relative_path)
    if not os.path.exists(full_path):
        story.append(Paragraph('(grafik tidak tersedia)', styles['Normal']))
        return
    img = Image(full_path)
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width
    img.drawHeight = width * ratio
    story.append(img)
    story.append(Spacer(1, 10))


def _signature_block(styles, penyusun):
    right_style = ParagraphStyle('Right', parent=styles['Normal'], alignment=TA_RIGHT, fontSize=10)
    tanggal = tanggal_indonesia(datetime.now())
    return [
        Spacer(1, 26),
        Paragraph(f'Jakarta, {tanggal}', right_style),
        Spacer(1, 40),
        Paragraph('( ' + '_' * 30 + ' )', right_style),
        Paragraph(penyusun, right_style),
        Paragraph('Penyusun', right_style),
    ]


def _base_doc(filename):
    out_path = os.path.join(STORAGE_DIR, filename)
    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        leftMargin=2 * cm, rightMargin=2 * cm,
    )
    return doc, out_path


def _get_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('ReportTitle', parent=styles['Title'],
                               textColor=BLACK, fontSize=16, spaceAfter=4,
                               alignment=TA_LEFT))
    styles.add(ParagraphStyle('H2Black', parent=styles['Heading2'],
                               textColor=BLACK, fontSize=12.5, spaceBefore=14, spaceAfter=8))
    return styles


# =====================================================================
# 1. LAPORAN DATASET
# =====================================================================
def build_report_dataset(metrics: dict, penyusun: str = '-') -> str:
    styles = _get_styles()
    doc, out_path = _base_doc('laporan_1_dataset.pdf')
    story = kop_surat(styles)

    story.append(Paragraph('Laporan Dataset', styles['ReportTitle']))
    story.append(Spacer(1, 10))

    distribusi = metrics.get('distribusi_label', {})
    data_ringkasan = [
        ['Total Data Terpakai', str(metrics.get('total_data', '-'))],
        ['Data Training', str(metrics.get('jumlah_train', '-'))],
        ['Data Testing', str(metrics.get('jumlah_test', '-'))],
        ['Data Training (setelah oversampling)', str(metrics.get('jumlah_train_balanced', '-'))],
    ]
    for label in ['Positif', 'Netral', 'Negatif']:
        data_ringkasan.append([f'Jumlah Label {label}', str(distribusi.get(label, 0))])

    story.append(Paragraph('Ringkasan Dataset', styles['H2Black']))
    t = Table(data_ringkasan, colWidths=[9 * cm, 6 * cm])
    t.setStyle(_table_style())
    story.append(t)
    story.append(Spacer(1, 14))

    story.append(Paragraph('Distribusi Sentimen', styles['H2Black']))
    _add_image(story, metrics.get('chart_distribusi'), 11 * cm, styles)

    story.append(Paragraph('Oversampling: Sebelum vs Sesudah', styles['H2Black']))
    _add_image(story, metrics.get('chart_oversampling'), 11 * cm, styles)

    story.extend(_signature_block(styles, penyusun))
    doc.build(story)
    return out_path


# =====================================================================
# 2. LAPORAN HASIL TRAINING & EVALUASI MODEL
# =====================================================================
def build_report_evaluasi(metrics: dict, penyusun: str = '-') -> str:
    styles = _get_styles()
    doc, out_path = _base_doc('laporan_2_evaluasi.pdf')
    story = kop_surat(styles)

    story.append(Paragraph('Laporan Hasil Training & Evaluasi Model', styles['ReportTitle']))
    story.append(Spacer(1, 10))

    spec = metrics.get('model_spec', {})
    story.append(Paragraph('Spesifikasi Model', styles['H2Black']))
    data_spec = [
        ['Algoritma', spec.get('algoritma', '-')],
        ['Ekstraksi Fitur', spec.get('ekstraksi_fitur', '-')],
        ['Balancing', spec.get('balancing', '-')],
        ['Split', spec.get('split', '-')],
    ]
    t = Table(data_spec, colWidths=[5 * cm, 10 * cm])
    t.setStyle(_table_style())
    story.append(t)
    story.append(Spacer(1, 14))

    story.append(Paragraph('Hasil Evaluasi', styles['H2Black']))
    data_eval = [
        ['Akurasi', f"{metrics.get('akurasi', 0) * 100:.2f}%"],
        ['Precision (weighted)', f"{metrics.get('precision', 0) * 100:.2f}%"],
        ['Recall (weighted)', f"{metrics.get('recall', 0) * 100:.2f}%"],
        ['F1-Score (weighted)', f"{metrics.get('f1', 0) * 100:.2f}%"],
    ]
    t = Table(data_eval, colWidths=[9 * cm, 6 * cm])
    t.setStyle(_table_style())
    story.append(t)
    story.append(Spacer(1, 14))

    story.append(Paragraph('Classification Report', styles['H2Black']))
    report = metrics.get('report', {})
    header = ['Kelas', 'Precision', 'Recall', 'F1-Score', 'Support']
    rows = [header]
    for label in ['Negatif', 'Netral', 'Positif']:
        r = report.get(label)
        if r:
            rows.append([label, f"{r['precision']:.2f}", f"{r['recall']:.2f}",
                         f"{r['f1-score']:.2f}", str(int(r['support']))])
    for key in ['macro avg', 'weighted avg']:
        r = report.get(key)
        if r:
            rows.append([key.title(), f"{r['precision']:.2f}", f"{r['recall']:.2f}",
                         f"{r['f1-score']:.2f}", str(int(r['support']))])
    t = Table(rows, colWidths=[4 * cm, 2.75 * cm, 2.75 * cm, 2.75 * cm, 2.75 * cm])
    t.setStyle(_table_style(header=True))
    story.append(t)

    story.extend(_signature_block(styles, penyusun))
    doc.build(story)
    return out_path


# =====================================================================
# 3. LAPORAN CONFUSION MATRIX
# =====================================================================
def build_report_confusion(metrics: dict, penyusun: str = '-') -> str:
    styles = _get_styles()
    doc, out_path = _base_doc('laporan_3_confusion.pdf')
    story = kop_surat(styles)

    story.append(Paragraph('Laporan Confusion Matrix', styles['ReportTitle']))
    story.append(Spacer(1, 10))

    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=10, leading=15)
    story.append(Paragraph(
        'Confusion matrix menunjukkan perbandingan antara label aktual dan label '
        'hasil prediksi model pada data uji (20% data yang tidak digunakan saat '
        'training). Sumbu vertikal menunjukkan label aktual, sumbu horizontal '
        'menunjukkan label hasil prediksi.', body_style))
    story.append(Spacer(1, 12))

    _add_image(story, metrics.get('chart_confusion'), 10 * cm, styles)

    story.extend(_signature_block(styles, penyusun))
    doc.build(story)
    return out_path


# =====================================================================
# 4. LAPORAN ANALISIS KATA
# =====================================================================
def build_report_kata(metrics: dict, penyusun: str = '-') -> str:
    styles = _get_styles()
    doc, out_path = _base_doc('laporan_4_kata.pdf')
    story = kop_surat(styles)

    story.append(Paragraph('Laporan Analisis Kata', styles['ReportTitle']))
    story.append(Spacer(1, 10))

    story.append(Paragraph('Top 20 Kata Paling Sering Muncul', styles['H2Black']))
    _add_image(story, metrics.get('chart_top_words'), 11 * cm, styles)

    wc = metrics.get('chart_wordclouds', {})
    if wc:
        story.append(Paragraph('Word Cloud per Kelas Sentimen', styles['H2Black']))
        for label in ['Positif', 'Netral', 'Negatif']:
            if label in wc:
                story.append(Paragraph(label, ParagraphStyle(
                    'WcLabel', parent=styles['Normal'], fontSize=10,
                    fontName='Helvetica-Bold', spaceBefore=6)))
                _add_image(story, wc[label], 9 * cm, styles)

    story.extend(_signature_block(styles, penyusun))
    doc.build(story)
    return out_path
